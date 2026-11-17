"""
Read-only SQL over the cleaned tables, for the question-answering agent.

Each call names the CSV files to query (this session's cleaned tables in
output/clean/ and its Ask-tab uploads in data/uploads/); each becomes a table
(expenses.csv -> expenses) in an in-memory DuckDB database. Other files in
those folders, e.g. tables from tests or earlier sessions, are never loaded.
The model sends one SQL query; this module runs it and returns the columns
and the first rows. Nothing it runs can change a file.

Why DuckDB: it reads CSVs directly and guesses real column types (DATE,
DOUBLE, BIGINT), speaks analytical SQL (window functions, date_trunc), and
runs inside this process, with no database server to manage. SQLite would
store dates and decimals as text; pandas isn't SQL.

Read-only comes from four layers, so a gap in one is covered by another:
  1. run() accepts exactly ONE statement, and only a SELECT (WITH, DESCRIBE,
     SHOW and SUMMARIZE parse as SELECT too). INSERT, COPY, ATTACH, SET,
     PRAGMA, INSTALL... are refused before DuckDB runs anything.
  2. After the tables are loaded, file access is switched off
     (enable_external_access=false) and the settings are locked, so even a
     SELECT can't read_csv('/etc/passwd'), and nothing can switch it back on.
  3. The tables are in-memory COPIES: the CSV files are never opened for
     writing, so even a write that got through would change only the copy.
  4. The executor container: read-only filesystem, no internet, no
     capabilities (compose.yaml).

This runs in the FastAPI server, not in the Jupyter kernel, so a query never
touches the cleaner's kernel or its tables.
"""

import datetime as dt
import decimal
import math
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import duckdb

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", Path(__file__).resolve().parent.parent / "output"))
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

MAX_ROWS = 200          # rows returned per query; the rest are only counted
CACHE_SIZE = 4          # file sets kept loaded: both tabs, plus a spare or two
COUNT_LIMIT = 1_000_000  # stop counting there: the exact total isn't worth the memory
TIMEOUT = 30            # seconds per query

# Applied after the tables are loaded, in this order. lock_configuration must
# come last: after it, no SET can undo the others.
LOCKDOWN = [
    "SET memory_limit = '1GB'",          # well inside the container's 4 GB
    "SET threads = 2",                   # the container has 2 CPUs
    "SET autoinstall_known_extensions = false",
    "SET autoload_known_extensions = false",
    "SET enable_external_access = false",  # no files, no network, no ATTACH
    "SET lock_configuration = true",
]


class SqlEngine:
    """Tables from an explicit list of CSV files, in an in-memory DuckDB
    database. The caller (the UI's session) says which files belong to it;
    nothing else in the folders is ever loaded, so one session never sees
    another's tables. One query at a time (a lock), each with a timeout.

    Only files inside output/clean/ (cleaned tables) and data/uploads/ (this
    app's uploads) may be listed. Anything else, e.g. /etc/passwd or
    output/clean/../x.csv, is reported in "problems" and never opened."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, data_dir: Path = DATA_DIR):
        self.roots = [Path(output_dir).resolve() / "clean", Path(data_dir).resolve() / "uploads"]
        # signature -> (connection, {table: file}, load problems). Most
        # recently used last. Each tab sends its own file list, so keeping a
        # few sets loaded means switching tabs doesn't reload every time.
        self.cache: OrderedDict[tuple, tuple] = OrderedDict()
        # DuckDB connections aren't meant to be shared by threads at once, and
        # FastAPI runs each request on its own thread: one call at a time.
        self.lock = threading.Lock()

    # --- Public -----------------------------------------------------------------

    def schema(self, files=()) -> dict:
        """Each listed table's columns, types and row count, for the model's
        prompt and the UI's table list. Plus files that weren't loaded, and why."""
        with self.lock:
            paths, problems = self._resolve(files)
            con, sources, load_problems = self._connection(paths)
            tables = []
            for name in sorted(sources):
                columns = con.execute(f'DESCRIBE "{name}"').fetchall()
                rows = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                tables.append({"name": name, "file": sources[name], "rows": rows,
                               "columns": [{"name": c[0], "type": c[1]} for c in columns]})
            return {"tables": tables, "problems": problems + load_problems}

    def run(self, query: str, files=(), max_rows: int = MAX_ROWS, timeout: float = TIMEOUT) -> dict:
        """Run one read-only query on the listed files. Never raises: every
        outcome is a result.

        status is "ok", "error" (DuckDB rejected or failed the query; its
        message is in error, e.g. a misspelt column with the right name
        suggested), "refused" (not a single SELECT) or "timeout"."""
        refusal = check_read_only(query)
        if refusal:
            return _result("refused", error=refusal)
        with self.lock:
            con = self._connection(self._resolve(files)[0])[0]
            # Interrupt from a timer thread: DuckDB checks for interrupts while
            # it works, so even a huge join stops within moments.
            timer = threading.Timer(timeout, con.interrupt)
            timer.start()
            try:
                cursor = con.execute(query)
                columns = [d[0] for d in cursor.description or []]
                rows = [[_plain(v) for v in row] for row in cursor.fetchmany(max_rows)]
                count, exact = len(rows), True
                # Count the rest without keeping it, so a SELECT of a million
                # rows doesn't hold a million rows in memory.
                while True:
                    chunk = cursor.fetchmany(10_000)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count >= COUNT_LIMIT:
                        exact = False
                        break
            except duckdb.InterruptException:
                return _result("timeout", error=f"The query was stopped after {timeout:.0f} seconds. "
                                                "Simplify it, e.g. filter or aggregate before joining.")
            except duckdb.Error as e:
                if not timer.is_alive():   # the timer fired: some interrupts surface as other errors
                    return _result("timeout", error=f"The query was stopped after {timeout:.0f} seconds.")
                return _result("error", error=f"{type(e).__name__}: {e}".strip())
            finally:
                timer.cancel()
            return _result("ok", columns=columns, rows=rows, row_count=count,
                           count_exact=exact, truncated=count > len(rows))

    # --- Loading ------------------------------------------------------------------

    def _resolve(self, files) -> tuple[list[tuple[str, Path]], list[str]]:
        """The listed files that may be loaded, as (name given, real path),
        and a problem line for each one that may not.

        resolve() follows "..", and links, to the real location before the
        check, so neither output/clean/../x.csv nor a link planted in an
        allowed folder can point the engine anywhere else."""
        allowed, problems = [], []
        for f in files:
            path = Path(f).resolve()
            if path.suffix.lower() != ".csv" or not any(path.is_relative_to(r) for r in self.roots):
                problems.append(f"{f}: not allowed (only cleaned files and this session's uploads)")
            elif not path.is_file():
                problems.append(f"{f}: not found")
            else:
                allowed.append((str(f), path))
        return allowed, problems

    def _connection(self, paths: list[tuple[str, Path]]):
        """(connection, {table: file}, problems) for these files, loaded once
        and reused while no file changes. Cheap when cached: one stat per file."""
        signature = tuple((given, str(p), p.stat().st_mtime_ns, p.stat().st_size) for given, p in paths)
        if signature in self.cache:
            self.cache.move_to_end(signature)
            return self.cache[signature]
        # A brand-new database for each set: a locked configuration can't be
        # unlocked, so the only way to load other files is to start over.
        con = duckdb.connect(":memory:")
        sources, problems = {}, []
        for given, p in paths:
            name = table_name(p.stem)
            if name in sources:
                # The later file wins: the caller's order decides (the Ask
                # tab lists its uploads last, so an upload replaces a
                # cleaned table of the same name).
                problems.append(f"{sources[name]}: replaced by {given} (same table name '{name}')")
                con.execute(f'DROP TABLE "{name}"')
                del sources[name]
            path = str(p).replace("'", "''")
            try:
                con.execute(f"CREATE TABLE \"{name}\" AS SELECT * FROM read_csv('{path}')")
                sources[name] = given
            except duckdb.Error as e:
                problems.append(f"{given}: could not be read ({str(e).splitlines()[0]})")
        for setting in LOCKDOWN:
            con.execute(setting)
        self.cache[signature] = (con, sources, problems)
        while len(self.cache) > CACHE_SIZE:
            old, _, _ = self.cache.popitem(last=False)[1]
            old.close()
        return self.cache[signature]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_read_only(query: str) -> str:
    """Why this query is refused, or "" if it's one SELECT.

    DuckDB's own parser does the work (extract_statements), so this is never
    fooled by keywords inside strings or comments the way a regex would be."""
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error:
        return ""   # unparseable: let DuckDB run it and report the syntax error itself
    if not statements:
        return "Refused: the query is empty. Send one SELECT statement."
    if len(statements) > 1:
        return (f"Refused: {len(statements)} statements. Send exactly ONE SELECT per query "
                "(no semicolons between statements).")
    kind = statements[0].type
    if kind != duckdb.StatementType.SELECT:
        return (f"Refused: this is a {kind.name} statement. The tables are read-only: only "
                "SELECT queries (including WITH, DESCRIBE, SHOW, SUMMARIZE) can run.")
    return ""


def table_name(stem: str) -> str:
    """'my orders-2026' -> 'my_orders_2026': a name that needs no quotes in SQL."""
    name = re.sub(r"\W+", "_", stem).strip("_").lower() or "table"
    return f"t_{name}" if name[0].isdigit() else name


def _plain(value):
    """One cell as a JSON-safe value: dates as ISO text, decimals as floats,
    infinity and NaN as text (JSON has no such numbers)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, decimal.Decimal):
        return _plain(float(value))
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    return str(value)


def _result(status: str, **fields) -> dict:
    """Every result has the same keys, whatever happened."""
    return {"status": status, "columns": [], "rows": [], "row_count": 0, "count_exact": True,
            "truncated": False, "error": None} | fields
