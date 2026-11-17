r"""
Checks for sql.py: does the read-only SQL engine answer SELECT queries, and
refuse everything that could change data or touch a file?

Run inside the sandbox image, from the repo root:
    docker compose run --rm --no-deps -v "${PWD}\executor\test_sql.py:/app/test_sql.py:ro" -v "${PWD}\executor\sql.py:/app/sql.py:ro" executor python test_sql.py
"""

import os
import tempfile
import time
from pathlib import Path

# Throwaway output/ and data/ folders, never your real ones. Only files in
# OUT/clean/ and DATA/uploads/ may be loaded, and only the ones listed.
OUT, DATA = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
FOLDER = OUT / "clean"
UPLOADS = DATA / "uploads" / "s1" / "ask"
FOLDER.mkdir()
UPLOADS.mkdir(parents=True)

from sql import SqlEngine   # noqa: E402

PASSED, FAILED = 0, 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global PASSED, FAILED
    passed = bool(passed)
    PASSED, FAILED = PASSED + passed, FAILED + (not passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


(FOLDER / "employees.csv").write_text(
    "employee_id,name,department,salary\n"
    "E001,Ava,Data,90000\n"
    "E002,Liam,Data,85000\n"
    "E003,Noah,Cloud,\n"
    "E004,Mia,Cloud,70000\n")
(FOLDER / "expenses.csv").write_text(
    "expense_id,employee_id,amount,date\n"
    "1,E001,100.50,2026-01-05\n"
    "2,E001,200.00,2026-02-10\n"
    "3,E002,50.25,2026-01-20\n"
    "4,E004,75.00,2026-03-01\n")
(FOLDER / "change_log.json").write_text("[]")   # not a CSV: never a table

EMP, EXP = str(FOLDER / "employees.csv"), str(FOLDER / "expenses.csv")
FILES = [EMP, EXP]


class Engine(SqlEngine):
    """The engine with FILES as the default list, so the older checks below
    read as before. Every call still goes through the files argument."""
    def run(self, query, files=None, **kw):
        return super().run(query, FILES if files is None else files, **kw)

    def schema(self, files=None):
        return super().schema(FILES if files is None else files)


engine = Engine(OUT, DATA)

# --- Schema ------------------------------------------------------------------
schema = engine.schema()
names = [t["name"] for t in schema["tables"]]
check("schema: one table per CSV, sorted", names == ["employees", "expenses"], str(names))
emp = next(t for t in schema["tables"] if t["name"] == "employees")
types = {c["name"]: c["type"] for c in emp["columns"]}
check("schema: row count", emp["rows"] == 4, str(emp["rows"]))
check("schema: real types (text id, number salary)",
      types.get("employee_id") == "VARCHAR" and types.get("salary") in ("BIGINT", "INTEGER", "DOUBLE"), str(types))
exp_types = {c["name"]: c["type"] for c in schema["tables"][1]["columns"]}
check("schema: dates are DATE", exp_types.get("date") == "DATE", str(exp_types))

# --- Answers -----------------------------------------------------------------
r = engine.run("SELECT e.name, ROUND(SUM(x.amount), 2) AS total FROM employees e "
               "JOIN expenses x USING (employee_id) GROUP BY e.name ORDER BY total DESC")
check("select: ok", r["status"] == "ok", str(r.get("error")))
check("select: columns", r["columns"] == ["name", "total"], str(r["columns"]))
check("select: rows", r["rows"] == [["Ava", 300.5], ["Mia", 75.0], ["Liam", 50.25]], str(r["rows"]))
check("select: row_count, not truncated", r["row_count"] == 3 and r["truncated"] is False, str(r))

r = engine.run("WITH t AS (SELECT employee_id, SUM(amount) s FROM expenses GROUP BY 1) SELECT ROUND(AVG(s), 2) FROM t")
check("with: allowed", r["status"] == "ok" and r["rows"] == [[141.92]], str(r))

r = engine.run("SELECT AVG(salary) AS avg_salary FROM employees")
check("missing values skipped by AVG", r["rows"] == [[81666.66666666667]], str(r["rows"]))

r = engine.run("SELECT date FROM expenses ORDER BY date LIMIT 1")
check("dates come back as ISO text", r["rows"] == [["2026-01-05"]], str(r["rows"]))

r = engine.run("SELECT salary FROM employees WHERE employee_id = 'E003'")
check("missing comes back as None", r["rows"] == [[None]], str(r["rows"]))

r = engine.run("SELECT 1.0/0 AS x")
check("infinity is JSON-safe", r["status"] == "ok" and r["rows"] == [["inf"]], str(r["rows"]))

r = engine.run("  select 1 as one;  ")
check("a trailing semicolon is fine", r["status"] == "ok" and r["rows"] == [[1]], str(r))

for q in ["DESCRIBE expenses", "SHOW TABLES", "SUMMARIZE employees", "FROM employees LIMIT 1"]:
    r = engine.run(q)
    check(f"allowed: {q}", r["status"] == "ok", str(r.get("error")))

# --- Row cap -----------------------------------------------------------------
r = engine.run("SELECT range AS n FROM range(1000)", max_rows=200)
check("row cap: 200 rows returned", len(r["rows"]) == 200, str(len(r["rows"])))
check("row cap: true count and truncated flag", r["row_count"] == 1000 and r["truncated"] is True, str(r["row_count"]))

# --- Errors come back as messages, not crashes ----------------------------------
r = engine.run("SELECT amout FROM expenses")
check("error: status", r["status"] == "error", r["status"])
check("error: DuckDB's own message, with the suggestion",
      "amout" in (r["error"] or "") and "amount" in (r["error"] or ""), str(r["error"]))

# --- Refused: anything that isn't one read-only query ---------------------------
REFUSE = {
    "insert": "INSERT INTO expenses VALUES (5, 'E003', 1, '2026-01-01')",
    "update": "UPDATE expenses SET amount = 0",
    "delete": "DELETE FROM expenses",
    "create": "CREATE TABLE x AS SELECT 1",
    "drop": "DROP TABLE expenses",
    "copy to file": "COPY expenses TO '/tmp/leak.csv'",
    "attach": "ATTACH '/tmp/other.db'",
    "set": "SET enable_external_access = true",
    "pragma": "PRAGMA version",
    "call": "CALL pragma_version()",
    "install": "INSTALL httpfs",
    "two statements": "SELECT 1; DELETE FROM expenses",
    "empty": "   ",
    "only a comment": "-- nothing here",
}
for name, q in REFUSE.items():
    r = engine.run(q)
    check(f"refused: {name}", r["status"] == "refused" and r["error"], f"{r['status']}: {r.get('error')}")
r = engine.run("SELECT COUNT(*) FROM expenses")
check("data unchanged after the refusals", r["rows"] == [[4]], str(r["rows"]))

# SELECTs that try to reach a file: blocked by DuckDB itself (layer 2).
for name, q in {"read_csv outside": "SELECT * FROM read_csv('/etc/passwd')",
                "read the source CSV": f"SELECT * FROM read_csv('{FOLDER / 'expenses.csv'}')",
                "glob": "SELECT * FROM glob('/data/*')"}.items():
    r = engine.run(q)
    check(f"file access blocked: {name}", r["status"] == "error" and "disabled" in (r["error"] or ""),
          str(r.get("error"))[:90])

r = engine.run("SELECT current_setting('enable_external_access') AS v")
check("external access is off", r["rows"] == [[False]], str(r["rows"]))

r = engine.run("SELECT value FROM duckdb_settings() WHERE name = 'autoload_known_extensions'")
check("extensions can't auto-load (e.g. httpfs for the internet)", r["rows"] == [["false"]], str(r))
r = engine.run("SELECT value FROM duckdb_settings() WHERE name = 'lock_configuration'")
check("settings are locked", r["rows"] == [["true"]], str(r))

# --- Timeout -------------------------------------------------------------------
start = time.monotonic()
r = engine.run("SELECT SUM(a.range * b.range) FROM range(200000) a, range(200000) b", timeout=2)
took = time.monotonic() - start
check("timeout: stopped", r["status"] == "timeout" and took < 10, f"{r['status']} after {took:.1f}s")
r = engine.run("SELECT COUNT(*) FROM employees")
check("timeout: engine still works", r["status"] == "ok" and r["rows"] == [[4]], str(r))

# --- Reload when a CSV changes ---------------------------------------------------
time.sleep(0.05)
(FOLDER / "expenses.csv").write_text(
    (FOLDER / "expenses.csv").read_text() + "5,E003,10.00,2026-03-05\n")
os.utime(FOLDER / "expenses.csv", (time.time() + 5, time.time() + 5))   # a clearly newer mtime
r = engine.run("SELECT COUNT(*) FROM expenses")
check("reload: changed CSV picked up", r["rows"] == [[5]], str(r["rows"]))

(FOLDER / "projects.csv").write_text("project,budget\nData Lake,1000\n")
check("reload: a CSV not in the list is not a table", "projects" not in [t["name"] for t in engine.schema()["tables"]])
FILES.append(str(FOLDER / "projects.csv"))
check("reload: a file added to the list becomes a table", "projects" in [t["name"] for t in engine.schema()["tables"]])

(FOLDER / "projects.csv").unlink()
r = engine.run("SELECT * FROM projects")
check("reload: deleted CSV is gone", r["status"] == "error", r["status"])
s = engine.schema()
check("reload: a listed file that's gone is reported", any(p.endswith("projects.csv: not found") for p in s["problems"]),
      str(s["problems"]))
FILES.pop()

# --- Odd file names and broken files ------------------------------------------------
(FOLDER / "my orders-2026.csv").write_text("id\n1\n")
FILES.append(str(FOLDER / "my orders-2026.csv"))
names = [t["name"] for t in engine.schema()["tables"]]
check("odd file name becomes a plain table name", "my_orders_2026" in names, str(names))
r = engine.run("SELECT COUNT(*) FROM my_orders_2026")
check("odd-name table is queryable", r["rows"] == [[1]], str(r))

(FOLDER / "my_orders_2026.csv").write_text("id\n2\n")   # same table name as 'my orders-2026'
FILES.append(str(FOLDER / "my_orders_2026.csv"))
s = engine.schema()
check("a clashing table name is reported, others still load",
      any("my orders-2026.csv" in p and "replaced by" in p for p in s["problems"])
      and "employees" in [t["name"] for t in s["tables"]], str(s["problems"]))
r = engine.run("SELECT id FROM my_orders_2026")
check("a clash: the LATER file in the list wins", r["rows"] == [[2]], str(r))
FILES[-2], FILES[-1] = FILES[-1], FILES[-2]
r = engine.run("SELECT id FROM my_orders_2026")
check("a clash: swap the order, the other file wins", r["rows"] == [[1]], str(r))

(FOLDER / "empty.csv").write_bytes(b"")
FILES.append(str(FOLDER / "empty.csv"))
r = engine.run("SELECT COUNT(*) FROM employees")
check("an empty CSV doesn't break the others", r["status"] == "ok", str(r))
del FILES[2:]

# --- No folders yet --------------------------------------------------------------------
empty = SqlEngine(OUT / "does_not_exist", DATA / "does_not_exist")
check("missing folders: no tables, no crash", empty.schema([])["tables"] == [])
r = empty.run("SELECT 1 AS x", [])
check("missing folders: queries still run", r["status"] == "ok", str(r))

# --- Only the files listed (step 5b) ------------------------------------------------
plain = SqlEngine(OUT, DATA)
(FOLDER / "other_session.csv").write_text("x\n1\n")   # another session's table, same folder
s = plain.schema([EMP])
check("scope: only the listed file is a table", [t["name"] for t in s["tables"]] == ["employees"],
      str([t["name"] for t in s["tables"]]))
check("scope: the table's file is the path given", s["tables"][0]["file"] == EMP, s["tables"][0]["file"])
r = plain.run("SELECT * FROM other_session", [EMP])
check("scope: an unlisted CSV can't be queried", r["status"] == "error", r["status"])
r = plain.run("SELECT * FROM expenses", [EMP])
check("scope: nor can a cleaned table left off the list", r["status"] == "error", r["status"])

s = plain.schema([])
check("scope: no files -> no tables, no problems", s["tables"] == [] and s["problems"] == [], str(s))
r = plain.run("SELECT 1 AS x", [])
check("scope: no files -> SELECT 1 still runs", r["status"] == "ok" and r["rows"] == [[1]], str(r))
r = plain.run("SELECT 1 AS x")
check("scope: files left out means none", r["status"] == "ok", str(r))

# An upload under data/uploads/ is allowed.
(UPLOADS / "sales.csv").write_text("region,total\nNorth,10\nSouth,32\n")
up = str(UPLOADS / "sales.csv")
r = plain.run("SELECT SUM(total) FROM sales", [EMP, up])
check("scope: an upload under data/uploads is a table", r["status"] == "ok" and r["rows"] == [[42]], str(r))
s = plain.schema([EMP, up])
check("scope: both tables listed", [t["name"] for t in s["tables"]] == ["employees", "sales"], str(s["tables"]))

# Anything else is reported, never loaded.
(OUT / "other.csv").write_text("secret\n1\n")      # in output/ but not output/clean/
(FOLDER / "notes.txt").write_text("a\n1\n")         # allowed folder, not a CSV
(DATA / "raw.csv").write_text("raw\n1\n")           # data/ but not data/uploads/
BAD = {"/etc/passwd": "not allowed", str(OUT / "other.csv"): "not allowed",
       str(FOLDER / ".." / "other.csv"): "not allowed", str(FOLDER / "notes.txt"): "not allowed",
       str(DATA / "raw.csv"): "not allowed", str(FOLDER / "missing.csv"): "not found",
       str(UPLOADS / ".." / ".." / ".." / "raw.csv"): "not allowed"}
for f, why in BAD.items():
    s = plain.schema([EMP, f])
    check(f"refused file: {f.replace(str(OUT), '<out>').replace(str(DATA), '<data>')} -> {why}",
          [t["name"] for t in s["tables"]] == ["employees"]
          and any(p.startswith(f"{f}: {why}") for p in s["problems"]), str(s["problems"]))
r = plain.run("SELECT * FROM other", [str(FOLDER / ".." / "other.csv")])
check("refused file: its table doesn't exist", r["status"] == "error", r["status"])
os.symlink("/etc/passwd", FOLDER / "link.csv")
s = plain.schema([str(FOLDER / "link.csv")])
check("refused file: a link pointing outside", s["tables"] == []
      and any("not allowed" in p for p in s["problems"]), str(s))

# The cache: switching between file sets (the two tabs) reuses connections.
cache = SqlEngine(OUT, DATA)
a, b = [EMP], [EMP, EXP]
con_a = cache._connection(cache._resolve(a)[0])[0]
con_b = cache._connection(cache._resolve(b)[0])[0]
check("cache: two file sets, two databases", con_a is not con_b)
check("cache: switching back reuses the first", cache._connection(cache._resolve(a)[0])[0] is con_a)
check("cache: and the second", cache._connection(cache._resolve(b)[0])[0] is con_b)
cache.run("SELECT 1", a)
cache.run("SELECT 1", b)
check("cache: queries alternating between sets don't rebuild",
      cache._connection(cache._resolve(a)[0])[0] is con_a and len(cache.cache) == 2, str(len(cache.cache)))
for extra in ["c1", "c2", "c3"]:
    (FOLDER / f"{extra}.csv").write_text("x\n1\n")
    cache._connection(cache._resolve([EMP, str(FOLDER / f"{extra}.csv")])[0])
check("cache: at most 4 file sets kept", len(cache.cache) == 4, str(len(cache.cache)))
check("cache: the least recently used set (b) is dropped", cache._connection(cache._resolve(b)[0])[0] is not con_b)
try:
    con_b.execute("SELECT 1")
    closed = False
except Exception:
    closed = True
check("cache: a dropped database is closed", closed)

print(f"\n{PASSED} passed, {FAILED} failed")
raise SystemExit(1 if FAILED else 0)
