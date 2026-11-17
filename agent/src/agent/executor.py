"""
Client for the sandboxed executor (the Docker container).

The agent never runs model-written code itself. It hands the code to this
client, which sends it to the executor over HTTP, and refuses to talk to any
executor that doesn't report it's running in the sandbox.
"""

import re

import httpx


class ExecutorNotSafe(Exception):
    """The executor isn't the sandboxed one, or isn't reachable."""


class ExecutorClient:
    def __init__(self, base_url: str, timeout: float, transport: httpx.BaseTransport | None = None):
        # `transport` lets tests swap in a fake server; normally it's None.
        self.http = httpx.Client(base_url=base_url, transport=transport, timeout=30)
        self.exec_timeout = timeout

    def check_sandboxed(self) -> None:
        """Raise ExecutorNotSafe unless the executor is up AND sandboxed.

        Called before any model code runs, so code is never sent to a server
        started by hand outside the container, where it would have full access
        to the host.
        """
        try:
            health = self.http.get("/health").json()
        except httpx.HTTPError as e:
            raise ExecutorNotSafe(
                f"Can't reach the executor ({e}). Start it from the repo root with: "
                "docker compose up -d") from e
        if health.get("sandboxed") is not True:
            raise ExecutorNotSafe(
                "The executor on this address is NOT the sandboxed container. "
                "Stop any server you started by hand, then run: docker compose up -d")
        if health.get("status") != "ok":
            self.reset()  # kernel died between sessions; start fresh

    def execute(self, code: str) -> dict:
        """Run code; return the executor's JSON response as a dict.

        Never raises for problems on the executor's side: they come back as
        an error result, so the session survives and the model is told.
        """
        try:
            r = self.http.post("/execute", json={"code": code, "timeout": self.exec_timeout},
                               timeout=self.exec_timeout + 60)  # leave the server time to clean up
        except httpx.HTTPError as e:
            # No answer at all: container stopped, Docker Desktop closed, timeout.
            return _executor_failure(
                f"The sandbox could not be reached ({type(e).__name__}). Your code may not "
                "have run. Tell the user; they may need to restart it with docker compose up -d.")

        if r.status_code == 503:
            # The kernel was dead before this code even ran. Restart it and
            # report that the same way the server reports a crash.
            self.reset()
            return {"status": "error", "stdout": "", "stderr": "",
                    "truncated": False, "kernel_restarted": True,
                    "error": "The kernel had stopped, so it was restarted before running "
                             "this code. All variables are gone; load the data again."}
        if r.status_code >= 500:
            # A bug in the executor itself. The code may well have run; the
            # failure was in handling its output.
            return _executor_failure(
                f"The sandbox had an internal error (HTTP {r.status_code}) while handling this "
                "code's output. This is not a problem with your code. The code probably ran, "
                "so variables it created may exist; check before relying on them.")
        r.raise_for_status()  # anything else unexpected (4xx) is a bug in this client
        return r.json()

    def reset(self) -> int:
        """Fresh kernel. Returns its number (see kernel())."""
        r = self.http.post("/reset", timeout=120)
        r.raise_for_status()
        return r.json()["kernel"]

    def sql(self, query: str, files=()) -> dict:
        """Run one read-only SQL query on the listed files (the executor's
        /sql, see executor/sql.py). files are sandbox paths such as
        /output/clean/expenses.csv; only they are tables. Like execute(),
        never raises: a sandbox problem comes back as an error result the
        model and the user can read."""
        try:
            # The server stops a query after 30 s; leave it time to say so.
            r = self.http.post("/sql", json={"query": query, "files": list(files)}, timeout=90)
        except httpx.HTTPError as e:
            return _executor_failure(
                f"The sandbox could not be reached ({type(e).__name__}). The query did not run. "
                "Tell the user; they may need to restart it with docker compose up -d.")
        if r.status_code >= 400:
            return _executor_failure(f"The sandbox had an internal error (HTTP {r.status_code}) "
                                     "while running this query. This is not a problem with your SQL.")
        return r.json()

    def schema(self, files=()) -> dict:
        """The listed files' tables: {"tables": [{name, file, rows, columns}],
        "problems": [...]} (problems: files not allowed, missing or unreadable).
        On failure: no tables, and "error" saying why (never raises)."""
        try:
            # POST: the file list rides in the body.
            r = self.http.post("/sql/schema", json={"files": list(files)}, timeout=90)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            return {"tables": [], "problems": [],
                    "error": f"The sandbox could not be reached or failed ({type(e).__name__}); "
                             "start it with docker compose up -d."}

    def kernel(self) -> int | None:
        """The number of the kernel running now, or None if the sandbox can't
        be reached. It goes up with every reset or restart; all clients share
        one kernel, so a change since your own reset means another client
        reset it."""
        try:
            return self.http.get("/health").json().get("kernel")
        except (httpx.HTTPError, ValueError):
            return None


def _executor_failure(message: str) -> dict:
    """A result in the executor's usual shape, describing a failure of the
    executor itself rather than of the code. executor_failure=True lets the
    loop tell the user too."""
    return {"status": "error", "stdout": "", "stderr": "",
            "truncated": False, "kernel_restarted": False,
            "executor_failure": True, "error": message}


# ---------------------------------------------------------------------------
# Turning a result into the text the MODEL reads
# ---------------------------------------------------------------------------

def shorten(text: str, limit: int) -> str:
    """Cut text to about `limit` characters, keeping its start and end.

    Both ends matter: the start of a printout shows its structure, and the
    end of a traceback holds the actual error message.
    """
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} characters cut]...\n{text[-tail:]}"


# Lines of a helper's printout the model must never lose: the checklist it
# copies into its reply ("Issues to report" and its numbered lines), each
# table's header, and what changed (load:, the [n] change-log lines,
# save_clean:). Everything else is detail: value counts, ranges, examples.
_KEY_LINE = re.compile(r"^(Issues to report|===|load:|save_clean:|\[\d+\] |\s+\d+\. )")


def shorten_report(text: str, limit: int) -> str:
    """Like shorten(), but for a helper's report: when it's too long, cut
    DETAIL lines and keep every key line.

    A plain middle cut can remove part of an "Issues to report" list, and
    the model can't report issues it never saw; the detail above the list is
    only context. Detail lines are kept in order until the budget runs out,
    and each run of cut lines is marked. If the key lines alone don't fit
    (or there are none), it falls back to shorten()."""
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    key = [bool(_KEY_LINE.match(line)) for line in lines]
    key_size = sum(len(line) + 1 for line, k in zip(lines, key) if k)
    if not any(key) or key_size > limit:
        return shorten(text, limit)
    budget = limit - key_size - 200   # room for the "cut" markers
    out, cut = [], 0
    for line, k in zip(lines, key):
        if k or len(line) + 1 <= budget:
            if cut:
                out.append(f"...[{cut} detail line{'s' if cut > 1 else ''} cut; every 'Issues to report' "
                           f"line is kept]...")
                cut = 0
            out.append(line)
            if not k:
                budget -= len(line) + 1
        else:
            cut += 1
    if cut:
        out.append(f"...[{cut} detail line{'s' if cut > 1 else ''} cut]...")
    return "\n".join(out)


def format_for_model(result: dict, max_chars: int) -> str:
    """Describe an execution result as plain text for the model."""
    parts = []
    if result.get("kernel_restarted"):
        parts.append("WARNING: the Python kernel was restarted. Every variable, import and "
                     "loaded DataFrame is gone; reload anything you need.")
    if result.get("stdout"):
        parts.append("Output:\n" + shorten_report(result["stdout"], max_chars))
    if result.get("stderr"):
        # Mostly warnings (e.g. pandas FutureWarning). Worth seeing, briefly.
        parts.append("Warnings:\n" + shorten(result["stderr"], 800))
    if result.get("error"):
        parts.append("Error:\n" + shorten(result["error"], max_chars))
    if result.get("status") == "timeout" and not result.get("error"):
        parts.append("The code took too long and was stopped.")
    return "\n\n".join(parts) or "(The code ran and printed nothing.)"

# ---------------------------------------------------------------------------
# Turning a SQL result into the text the MODEL reads
# ---------------------------------------------------------------------------

SQL_ROWS_FOR_MODEL = 50     # the UI shows up to 200; the model needs far fewer
SQL_CHARS_FOR_MODEL = 3000


def _cell(value) -> str:
    """One value in a Markdown table. Numbers keep their full precision (str
    of a float is its shortest exact form), because the number check compares
    the model's answer with these digits. A | or line break inside a value
    would break the table, so they're escaped or flattened."""
    if value is None:
        return "NULL"
    return " ".join(str(value).split("\n")).replace("|", "\\|")


def format_sql_for_model(result: dict, max_rows: int = SQL_ROWS_FOR_MODEL,
                         max_chars: int = SQL_CHARS_FOR_MODEL) -> str:
    """A /sql result as plain text for the model: a count line, then a
    Markdown table of the first rows. Rows are added until max_rows or
    max_chars is reached, and the count line says how many were shown, so
    the model never mistakes the first 50 rows for all of them."""
    status = result.get("status")
    if status != "ok":
        message = (result.get("error") or "").strip() or "(no message)"
        if status == "refused" or message.startswith("Refused"):
            return message if message.startswith("Refused") else f"Refused: {message}"
        if status == "timeout":
            return f"Timeout: {message}"
        return f"Error:\n{message}"

    columns, rows = result.get("columns") or [], result.get("rows") or []
    count = result.get("row_count", len(rows))
    if count == 0 or not columns:
        return "0 rows: the query matched nothing."

    lines = ["| " + " | ".join(_cell(c) for c in columns) + " |",
             "|" + "---|" * len(columns)]
    size = sum(len(line) + 1 for line in lines) + 60   # room for the count line
    shown = 0
    for row in rows[:max_rows]:
        line = "| " + " | ".join(_cell(v) for v in row) + " |"
        if size + len(line) + 1 > max_chars and shown:
            break
        lines.append(line)
        size += len(line) + 1
        shown += 1

    total = f"more than {count:,} rows" if not result.get("count_exact", True) else \
            f"{count:,} row{'' if count == 1 else 's'}"
    which = "all shown" if shown == count else f"first {shown} shown"
    return f"{total} ({which})\n" + "\n".join(lines)
