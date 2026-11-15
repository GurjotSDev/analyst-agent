"""
Client for the sandboxed executor (the Docker container from stage 3).

The agent never runs model-written code itself. It hands the code to this
client, which sends it to the executor over HTTP, and refuses to talk to any
executor that doesn't report it's running in the sandbox.
"""

import re

import httpx  # HTTP client. The ollama library uses it too, so it's already installed.


class ExecutorNotSafe(Exception):
    """The executor isn't the sandboxed one, or isn't reachable."""


class ExecutorClient:
    def __init__(self, base_url: str, timeout: float, transport: httpx.BaseTransport | None = None):
        # A Client reuses one connection for many requests (faster than
        # httpx.post() each time). base_url means calls can say "/execute"
        # instead of the full address. `transport` lets tests swap in a fake
        # server; normally it's None and real HTTP is used.
        self.http = httpx.Client(base_url=base_url, transport=transport, timeout=30)
        self.exec_timeout = timeout

    def check_sandboxed(self) -> None:
        """Raise ExecutorNotSafe unless the executor is up AND sandboxed.

        Called before any model code runs. It protects against a mistake like
        running the stage 2 server directly on Windows, where model code would
        have full access to your files.
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
        an error result, so the agent keeps going and the model is told.
        Before this, one server bug (a folder deleted while it ran) crashed the
        whole session and lost everything in the kernel.
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
            # A bug in the executor itself (HTTP 500 = "internal server error").
            # The code may well have run; the failure was in handling its output.
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

    def kernel(self) -> int | None:
        """The number of the kernel running now, or None if the sandbox can't
        be reached. It goes up by one with every reset or restart. Every
        program shares this one kernel, so a number that changed since your
        own reset means someone else reset it."""
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

    Why: check() on two wide tables printed 6,672 characters, and shorten()
    cut the middle, which held most of the first table's "Issues to report".
    The model then never saw those issues, so it couldn't report them. The
    numbered list is the one part it must see whole; the detail above it is
    context. Detail lines are kept in order until the budget runs out, and
    each run of cut lines is marked. If the key lines alone don't fit (or
    there are none), it falls back to shorten()."""
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