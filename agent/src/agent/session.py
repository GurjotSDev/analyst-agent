"""
One cleanup conversation, as the web UI shows it. No Gradio in here.

The Agent produces events (ModelReply, NeedsApproval, CodeResult, Notice,
TurnDone). A Session turns them into chat messages and remembers what the UI
needs to show: the conversation, and the code waiting for approval, if any.
ui.py only draws a Session; keeping the logic here means it can be tested
with a scripted model and the real sandbox, without a browser.

A Session also owns the uploaded files (data/uploads/<session id>/, deleted
again by new() and close()) and reads the change log and saved files from
the sandbox for the UI's change-log table and downloads.

Chat messages are dicts in the format Gradio's Chatbot expects:
    {"role": "user" | "assistant", "content": "...",
     "metadata": {"title": "..."}}     # optional: a collapsible section
"""

import json
import re
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .codeblocks import CODE_BLOCK
from .executor import ExecutorClient
from .llm import OllamaLLM
from .loop import Agent, CodeResult, ModelReply, NeedsApproval, Notice, TurnDone
from .prompts import build_system_prompt

MAX_SHOWN = 4000   # characters of output shown per result (the model gets its own limit)
REPORT_FILES = ["cleaning_report.pdf", "data_dictionary.md", "change_log.json"]

# Read the change log, the saved tables and where each table came from, for
# the UI. One print of an expression: it creates no variables in the kernel,
# and it never goes into the model's history. default=str: an example value
# can be a date or another type json doesn't know.
PEEK = ("print(__import__('json').dumps({'log': [[e['n'], e['table'], e['action'] + (' ' + repr(e['column']) "
        "if e['column'] else ''), e['summary'], e['reason'], e['examples'][:1], e['removed'][:2]] "
        "for e in __import__('cleaning')._LOG], "
        "'saved': sorted(__import__('cleaning')._SAVED), "
        "'sources': __import__('cleaning')._SOURCES}, default=str))")

PREVIEW_ROWS = 5   # rows of an upload shown in the UI's preview


def remove_folder(path: Path) -> str | None:
    """Delete a folder of uploads. Returns None if it's gone, or why not.
    Never silently: these are copies of sensitive data."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    return None


def csv_summary(path: Path) -> tuple[int, list[str], list[list[str]]]:
    """(data rows, header, first PREVIEW_ROWS rows) of a CSV, for the UI.
    The csv module, not pandas: it's all the UI needs, and it handles quoted
    values with commas or line breaks in them, which counting lines wouldn't.
    It reads the file once, row by row, so a large upload isn't loaded whole."""
    import csv
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        preview, rows = [], 0
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue   # blank lines: pandas skips them too
            if rows < PREVIEW_ROWS:
                preview.append(row)
            rows += 1
    return rows, header, preview


def _example(entry: list) -> str:
    """One before -> after for the change-log table, e.g.
    "ticket_id=T1009: priority 'Med' -> 'Medium'", or the rows removed."""
    examples, removed = entry[5], entry[6]
    if examples:
        x = examples[0]
        show = lambda v: "(missing)" if v is None else repr(v)   # noqa: E731
        return f"{x['row']}: {x['column']} {show(x['before'])} → {show(x['after'])}"
    if removed:
        return "removed " + ", ".join(removed)
    return ""


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _safe_name(name: str) -> str:
    """A file name that's safe on disk and as a table name: 'my orders.csv'
    becomes 'my_orders.csv' (and the table 'my_orders')."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name).lstrip(".") or "file.csv"


@dataclass
class Session:
    agent: Agent
    messages: list[dict] = field(default_factory=list)
    pending_code: str | None = None     # code waiting for Run it / Don't run
    data_dir: Path | None = None        # uploads go to data_dir/uploads/<id>/ (None: no uploads)
    output_dir: Path | None = None      # where save_clean() writes, as seen from here
    session_id: str = field(default_factory=_new_id)
    files: list[str] = field(default_factory=list)    # uploaded files, relative to /data
    log_rows: list[list] = field(default_factory=list)  # the change log, for the table
    downloads: list[Path] = field(default_factory=list)  # saved files and reports
    file_info: dict[str, int] = field(default_factory=dict)  # upload name -> data rows
    columns: dict[str, int] = field(default_factory=dict)    # upload name -> columns
    preview: tuple[str, list[str], list[list[str]]] | None = None  # (name, header, rows) of the last upload
    _working: dict | None = None         # the "Thinking…" message while the bot works

    @property
    def upload_dir(self) -> Path | None:
        return self.data_dir / "uploads" / self.session_id if self.data_dir else None

    # --- What the UI calls --------------------------------------------------
    # Each of these is a generator: it yields after every event, so the UI can
    # redraw as the work happens instead of after the whole turn.

    def send(self, text: str) -> Iterator[None]:
        text = text.strip()
        if not text:
            return
        if self.pending_code is not None:
            self._say("Run or decline the waiting code first (the buttons under the chat).")
            yield
            return
        self.messages.append({"role": "user", "content": text})
        yield
        yield from self._show(self.agent.send(text))

    def approve(self) -> Iterator[None]:
        if self.pending_code is None:
            return
        self.pending_code = None
        yield from self._show(self.agent.resume(approved=True), working="Running the code…")

    def decline(self, feedback: str = "") -> Iterator[None]:
        if self.pending_code is None:
            return
        # The waiting code is shown only beside the buttons, so keep a copy
        # in the chat: the history should show what was turned down.
        self._say(f"```python\n{self.pending_code}\n```", title="Code you didn't run")
        self.pending_code = None
        feedback = feedback.strip()
        self.messages.append({"role": "user", "content":
                              f"Don't run that. {feedback}" if feedback else "Don't run that."})
        yield
        yield from self._show(self.agent.resume(approved=False, feedback=feedback))

    def add_files(self, paths: list) -> list[str]:
        """Copy uploaded CSVs into this session's folder and tell the model.
        Returns the names added or replaced. Gradio's own temporary copy is deleted."""
        if not paths or self.upload_dir is None:
            return []
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        added, replaced = [], []
        for p in paths:
            p = Path(p)
            name = _safe_name(p.name)
            if name.lower().endswith(".csv"):
                # Same name as an earlier upload: the new file replaces it, so
                # the model never sees two copies. Say so, not "Uploaded".
                (replaced if (self.upload_dir / name).exists() else added).append(name)
                shutil.copyfile(p, self.upload_dir / name)
                rows, header, first = csv_summary(self.upload_dir / name)
                self.file_info[name], self.columns[name] = rows, len(header)
                self.preview = (name, header, first)
            try:
                p.unlink()   # Gradio's copy in /tmp: we have our own now
            except OSError:
                pass
        if added or replaced:
            prefix = f"uploads/{self.session_id}/"
            self.files = sorted(set(self.files) | {prefix + n for n in added})
            self._update_prompt()
            said = "; ".join(x for x in [added and f"uploaded {', '.join(added)}",
                                         replaced and f"replaced {', '.join(replaced)} with a new version"] if x)
            if len(self.agent.history) > 1:   # mid-conversation: tell the model too
                self.agent.history.append({"role": "user", "content":
                    f"[Files] I {said}. Every file is listed under 'Files available'. "
                    f"Load a replaced file again if you already loaded it."})
            self._say(f"*{said[0].upper() + said[1:]}.*")
        return added + replaced

    def refresh(self) -> None:
        """Read the change log and saved tables from the sandbox."""
        try:
            out = self.agent.executor.execute(PEEK).get("stdout", "")
            state = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return   # no answer (the sandbox failed or restarted): keep the last table
        # The kernel is shared: the tests, the CLI or the eval may have loaded
        # tables into it too. With uploads, show only tables loaded from this
        # session's own folder (sandbox paths look like /data/uploads/<id>/x.csv).
        mine = (lambda t: f"uploads/{self.session_id}/" in state["sources"].get(t, "")) \
            if self.data_dir is not None else (lambda t: True)
        self.log_rows = [row[:5] + [_example(row)] for row in state["log"] if mine(row[1])]
        if self.output_dir is None:
            return
        clean = self.output_dir / "clean"
        saved = [t for t in state["saved"] if mine(t)]
        names = [f"{t}.csv" for t in saved] + (REPORT_FILES if saved else [])
        self.downloads = [clean / n for n in names if (clean / n).exists()]

    def close(self) -> str | None:
        """Delete this session's uploads. Called on New session and on exit.
        Returns None, or why they couldn't be deleted."""
        if self.upload_dir is None:
            return None
        problem = remove_folder(self.upload_dir)
        if problem:
            print(f"WARNING: could not delete the uploads in {self.upload_dir}: {problem}")
        return problem

    def set_auto(self, auto: bool) -> None:
        """Auto mode: code runs without asking. Off by default."""
        self.agent.require_approval = not auto

    def new(self) -> None:
        """Fresh kernel, empty conversation, uploads deleted."""
        problem = self.close()
        old_dir = self.upload_dir
        self.session_id = _new_id()
        self.files = []
        self.file_info, self.columns, self.preview = {}, {}, None
        self.log_rows, self.downloads = [], []
        self._update_prompt()
        self.agent.new_analysis()
        self.messages = []
        self._working = None
        self.pending_code = None
        if problem:
            self._say(f"**Warning:** the previous session's uploads could not be deleted ({problem}). "
                      f"Delete the folder {old_dir} yourself.")

    def _update_prompt(self) -> None:
        """The system prompt lists the files; keep it in step with uploads."""
        if self.data_dir is None:
            return
        self.agent.system_prompt = build_system_prompt(self.files)
        if self.agent.history:
            self.agent.history[0] = {"role": "system", "content": self.agent.system_prompt}

    # --- Events -> chat messages --------------------------------------------

    def _show(self, events, working: str = "Thinking…") -> Iterator[None]:
        """Turn the agent's events into chat messages, redrawing after each.
        While the agent works (the model can take minutes, and nothing else
        moves on screen), a "Thinking…" or "Running the code…" message with a
        spinner sits at the bottom of the chat, so it never looks stuck."""
        self._set_working(working)
        yield
        try:
            for event in events:
                self._set_working(None)
                self._add(event)
                self._set_working(_next_step(event, self.agent.require_approval))
                yield
        except Exception as e:
            # A failure (Ollama stopped, sandbox down) is shown in the chat,
            # and the session stays usable.
            self._set_working(None)
            self.pending_code = None
            self.agent.pending_code = None
            self._say(f"**Error:** {type(e).__name__}: {e}")
            yield
        if self._working is not None:
            self._set_working(None)
            yield

    def _set_working(self, title: str | None) -> None:
        """Show, change or remove the placeholder message. Gradio draws
        metadata status "pending" as a spinner next to the title."""
        if self._working is not None and self._working in self.messages:
            self.messages.remove(self._working)
        self._working = None
        if title:
            self._working = {"role": "assistant", "content": "",
                             "metadata": {"title": title, "status": "pending"}}
            self.messages.append(self._working)

    def _add(self, event) -> None:
        match event:
            case ModelReply(text=text):
                # The code is shown separately (beside the buttons while it
                # waits, then with its result), so the reply's own copy is left out.
                shown = _plain_arrows(CODE_BLOCK.sub("", text).strip())
                if shown:
                    self._say(shown)
            case NeedsApproval(code=code):
                # Shown in the code panel beside Run it / Don't run, not in the
                # chat: one copy, right where the decision is made.
                self.pending_code = code
            case CodeResult(code=code, result=result):
                self._say(_describe_result(code, result), title=_result_title(result))
            case Notice(text=text):
                self._say(f"*Note: {text}*")
            case TurnDone(reason="declined"):
                self._say("*The code was not run.*")
            case TurnDone():
                pass

    def _say(self, content: str, title: str | None = None) -> None:
        message = {"role": "assistant", "content": content}
        if title:
            message["metadata"] = {"title": title}   # Gradio draws this as a collapsible section
        self.messages.append(message)


def _result_title(result: dict) -> str:
    if result.get("kernel_restarted"):
        return "Ran the code: the kernel was restarted"
    return {"ok": "Ran the code", "timeout": "The code timed out"}.get(
        result["status"], "The code raised an error")


# LaTeX arrows the model sometimes writes ("'a' $\rightarrow$ 'A'"). The chat
# doesn't render single-$ LaTeX, so they'd show as raw code. The prompt asks
# for → instead; this catches the times the model forgets.
LATEX_ARROW = re.compile(r"\$\s*\\(?:rightarrow|to|longrightarrow)\s*\$")


def _plain_arrows(text: str) -> str:
    return LATEX_ARROW.sub("→", text)


def _next_step(event, require_approval: bool) -> str | None:
    """What the agent does after this event, for the placeholder's title.
    None: nothing; the turn is over or waits for the user."""
    match event:
        case ModelReply(text=text):
            # A reply with code runs next (auto mode) or waits for approval.
            if CODE_BLOCK.search(text):
                return None if require_approval else "Running the code…"
            return None
        case NeedsApproval() | TurnDone():
            return None
        case _:   # a result or a note: the model reads it and replies
            return "Thinking…"


def _describe_result(code: str, result: dict) -> str:
    """The result as Markdown: the code that ran, then what it printed, then
    the error, if any. The code is always included: while it waited it was
    only in the panel beside the buttons, and the chat should show what ran."""
    parts = [f"```python\n{code}\n```"]
    out = result.get("stdout", "")
    if out:
        cut = f"\n... ({len(out) - MAX_SHOWN:,} more characters)" if len(out) > MAX_SHOWN else ""
        parts.append(f"```text\n{out[:MAX_SHOWN].rstrip()}{cut}\n```")
    if result.get("error"):
        parts.append(f"```text\n{_short_error(result['error'])}\n```")
    return "\n\n".join(parts)


def _short_error(error: str) -> str:
    """The error as a person needs it: its last line, e.g. "CleaningError:
    opened_date: not real dates ...". The traceback above it (lines of
    cleaning.py) is for the model, which still gets all of it. A short last
    line like "KeyError: 'x'" says too little alone, so then the last few
    lines come too, to show where it happened."""
    lines = [line for line in error.strip().splitlines() if line.strip()]
    if not lines:
        return error
    if len(lines[-1]) >= 40:
        return lines[-1]
    return "\n".join(lines[-6:])


def make_session() -> Session:
    """The real thing: the sandbox, the local model, and uploads.
    Raises ExecutorNotSafe or ModelNotAvailable with a message saying what to fix."""
    # Uploads from a session that ended without cleaning up (a crash, a
    # power cut) are deleted first, before anything that might fail: the
    # data is sensitive, and the app may never get further than this.
    problem = remove_folder(config.DATA_DIR / "uploads")
    if problem:
        print(f"WARNING: could not delete old uploads in {config.DATA_DIR / 'uploads'}: {problem}")
    executor = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
    executor.check_sandboxed()
    llm = OllamaLLM(config.MODEL, config.OLLAMA_HOST, config.THINK, config.NUM_CTX)
    llm.check_available()
    agent = Agent(llm=llm, executor=executor, system_prompt=build_system_prompt([]),
                  max_steps=config.MAX_STEPS, max_result_chars=config.MAX_RESULT_CHARS)
    session = Session(agent, data_dir=config.DATA_DIR, output_dir=config.OUTPUT_DIR)
    session.new()   # never inherit an earlier session's kernel
    return session