"""
One question-answering conversation, as the Ask questions tab shows it. No
Gradio in here, so it can be tested with a scripted model and the real sandbox.

Smaller than the cleaning Session: queries can't change anything, so there's
no approval and no change log. It holds the SqlAgent, the chat, the tables
(from the executor's /sql/schema) and the last query's rows for the UI's
results table. Events become chat messages through ChatDisplay, the same
code the Clean tab uses.

Which tables: exactly this session's, never every CSV in output/clean/.
They are the cleaning session's saved tables (Session.sql_files()) plus CSVs
uploaded on this tab, which are treated as already clean. The list is sent
with every /sql call; files from tests or earlier sessions are never loaded.
Ask uploads go to data/uploads/<session id>/ask/, inside the folder New
session and exit already delete.
"""

import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from . import config
from .executor import format_sql_for_model
from .loop import CodeResult, SqlAgent
from .session import MAX_SHOWN, ChatDisplay, Session, _safe_name, csv_summary, remove_folder
from .sql_prompts import build_sql_prompt, keep_tables

# The executor's report of a clash (sql.py _connection): the later file wins.
CLASH = re.compile(r"^(?P<cleaned>.+): replaced by (?P<upload>.+) \(same table name '(?P<table>[^']+)'\)$")

ROWS_IN_CHAT = 20   # rows of each result shown in the chat; the UI's table shows them all

# What the model is told when the tables change mid-conversation: its earlier
# results may be out of date, and the schema in its instructions is new.
TABLES_CHANGED = ("[Tables] The tables have changed. The list under \"Tables\" in your "
                  "instructions is up to date. Query again rather than reusing earlier results.")


@dataclass
class AskSession(ChatDisplay):
    agent: SqlAgent
    output_dir: Path | None = None     # where data_dictionary.md is (output_dir/clean/), as seen from here
    clean: Session | None = None       # the Clean tab's session: its saved tables and its upload folder
    messages: list[dict] = field(default_factory=list)
    files: list[str] = field(default_factory=list)      # names uploaded on this tab
    file_info: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (rows, columns)
    table_origin: dict[str, str] = field(default_factory=dict)  # table -> "cleaned" or "uploaded"
    tables: list[dict] = field(default_factory=list)   # from /sql/schema: name, file, rows, columns
    problems: list[str] = field(default_factory=list)  # CSVs that couldn't be loaded as tables
    schema_error: str | None = None    # why the tables couldn't be read, if they couldn't
    last_result: dict | None = None    # the last successful query: query, columns, rows, row_count, truncated
    _working: dict | None = None       # the "Thinking…" message while the bot works

    running_title: ClassVar[str] = "Running the query…"

    @property
    def upload_dir(self) -> Path | None:
        """data/uploads/<session id>/ask/: inside the cleaning session's own
        folder, so New session and exit delete these uploads too."""
        if self.clean is None or self.clean.upload_dir is None:
            return None
        return self.clean.upload_dir / "ask"

    # --- What the UI calls --------------------------------------------------

    def send(self, text: str) -> Iterator[None]:
        """Ask a question. A generator, like Session.send: it yields after
        every event so the UI can redraw as the work happens."""
        text = text.strip()
        if not text:
            return
        self.messages.append({"role": "user", "content": text})
        yield
        yield from self._show(self.agent.send(text))

    def add_files(self, paths: list) -> list[str]:
        """Copy CSVs uploaded on this tab into data/uploads/<id>/ask/ and make
        them tables. Returns the names added or replaced. Like the Clean
        tab's add_files: same safe names, Gradio's temporary copy deleted."""
        if not paths or self.upload_dir is None:
            return []
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        added, replaced = [], []
        for p in paths:
            p = Path(p)
            name = _safe_name(p.name)
            if name.lower().endswith(".csv"):
                (replaced if (self.upload_dir / name).exists() else added).append(name)
                shutil.copyfile(p, self.upload_dir / name)
                rows, header, _ = csv_summary(self.upload_dir / name)
                self.file_info[name] = (rows, len(header))
            try:
                p.unlink()   # Gradio's copy in /tmp: we have our own now
            except OSError:
                pass
        if not (added or replaced):
            return []
        self.files = sorted(set(self.files) | set(added))
        clashes = self.refresh_tables()
        said = "; ".join(x for x in [added and f"uploaded {', '.join(added)}",
                                     replaced and f"replaced {', '.join(replaced)} with a new version"] if x)
        self._say(f"*{said[0].upper() + said[1:]}.*")
        for name, table in clashes:
            if name in added + replaced:
                self._say(f"*{name} replaces the cleaned table {table} for questions.*")
        return added + replaced

    def remove_file(self, name: str) -> str | None:
        """Delete one Ask upload. Returns None when done, or why not. A
        cleaned table it was hiding (same name) comes back."""
        if name not in self.file_info or self.upload_dir is None:
            return f"{name} isn't one of this tab's uploads."
        (self.upload_dir / name).unlink(missing_ok=True)
        self.files = [f for f in self.files if f != name]
        self.file_info.pop(name, None)
        self.refresh_tables()
        self._say(f"*Removed {name}.*")
        return None

    def refresh_tables(self) -> list[tuple[str, str]]:
        """Work out this session's files, read their tables and the data
        dictionary, and rebuild the system prompt. Called when the tab opens,
        by its button, and after anything that changes the files on either
        tab. If the tables changed mid-conversation, the model is told too.

        Returns the clashes: (upload name, table) for each upload that
        replaces a cleaned table of the same name."""
        before = _signature(self.tables)
        cleaned = self.clean.sql_files() if self.clean is not None else []
        uploaded = [f"/data/uploads/{self.clean.session_id}/ask/{n}" for n in self.files] \
            if self.clean is not None else []
        # Uploads last: when two files make the same table, the later one wins.
        self.agent.files = cleaned + uploaded
        schema = self.agent.executor.schema(self.agent.files)
        self.tables = schema.get("tables", [])
        self.schema_error = schema.get("error")
        # "x: replaced by y (same table name 't')" is a clash the chat
        # explains, not a failure: kept out of the problems list.
        problems = schema.get("problems", [])
        self.problems = [p for p in problems if ": replaced by " not in p]
        clashes = []
        for p in problems:
            m = CLASH.match(p)
            if m:
                clashes.append((m.group("upload").rsplit("/", 1)[-1], m.group("table")))
        self.table_origin = {t["name"]: "uploaded" if t["file"].startswith("/data/uploads/") else "cleaned"
                             for t in self.tables}
        # Notes from the data dictionary describe what cleaning did, so only
        # cleaned tables get them, never an upload with the same name.
        notes = keep_tables(self._read_dictionary(),
                            [t for t, origin in self.table_origin.items() if origin == "cleaned"])
        self.agent.system_prompt = build_sql_prompt(schema | {"problems": self.problems}, notes)
        self.agent.history[0] = {"role": "system", "content": self.agent.system_prompt}
        if len(self.agent.history) > 1 and _signature(self.tables) != before:
            self.agent.history.append({"role": "user", "content": TABLES_CHANGED})
        return clashes

    def reset(self) -> None:
        """New session: empty chat, no uploads, no result, and the tables of
        the (new) cleaning session, i.e. none. Called after Session.new(),
        which has already deleted the old upload folder, ask/ included."""
        if self.upload_dir is not None:
            remove_folder(self.upload_dir)   # in case the cleaning session didn't
        self.new()
        self.files, self.file_info = [], {}
        self.refresh_tables()

    def new(self) -> None:
        """An empty conversation. The tables, the cleaned files and the Clean
        tab are left as they are."""
        self.agent.new_analysis()
        self.messages = []
        self._working = None
        self.last_result = None

    # --- Internals ----------------------------------------------------------

    def _read_dictionary(self) -> str:
        """output/clean/data_dictionary.md, or "" if there is none yet."""
        if self.output_dir is None:
            return ""
        try:
            return (self.output_dir / "clean" / "data_dictionary.md").read_text(encoding="utf-8")
        except OSError:
            return ""

    # --- Events -> chat messages (see ChatDisplay) ---------------------------

    def _add(self, event) -> None:
        if isinstance(event, CodeResult) and event.result.get("status") == "ok":
            # Every returned row (up to the server's 200), for the UI's table.
            # A failed query keeps the last good result on show.
            r = event.result
            self.last_result = {"query": event.code, "columns": r.get("columns", []),
                                "rows": r.get("rows", []), "row_count": r.get("row_count", 0),
                                "truncated": r.get("truncated", False)}
        super()._add(event)

    def _result_title(self, result: dict) -> str:
        return {"ok": "Ran a query", "refused": "The query was refused",
                "timeout": "The query timed out"}.get(result.get("status"), "The query failed")

    def _describe_result(self, code: str, result: dict) -> str:
        """The query, then its first ROWS_IN_CHAT rows as a Markdown table
        (or the error), shown in a collapsible "Ran a query" section."""
        parts = [f"```sql\n{code}\n```"]
        text = format_sql_for_model(result, max_rows=ROWS_IN_CHAT, max_chars=MAX_SHOWN)
        if result.get("status") == "ok":
            # The count line, then a blank line: a Markdown table must start
            # its own block, or it's drawn as plain text.
            count, _, table = text.partition("\n")
            parts.append(f"{count}\n\n{table}" if table else count)
        else:
            parts.append(f"```text\n{text}\n```")
        return "\n\n".join(parts)


def _signature(tables: list[dict]) -> list:
    """What the model relies on: each table's name, row count and columns."""
    return [(t["name"], t["rows"], [(c["name"], c["type"]) for c in t["columns"]]) for t in tables]


def make_ask_session(session: Session) -> AskSession:
    """The Ask questions tab's session, sharing the cleaning session's model
    and sandbox (both were already checked by make_session()). One model for
    both tabs: the UI queues their calls so it's never asked two things at once."""
    agent = SqlAgent(llm=session.agent.llm, executor=session.agent.executor,
                     system_prompt=build_sql_prompt({"tables": []}),
                     max_steps=config.MAX_STEPS, max_result_chars=config.MAX_RESULT_CHARS)
    ask = AskSession(agent, output_dir=session.output_dir, clean=session)
    ask.refresh_tables()
    return ask
