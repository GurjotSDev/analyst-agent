"""
Checks for session.py: the UI's logic, without a browser or a real model.

A scripted model plays back fixed replies; the sandbox is real. Needs the
sandbox running (docker compose up -d). Run from agent/:
    uv run python test_session.py
"""

from agent import config
from agent.executor import ExecutorClient
from agent.loop import Agent
from agent.session import Session


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages):
        return self.replies.pop(0)


class BrokenLLM:
    def chat(self, messages):
        raise ConnectionError("Ollama is not running")


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


executor = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
executor.check_sandboxed()


def new_session(replies, approval=True) -> Session:
    agent = Agent(llm=ScriptedLLM(replies), executor=executor, system_prompt="(test)",
                  require_approval=approval)
    s = Session(agent)
    s.new()
    return s


def titles(s: Session) -> list:
    return [m.get("metadata", {}).get("title") for m in s.messages]


code = "```python\nprint(6 * 7)\n```"

# 1. With approval on, code waits, and nothing runs until Run it.
def working(s: Session) -> list:
    """Titles of the spinner placeholders in the chat right now."""
    return [m["metadata"]["title"] for m in s.messages if m.get("metadata", {}).get("status") == "pending"]


s = new_session(["Let me check.\n\n" + code, "The answer is 42."])
steps = s.send("What is 6 times 7?")
next(steps)                                           # the user's message is drawn
next(steps)                                           # then, before the model answers...
check("spinner while the model thinks", working(s) == ["Thinking…"], str(working(s)))
updates = list(steps)
check("a redraw for every step", len(updates) >= 1, f"{len(updates)} more redraws")
check("spinner gone when the turn pauses", working(s) == [])
check("code waits for approval", s.pending_code == "print(6 * 7)")
check("user message shown", s.messages[0] == {"role": "user", "content": "What is 6 times 7?"})
check("reply shown without its code", s.messages[1]["content"] == "Let me check.")
check("waiting code not repeated in the chat", not any("print(6 * 7)" in m["content"] for m in s.messages))
steps = s.approve()
next(steps)
check("spinner while the code runs", working(s) == ["Running the code…"], str(working(s)))
list(steps)
check("runs after approval", s.pending_code is None and "Ran the code" in titles(s))
check("no spinner left at the end", working(s) == [])
result = next(m for m in s.messages if m.get("metadata", {}).get("title") == "Ran the code")
check("result shows the output", "42" in result["content"])
check("result shows the code that ran", "print(6 * 7)" in result["content"])
check("final answer shown", s.messages[-1]["content"] == "The answer is 42.")

# 2. Declining with feedback: shown as your message, the model tries again.
s = new_session([code, "```python\nprint('second try')\n```"])
list(s.send("go"))
list(s.decline("print something else"))
check("feedback shown as your message", {"role": "user", "content": "Don't run that. print something else"}
      in s.messages)
check("model tried again", s.pending_code == "print('second try')")

# 3. A plain decline ends the turn and says so.
s = new_session([code])
list(s.send("go"))
list(s.decline())
check("plain decline: nothing waiting", s.pending_code is None)
check("plain decline: said so", s.messages[-1]["content"] == "*The code was not run.*")
check("declined code kept in the chat", "Code you didn't run" in titles(s)
      and any("print(6 * 7)" in m["content"] for m in s.messages))

# 4. Sending while code waits is refused, nothing lost.
s = new_session([code])
list(s.send("go"))
list(s.send("another question"))
check("can't send while code waits", s.pending_code == "print(6 * 7)"
      and "Run or decline" in s.messages[-1]["content"])

# 5. Auto mode: runs straight away, and the result includes the code.
s = new_session([code, "Done."], approval=False)
list(s.send("go"))
ran = next(m for m in s.messages if m.get("metadata", {}).get("title") == "Ran the code")
check("auto mode: no waiting", s.pending_code is None)
check("auto mode: code shown with the result", "print(6 * 7)" in ran["content"])
s.set_auto(False)
check("auto mode can be switched off", s.agent.require_approval is True)

# 6. Errors in the code are shown with their last line.
s = new_session(["```python\n1 / 0\n```", "That divided by zero."], approval=False)
list(s.send("go"))
check("error result titled", "The code raised an error" in titles(s))
check("error's last line shown", any("ZeroDivisionError" in m["content"] for m in s.messages))
s = new_session(["Merged 'cloud ops' $\\rightarrow$ 'Cloud Ops'."])
list(s.send("go"))
check("LaTeX arrows shown as a plain arrow", s.messages[-1]["content"] == "Merged 'cloud ops' → 'Cloud Ops'.",
      ascii(s.messages[-1]["content"]))   # ascii(): a Windows console can't print the arrow
from agent.session import _short_error
long_error = ("Traceback (most recent call last):\n  File \"cleaning.py\", line 499\n"
              "    f\"(e.g. {_examples(bad)}). Fix them first\"\n"
              "CleaningError: opened_date: not real dates in the given order: 1 value (e.g. '2026-02-30').")
check("long error: only its last line", _short_error(long_error).startswith("CleaningError:")
      and "line 499" not in _short_error(long_error))
check("short error: a few lines for context", "line 499" in _short_error(long_error.rsplit("\n", 1)[0]
      + "\nKeyError: 'x'"))

# 7. A failure (e.g. Ollama stopped) is shown, and the session stays usable.
s = Session(Agent(llm=BrokenLLM(), executor=executor, system_prompt="(test)"))
list(s.send("go"))
check("failure shown in the chat", "Ollama is not running" in s.messages[-1]["content"])
s.agent.llm = ScriptedLLM(["Back again."])
list(s.send("hello"))
check("session usable after a failure", s.messages[-1]["content"] == "Back again.")

# 8. New session: empty chat and a fresh kernel.
s = new_session(["```python\nx = 5\n```", "Set."], approval=False)
list(s.send("go"))
s.new()
check("new session: chat empty", s.messages == [] and s.pending_code is None)
check("new session: kernel fresh", "NameError" in (executor.execute("print(x)")["error"] or ""))

# =============================================================================
# F3: uploads, the change log, downloads, and deleting uploads
# =============================================================================
import tempfile
from pathlib import Path

# Saves go to a test folder, never your real output/clean/.
TEST_OUT = config.OUTPUT_DIR / "evals" / "ui-test"
upload = Path(tempfile.mkdtemp()) / "my orders.csv"
upload.write_text("order_id,total\n1,10\n2,-5\n3,20\n")

s = Session(Agent(llm=ScriptedLLM([
        "```python\norders = load('/data/' + 'PATH')\norders = drop_rows(orders, orders['total'] < 0, "
        "reason='refunds')\nsave_clean(orders)\n```", "Removed 1 refund."]),
        executor=executor, system_prompt="(test)", require_approval=False),
    data_dir=config.DATA_DIR, output_dir=TEST_OUT)
s.new()
executor.execute(f"import cleaning\ncleaning.OUTPUT_DIR = cleaning.OUTPUT_DIR / 'evals' / 'ui-test'")
added = s.add_files([upload])
stored = s.upload_dir / "my_orders.csv"
check("upload copied into the session folder", added == ["my_orders.csv"] and stored.exists())
check("unsafe characters replaced", "my_orders.csv" in s.files[0])
check("Gradio's temporary copy deleted", not upload.exists())
check("prompt lists the upload", f"/data/uploads/{s.session_id}/my_orders.csv" in s.agent.history[0]["content"])
check("upload shown in the chat", "Uploaded my_orders.csv" in s.messages[-1]["content"])
check("upload summary: rows and columns", s.file_info["my_orders.csv"] == 3 and s.columns["my_orders.csv"] == 2,
      f"{s.file_info} {s.columns}")
check("upload preview: header and first rows", s.preview == ("my_orders.csv", ["order_id", "total"],
                                                             [["1", "10"], ["2", "-5"], ["3", "20"]]), str(s.preview))

# Another program's table in the shared kernel must not show in this log.
# (Loaded and changed for real, from outside this session's upload folder.)
executor.execute("import cleaning, pandas as pd\n"
                 "cleaning.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                 "pd.DataFrame({'a': [1, 2]}).to_csv(cleaning.OUTPUT_DIR / 'other.csv', index=False)\n"
                 "_t = cleaning.load(cleaning.OUTPUT_DIR / 'other.csv')\n"
                 "_t = cleaning.drop_rows(_t, _t['a'] == 2, reason='not ours')\ndel _t")
s.agent.llm.replies[0] = s.agent.llm.replies[0].replace("PATH", s.files[0])
list(s.send("clean it"))
s.refresh()
check("change log read from the sandbox", [row[2] for row in s.log_rows] == ["drop_rows"]
      and s.log_rows[0][4] == "refunds", str(s.log_rows))
check("change log: other programs' tables left out", all(row[1] != "other" for row in s.log_rows))
check("change log: example column", s.log_rows[0][5] == "removed order_id=2", str(s.log_rows[0]))
names = sorted(p.name for p in s.downloads)
check("downloads: cleaned file and reports", names == ["change_log.json", "cleaning_report.pdf",
                                                       "data_dictionary.md", "my_orders.csv"], str(names))

# A second upload mid-conversation: the model is told, the prompt updated.
second = Path(tempfile.mkdtemp()) / "customers.csv"
second.write_text("customer_id\n1\n")
s.add_files([second])
check("mid-conversation upload: model told", s.agent.history[-1]["content"].startswith("[Files] I uploaded customers.csv."))
check("prompt lists both files", "customers.csv" in s.agent.history[0]["content"]
      and "my_orders.csv" in s.agent.history[0]["content"])

# The same file uploaded twice: replaced, never listed twice.
dup = Path(tempfile.mkdtemp()) / "customers.csv"
dup.write_text("customer_id\n1\n2\n")
s.add_files([dup])
check("same name again: replaced, listed once", sum("customers.csv" in f for f in s.files) == 1
      and (s.upload_dir / "customers.csv").read_text().count("\n") == 3)
check("same name again: chat says replaced", s.messages[-1]["content"] == "*Replaced customers.csv with a new version.*",
      s.messages[-1]["content"])
check("same name again: model told", "replaced customers.csv" in s.agent.history[-1]["content"])

not_csv = Path(tempfile.mkdtemp()) / "notes.txt"
not_csv.write_text("hi")
check("non-CSV files ignored", s.add_files([not_csv]) == [])

old_dir = s.upload_dir
s.new()
check("new session deletes the uploads", not old_dir.exists())
check("new session: fresh id, no files, empty log", s.upload_dir != old_dir and s.files == []
      and s.log_rows == [] and s.downloads == [] and s.preview is None and s.file_info == {})
check("prompt says to upload files", "ask the user to upload" in s.agent.history[0]["content"])

again = Path(tempfile.mkdtemp()) / "x.csv"
again.write_text("a\n1\n")
s.add_files([again])
s.close()
check("close deletes the uploads", not s.upload_dir.exists())

import shutil
shutil.rmtree(TEST_OUT, ignore_errors=True)