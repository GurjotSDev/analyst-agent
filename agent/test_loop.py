"""
Checks for the agent loop, without a real model.

A ScriptedLLM plays back fixed replies, so each test is fast and gives the same
result every time. Everything else is real: the loop, the code extraction, and
the sandboxed executor running code on the sample data in data/.

Needs the executor running (from the repo root: docker compose up -d).
Run from agent/:
    uv run python test_loop.py
"""

import httpx

from agent import config
from agent.executor import ExecutorClient, ExecutorNotSafe, shorten
from agent.loop import Agent, CodeResult, ModelReply, NeedsApproval, Notice, TurnDone


class ScriptedLLM:
    """Stands in for the model: returns the next prepared reply on each call,
    and records what it was sent so tests can inspect the history."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.seen.append(list(messages))
        return self.replies.pop(0)


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


def py(code: str) -> str:
    """Wrap code in a fence, the way the model writes it."""
    return f"Let me check.\n\n```python\n{code}\n```"


executor = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
executor.check_sandboxed()


def new_agent(replies, **kwargs) -> tuple[Agent, ScriptedLLM]:
    llm = ScriptedLLM(replies)
    agent = Agent(llm=llm, executor=executor, system_prompt="(test)", **kwargs)
    agent.new_analysis()
    return agent, llm


# 1. The basic loop: code -> result -> code -> result -> answer.
agent, llm = new_agent([
    py("import pandas as pd\nemp = pd.read_csv('/data/employees.csv')\nprint(emp.shape)"),
    py("print((emp['salary'] < 100_000).sum())"),   # uses emp from the previous step
    "240 employees earn under $100k.",
], require_approval=False)
events = list(agent.send("How many employees earn under 100k?"))
results = [e for e in events if isinstance(e, CodeResult)]
check("loop runs to an answer", isinstance(events[-1], TurnDone) and events[-1].reason == "answered")
check("two steps executed", len(results) == 2, f"steps={agent.steps}")
check("real data loaded", "(412, 6)" in results[0].result["stdout"], results[0].result["stdout"].strip())
check("state carries over", results[1].result["stdout"].strip() == "240", results[1].result["stdout"].strip())
check("model saw the result", "[Execution result]" in llm.seen[1][-1]["content"]
      and "(412, 6)" in llm.seen[1][-1]["content"])

# 2. Approval: the loop pauses, then runs the code after "yes".
agent, llm = new_agent([py("print('approved code ran')"), "Done."])
events = list(agent.send("go"))
check("pauses for approval", isinstance(events[-1], NeedsApproval) and agent.pending_code)
events = list(agent.resume(approved=True))
check("runs after approval", any(isinstance(e, CodeResult) and "approved code ran" in e.result["stdout"]
                                 for e in events))

# 3. Decline with feedback: nothing runs, the model gets the feedback and tries again.
agent, llm = new_agent([py("import os\nos.listdir('/')"), py("print('second try')"), "Done."])
list(agent.send("go"))
events = list(agent.resume(approved=False, feedback="only look at /data"))
check("declined code not run", not any(isinstance(e, CodeResult) and "os.listdir('/')" in e.code
                                       for e in events))
check("model got feedback", "only look at /data" in llm.seen[1][-1]["content"])
check("model tried again", isinstance(events[-1], NeedsApproval) and "second try" in events[-1].code)

# 4. Plain decline ends the turn.
agent, llm = new_agent([py("print(1)")])
list(agent.send("go"))
events = list(agent.resume(approved=False))
check("plain decline ends turn", events[-1] == TurnDone("declined"))

# 5. Two code blocks in one reply: only the first runs, and the model is told.
agent, llm = new_agent([py("print('first')") + "\n" + py("print('second')"), "Done."],
                       require_approval=False)
events = list(agent.send("go"))
outputs = [e.result["stdout"].strip() for e in events if isinstance(e, CodeResult)]
check("only first block runs", outputs == ["first"], str(outputs))
check("user notified", any(isinstance(e, Notice) for e in events))
check("model told", "Only the first was run" in llm.seen[1][-1]["content"])

# 6. Step limit: stops at max_steps and asks for a summary instead of more code.
agent, llm = new_agent([py("print(1)")] * 3 + ["Summary so far: nothing yet."],
                       require_approval=False, max_steps=3)
events = list(agent.send("go"))
check("step limit enforced", agent.steps == 3 and events[-1] == TurnDone("step_limit"),
      f"steps={agent.steps}")
check("asked for summary", "step limit" in llm.seen[-1][-1]["content"])

# 7. Errors reach the model as text, and the loop keeps going.
agent, llm = new_agent([py("pd.read_csv('/data/nope.csv')"), "That file doesn't exist."],
                       require_approval=False)
events = list(agent.send("go"))
check("error passed to model", "Error:" in llm.seen[1][-1]["content"])

# 8. Long output is cut to the limit before the model sees it.
agent, llm = new_agent([py("print('x' * 50_000)"), "Done."], require_approval=False,
                       max_result_chars=3000)
events = list(agent.send("go"))
seen = llm.seen[1][-1]["content"]
check("output trimmed for model", len(seen) < 3500 and "characters cut" in seen, f"{len(seen)} chars")

# 9. A code block that was never closed (reply cut off) is not run.
agent, llm = new_agent(["```python\nprint('half a program'"], require_approval=False)
events = list(agent.send("go"))
check("unclosed block ignored", not any(isinstance(e, CodeResult) for e in events))

# 10. Refuses an executor that isn't sandboxed. MockTransport is httpx's fake
#     server: it answers requests with this function instead of the network.
fake = httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok"}))
try:
    ExecutorClient("http://127.0.0.1:8000", 60, transport=fake).check_sandboxed()
    check("refuses unsandboxed executor", False)
except ExecutorNotSafe:
    check("refuses unsandboxed executor", True)

# 11. The same error over and over: nudge on the 3rd, stop on the 5th.
from agent.loop import NUDGE_AT, STOP_AT, error_signature

failing = py("df = pd.DataFrame({'a': [1]})\ndf['missing_col']")
agent, llm = new_agent([failing] * 6 + ["never reached"], require_approval=False)
events = list(agent.send("go"))
results = [e for e in events if isinstance(e, CodeResult)]
nudged = [i for i, e in enumerate(results) if "Stop trying variations" in e.model_text]
check("nudged on 3rd repeat", nudged == [NUDGE_AT - 1], f"nudged at attempt(s) {[i + 1 for i in nudged]}")
check("stopped after 5 repeats", len(results) == STOP_AT and events[-1] == TurnDone("stuck"),
      f"{len(results)} runs, ended {events[-1]}")
check("user told it stopped", any(isinstance(e, Notice) and "Stopped" in e.text for e in events))
check("model told why it stopped", "[Stopped]" in agent.history[-1]["content"])
# A new message starts with clean counts:
events = list(new_agent([py("print('fine')"), "Done."], require_approval=False)[0].send("go"))
check("counts reset each message", events[-1] == TurnDone("answered"))

# Different errors don't count together.
check("error fingerprint", error_signature({"status": "error", "error": "Traceback...\n  x\nKeyError: 'a'\n"})
      == "KeyError: 'a'")
check("sandbox failures not counted", error_signature(
    {"status": "error", "error": "x", "executor_failure": True}) is None)

# 12. Sandbox failures come back as results, never as a crash.
def fake_server(status: int):
    def answer(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "sandboxed": True})
        return httpx.Response(status, json={"detail": "boom"})
    return ExecutorClient("http://127.0.0.1:8000", 60, transport=httpx.MockTransport(answer))

r = fake_server(500).execute("print(1)")
check("HTTP 500 becomes an error result", r["status"] == "error" and r.get("executor_failure")
      and "internal error" in r["error"])

def unreachable(request):
    raise httpx.ConnectError("refused")
r = ExecutorClient("http://127.0.0.1:8000", 60, transport=httpx.MockTransport(unreachable)).execute("x")
check("unreachable sandbox becomes an error result", r.get("executor_failure") and "could not be reached" in r["error"])

agent, llm = new_agent([py("print(1)"), "Done."], require_approval=False)
agent.executor = fake_server(500)
events = list(agent.send("go"))
check("loop survives a sandbox failure", events[-1] == TurnDone("answered"))
check("user told about sandbox failure", any(isinstance(e, Notice) and "Sandbox problem" in e.text
                                              for e in events))

# 13. shorten() keeps the start and the end.
s = shorten("A" * 100 + "Z" * 100, 60)
check("shorten keeps both ends", s.startswith("A") and s.endswith("Z"))

# 13b. A long report loses detail lines, never checklist lines.
from agent.executor import shorten_report
report = "\n".join(["load: a = a.csv, 26 rows", "=== a: 26 rows x 30 columns ==="]
                   + [f"col{i}: 'x' 5, 'y' 3, 'z' 1, plus a long list of example values {i}" for i in range(60)]
                   + ["Issues to report (3):", "  1. col1: 2 values with stray spaces",
                      "  2. day_rate: likely an error: 1 value (75,000)", "  3. start_date: impossible dates: 1 value"]
                   + ["=== b: 76 rows x 5 columns ==="]
                   + [f"det{i}: lots of detail about this column {i}" * 2 for i in range(40)]
                   + ["Issues to report (1):", "  1. week_start: 1 date in the future, latest 2030-03-02"])
kept = shorten_report(report, 2500)
check("report: fits the limit", len(kept) <= 2600, f"{len(kept)} chars")
check("report: every issue line kept", all(line in kept for line in report.splitlines()
                                            if line.startswith("  ") and ". " in line[:6]))
check("report: headers and load kept", "=== b: 76 rows" in kept and kept.startswith("load: a"))
check("report: cut marked", "detail lines cut; every 'Issues to report' line is kept" in kept)
check("plain output: old cut", "characters cut" in shorten_report("x" * 9000, 5000))

# 14. A plain answer ends the turn, figures or not: the counts come from
#     check()'s report, so the model is never made to recompute them.
agent, llm = new_agent(["The salary of 1,250,000 is likely an error."], require_approval=False)
events = list(agent.send("Which salary is wrong?"))
check("answer without code ends the turn", len(llm.seen) == 1 and events[-1] == TurnDone("answered"))

# 15. Invented results after a code block never reach the history.
fake = ("```python\nprint('real')\n```\n\n[Execution result]\n```\nexpenses: 150 rows\n```\n\n"
        "So there are 150 rows.\n\n```python\nprint('second')\n```")
agent, llm = new_agent([fake, "Done."], require_approval=False)
events = list(agent.send("go"))
stored = agent.history[2]["content"]
check("reply cut after first block", stored.endswith("```") and "150 rows" not in stored, repr(stored[-30:]))
check("user sees the cut reply too", all("150 rows" not in e.text for e in events if isinstance(e, ModelReply)))
check("model told why", "everything after it was removed" in llm.seen[1][-1]["content"])

one_block = "```python\nprint('saving')\n```\nSaved all files to output/clean."
agent, llm = new_agent([one_block, "Done."], require_approval=False)
list(agent.send("go"))
check("invented summary after one block cut", "Saved all files" not in agent.history[2]["content"])
check("model told not to write results", "Never write results" in llm.seen[1][-1]["content"])

agent, llm = new_agent([fake, "Done."])               # approval mode: note still delivered
list(agent.send("go"))
list(agent.resume(approved=True))
check("note delivered in approval mode", "everything after it was removed" in llm.seen[1][-1]["content"])

agent, llm = new_agent(["Plain answer, no code. It has 3 parts.", "x"], require_approval=False)
list(agent.send("go"))
check("reply without code kept whole", agent.history[2]["content"] == "Plain answer, no code. It has 3 parts.")

# 16. Another program resets the shared kernel mid-conversation: the code is
#     NOT run, the model is told why, and the next block runs normally.
agent, llm = new_agent([py("x = 5"), "Set.", py("print(x)"),
                        py("x = 5\nprint('reloaded', x)"), "Done."], require_approval=False)
list(agent.send("set x"))
executor.reset()                                   # "the tests", from outside
events = list(agent.send("print x"))
results = [e for e in events if isinstance(e, CodeResult)]
check("code not run after outside reset", "NOT run" in (results[0].result["error"] or ""))
check("model told to load again", "another program reset" in results[0].model_text)
check("user told", any(isinstance(e, Notice) and "Another program" in e.text for e in events))
check("next block runs in the new kernel", "reloaded 5" in results[1].result["stdout"])
agent, llm = new_agent([py("print(1)"), "ok"], require_approval=False)   # its own reset only
events = list(agent.send("go"))
check("own reset is not 'another program'", not any(isinstance(e, Notice) and "Another program" in e.text
                                                    for e in events))
# 17. An error part-way through a block: the model is told which fixes already
#     ran, so it continues from the failing line instead of repeating them.
from agent.loop import applied_note
failed = {"status": "error", "error": "CleaningError: not real dates",
          "stdout": "[4] t: strip_spaces 'name': 2 cells changed\n[5] t: to_numbers 'rate': 9 cells changed\n"}
check("applied note: names the fixes", "2 fixes ran and were logged ([4] to [5] above)" in applied_note(failed),
      applied_note(failed))
check("applied note: one fix", "1 fix ran and was logged ([4] above)" in applied_note(
      {**failed, "stdout": "[4] t: strip_spaces 'name': 2 cells changed\n"}))
check("applied note: none when nothing was logged", applied_note({**failed, "stdout": "load: t, 5 rows\n"}) == "")
check("applied note: none on success", applied_note({**failed, "status": "ok"}) == "")
agent, llm = new_agent([py("print('[1] t: drop_rows: 1 row removed')\nraise ValueError('boom')"), "ok"],
                       require_approval=False)
list(agent.send("go"))
check("applied note reaches the model", "don't run them again" in llm.seen[1][-1]["content"])

# 18. A reply that says the files were saved, when save_clean() never ran
#     since the last change: the model is told, once, and gets to save.
fix = py("print('[1] t: drop_rows: 1 row removed')")
agent, llm = new_agent([fix, "Done. The files have been saved to output/clean/.",
                        py("print('save_clean: clean/t.csv, 4 rows')"), "Saved: clean/t.csv."],
                       require_approval=False)
events = list(agent.send("fix and save"))
check("false 'saved' caught", "save_clean() has not run since the last change" in llm.seen[2][-1]["content"])
check("user told about it", any(isinstance(e, Notice) and "hadn't run" in e.text for e in events))
check("model then saved", agent.unsaved is False and events[-1] == TurnDone("answered"))
agent, llm = new_agent([fix, "The files have been saved.", "Sorry, nothing was saved yet."],
                       require_approval=False)
events = list(agent.send("go"))
check("checked once per turn, no loop", sum(isinstance(e, Notice) for e in events) == 1
      and events[-1] == TurnDone("answered"))
agent, llm = new_agent([py("print('[1] t: x\\nsave_clean: clean/t.csv')"), "The files have been saved."],
                       require_approval=False)
events = list(agent.send("go"))
check("real save: no false alarm", not any(isinstance(e, Notice) for e in events))
agent, llm = new_agent([fix, "Once you approve, the files will be saved."], require_approval=False)
check("a promise to save is not a claim", not any(isinstance(e, Notice) for e in agent.send("go")))

# 19. An outside reset before ANY of our code ran: nothing was lost, so the
#     code just runs in the new kernel, with no warning and no second approval.
agent, llm = new_agent([py("print('first run')"), "ok"], require_approval=False)
executor.reset()
events = list(agent.send("go"))
results = [e for e in events if isinstance(e, CodeResult)]
check("reset before first run: code runs", "first run" in results[0].result["stdout"])
check("reset before first run: no warning", not any(isinstance(e, Notice) for e in events))