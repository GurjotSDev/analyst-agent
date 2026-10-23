"""
Stage 4 checks for the agent loop, WITHOUT a real model.

A ScriptedLLM plays back fixed replies, so each test is fast and gives the same
result every time. Everything else is real: the loop, the code extraction, and
the sandboxed executor running code on the fake data in data/.

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

# 11. Charts without a title or axis labels: the model is told to redraw.
from agent.loop import missing_chart_parts

bare = "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.bar(['a', 'b'], [3, 5])\nplt.show()"
agent, llm = new_agent([py(bare), "Done."], require_approval=False)
events = list(agent.send("go"))
seen = llm.seen[1][-1]["content"]
check("unlabelled chart flagged", "missing a title and an x-axis label and a y-axis label" in seen)
check("user told about redraw", any(isinstance(e, Notice) and "redraw" in e.text for e in events))
# Delete charts the tests created.
for e in events:
    if isinstance(e, CodeResult):
        for c in e.result["charts"]:
            executor.execute(f"import os\nos.remove('charts/{c['id']}.png')")

labelled = bare.replace("plt.show()", "ax.set_title('B is largest')\nax.set_xlabel('Group')\n"
                                      "ax.set_ylabel('Count')\nplt.show()")
agent, llm = new_agent([py(labelled), "Done."], require_approval=False)
events = list(agent.send("go"))
check("labelled chart not flagged", "missing" not in llm.seen[1][-1]["content"])
for e in events:
    if isinstance(e, CodeResult):
        for c in e.result["charts"]:
            executor.execute(f"import os\nos.remove('charts/{c['id']}.png')")

# The text check itself, on the other common ways to set labels:
check("ax.set(...) counts", missing_chart_parts("ax.set(title='T', xlabel='x', ylabel='y')") == [])
check("df.plot(title=...) counts", missing_chart_parts("df.plot(title='T', xlabel='x', ylabel='y')") == [])
check("plt.title(...) counts", missing_chart_parts("plt.title('T'); plt.xlabel('x'); plt.ylabel('y')") == [])
check("'subtitle' is not a title", "a title" in missing_chart_parts("subtitle_text = 'x'"))

# 12. The same error over and over: nudge on the 3rd, stop on the 5th.
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

# 13. Sandbox failures come back as results, never as a crash.
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

# 14. shorten() keeps the start and the end.
s = shorten("A" * 100 + "Z" * 100, 60)
check("shorten keeps both ends", s.startswith("A") and s.endswith("Z"))