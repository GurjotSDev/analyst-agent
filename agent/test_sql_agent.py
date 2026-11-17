"""
Checks for the SQL side of the agent: the executor client's sql()/schema(),
how a query result is written out for the model, the shared code-block
extractor, the number check, and SqlAgent's loop (with a scripted model).

Fake servers (httpx.MockTransport) cover the failure cases; a few checks use
the real sandbox. Needs the executor running (from the repo root:
docker compose up -d). Run from agent/:
    uv run python test_sql_agent.py
"""

import json

import httpx

from agent import codeblocks, config
from agent.executor import ExecutorClient, format_sql_for_model
from agent.loop import (CodeResult, NeedsApproval, Notice, SqlAgent, TurnDone, sql_error_signature,
                        unsourced_numbers)

PASSED, FAILED = 0, 0


def check(name: str, passed, detail: str = "") -> None:
    global PASSED, FAILED
    passed = bool(passed)
    PASSED, FAILED = PASSED + passed, FAILED + (not passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


def fake(handler) -> ExecutorClient:
    """A client talking to a pretend server: handler(request) -> httpx.Response."""
    return ExecutorClient("http://127.0.0.1:8000", 60, transport=httpx.MockTransport(handler))


def ok(columns, rows, row_count=None, exact=True) -> dict:
    """A /sql result in the server's shape."""
    count = len(rows) if row_count is None else row_count
    return {"status": "ok", "columns": columns, "rows": rows, "row_count": count,
            "count_exact": exact, "truncated": count > len(rows), "error": None}


# ---------------------------------------------------------------------------
# 1. The client
# ---------------------------------------------------------------------------

sent = {}


def server(request: httpx.Request) -> httpx.Response:
    sent["path"], sent["body"], sent["method"] = request.url.path, request.content, request.method
    sent["json"] = json.loads(request.content or b"{}")
    if request.url.path == "/sql":
        return httpx.Response(200, json=ok(["n"], [[1]]))
    return httpx.Response(200, json={"folder": "/output/clean", "problems": [],
                                     "tables": [{"name": "t", "file": "t.csv", "rows": 1,
                                                 "columns": [{"name": "n", "type": "BIGINT"}]}]})


client = fake(server)
r = client.sql("SELECT 1 AS n")
check("sql: posts the query to /sql", sent["path"] == "/sql" and b"SELECT 1 AS n" in sent["body"], str(sent))
check("sql: returns the server's result", r["status"] == "ok" and r["rows"] == [[1]], str(r))
check("sql: no files given -> an empty list is sent", sent["json"].get("files") == [], str(sent["json"]))
r = client.sql("SELECT 1 AS n", ["/output/clean/t.csv", "/data/uploads/x/ask/u.csv"])
check("sql: sends the file list", sent["json"] == {"query": "SELECT 1 AS n",
      "files": ["/output/clean/t.csv", "/data/uploads/x/ask/u.csv"]}, str(sent["json"]))
s = client.schema()
check("schema: POST /sql/schema", sent["path"] == "/sql/schema" and sent["method"] == "POST"
      and s["tables"][0]["name"] == "t", str(s))
check("schema: no files given -> an empty list is sent", sent["json"] == {"files": []}, str(sent["json"]))
client.schema(("/output/clean/t.csv",))
check("schema: sends the file list", sent["json"] == {"files": ["/output/clean/t.csv"]}, str(sent["json"]))


def unreachable(request):
    raise httpx.ConnectError("refused")


r = fake(unreachable).sql("SELECT 1")
check("sql: unreachable sandbox is a result, not a crash",
      r["status"] == "error" and r.get("executor_failure") and "reached" in r["error"], str(r))
s = fake(unreachable).schema()
check("schema: unreachable sandbox gives no tables and says why",
      s["tables"] == [] and s.get("error") and "reached" in s["error"], str(s))

r = fake(lambda req: httpx.Response(500, text="boom")).sql("SELECT 1")
check("sql: server error is a result", r["status"] == "error" and r.get("executor_failure"), str(r))
s = fake(lambda req: httpx.Response(500, text="boom")).schema()
check("schema: server error gives no tables", s["tables"] == [] and s.get("error"), str(s))

# ---------------------------------------------------------------------------
# 2. What the model reads
# ---------------------------------------------------------------------------

text = format_sql_for_model(ok(["name", "total"], [["Ava", 300.5], ["Liam", None]]))
check("format: row count line", text.startswith("2 rows"), text)
check("format: a header row", "| name | total |" in text, text)
check("format: values, missing shown as NULL", "| Ava | 300.5 |" in text and "| Liam | NULL |" in text, text)

text = format_sql_for_model(ok(["avg"], [[81666.66666666667]]))
check("format: full precision kept (the number check compares against it)", "81666.66666666667" in text, text)
check("format: one row says 1 row", text.startswith("1 row"), text)

text = format_sql_for_model(ok(["n"], []))
check("format: no rows says so", "0 rows" in text and "matched nothing" in text, text)

rows = [[i, f"name {i}"] for i in range(200)]
text = format_sql_for_model(ok(["id", "name"], rows, row_count=1000))
check("format: at most 50 rows go to the model", "| 49 | name 49 |" in text and "| 50 | name 50 |" not in text, "")
check("format: says how many rows exist and how many are shown",
      "1,000 rows" in text and "first 50 shown" in text, text.splitlines()[0])

text = format_sql_for_model(ok(["n"], [[1]] * 200, row_count=1_000_000, exact=False))
check("format: an inexact count says 'more than'", "more than 1,000,000 rows" in text, text.splitlines()[0])

wide = [["x" * 400, i] for i in range(50)]
text = format_sql_for_model(ok(["long", "i"], wide), max_chars=3000)
check("format: long rows are cut to the character budget", len(text) <= 3200, str(len(text)))
check("format: and it says rows were left out", "shown" in text.splitlines()[0], text.splitlines()[0])

text = format_sql_for_model({"status": "error", "error": "BinderException: column \"amout\" not found\n"
                                                         "Candidate bindings: \"amount\""})
check("format: an error keeps DuckDB's full message", text.startswith("Error:") and "Candidate" in text, text)

text = format_sql_for_model({"status": "refused", "error": "Refused: this is a DELETE statement. ..."})
check("format: a refusal says so", text.startswith("Refused:") and "DELETE" in text, text)

text = format_sql_for_model({"status": "timeout", "error": "The query was stopped after 30 seconds."})
check("format: a timeout says so", "stopped after 30 seconds" in text, text)

text = format_sql_for_model({"status": "error", "executor_failure": True, "error": "The sandbox could not be reached."})
check("format: sandbox failure passes the message on", "could not be reached" in text, text)

text = format_sql_for_model(ok(["note"], [["a | b\nc"]]))
check("format: pipes and line breaks inside values don't break the table", "a \\| b c" in text, text)

# ---------------------------------------------------------------------------
# 3. Against the real sandbox
# ---------------------------------------------------------------------------

real = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
real.check_sandboxed()
# A table of this test's own in output/clean/ (the sandbox sees it as
# /output/clean/...), deleted at the end. Only listed files are tables.
TEST_CSV = config.OUTPUT_DIR / "clean" / "zz_test_sql_agent.csv"
TEST_CSV.parent.mkdir(parents=True, exist_ok=True)
TEST_CSV.write_text("n\n1\n2\n3\n", encoding="utf-8")
TEST_FILES = ["/output/clean/zz_test_sql_agent.csv"]
r = real.sql("SELECT 6 * 7 AS answer")
check("real: query runs", r["status"] == "ok" and r["rows"] == [[42]], str(r))
r = real.sql("DELETE FROM zz_test_sql_agent", TEST_FILES)
check("real: write refused", r["status"] == "refused", str(r.get("error")))
s = real.schema(TEST_FILES)
check("real: schema lists the listed table", [t["name"] for t in s["tables"]] == ["zz_test_sql_agent"]
      and "error" not in s, str(s)[:200])
check("real: no files, no tables", real.schema()["tables"] == [])
r = real.sql("SELECT SUM(n) FROM zz_test_sql_agent", TEST_FILES)
check("real: a query on the listed table", r["status"] == "ok" and r["rows"] == [[6]], str(r))

# ---------------------------------------------------------------------------
# 4. Code blocks: one extractor for ```python and ```sql
# ---------------------------------------------------------------------------

both = "Plan.\n```python\nprint(1)\n```\nand\n```sql\nSELECT 1\n```\n```SQL\nSELECT 2\n```"
check("blocks: sql blocks found, python ignored", codeblocks.extract(both, "sql") == ["SELECT 1", "SELECT 2"],
      str(codeblocks.extract(both, "sql")))
check("blocks: python blocks found, sql ignored", codeblocks.extract(both, "python") == ["print(1)"]
      and codeblocks.extract_code(both) == ["print(1)"])
kept, cut = codeblocks.trim_after_first("Q.\n```sql\nSELECT 1\n```\nThe answer is 7.", "sql")
check("blocks: text after the first sql block is cut", kept.endswith("```") and cut == "The answer is 7.",
      repr((kept, cut)))
check("blocks: an unclosed sql block is ignored", codeblocks.extract("```sql\nSELECT 1", "sql") == [])
check("blocks: a SQL result is labelled [Query result]",
      codeblocks.result_message("1 row", "Query result")["content"] == "[Query result]\n1 row")
check("blocks: the default label is unchanged",
      codeblocks.result_message("x")["content"] == "[Execution result]\nx")

# ---------------------------------------------------------------------------
# 5. The number check, on its own
# ---------------------------------------------------------------------------

RESULTS = ["1 row (all shown)\n| total |\n|---|\n| 1520.43 |",
           "3 rows (all shown)\n| dept | avg |\n|---|---|\n| Sales | 81666.66666666667 |\n"
           "| IT | 0.125 |\n| Ops | 2024-03-15 |"]


def unsourced(reply, question="How much in total?"):
    return unsourced_numbers(reply, question, RESULTS)


check("numbers: $ and commas ignored, written precision matched", unsourced("Total is $1,520.4.") == [])
check("numbers: 1,520 matches 1520.43 (cut off)", unsourced("Total is 1,520.") == [])
check("numbers: 81,667 matches 81666.67 (rounded)", unsourced("About 81,667.") == [])
check("numbers: 81,666.67 matches at two decimals", unsourced("Exactly 81,666.67.") == [])
check("numbers: more digits than the result has is a miss", unsourced("Total 1520.44") == ["1520.44"])
check("numbers: a wrong whole number is a miss", unsourced("Total is 1,521.") == ["1,521"])
check("numbers: 82k matches 81666.67", unsourced("About 82k on average.") == [])
check("numbers: 1.2 million no query returned is a miss", unsourced("It is 1.2 million.") == ["1.2 million"])
check("numbers: 12.5% matches a 0.125 fraction", unsourced("IT is 12.5% of the total.") == [])
check("numbers: a negative is matched by size", unsourced("It fell by -1,520.4.") == [])
check("numbers: the row-count line backs a count", unsourced("There are 3 departments.") == [])
check("exempt: numbers in the question", unsourced("Over 1,000 there is one.", "Expenses over 1000?") == [])
check("exempt: years", unsourced("In 2019 and 2024 it was 1,520.4.") == [])
check("not exempt: a year-sized number with a comma", unsourced("It was 1,999.") == ["1,999"])
check("exempt: small counts after a ranking word", unsourced("The top 7 and the first 5:") == [])
check("not exempt: a small number with no ranking word", unsourced("Bob spent 7.") == ["7"])
check("not exempt: a big number after a ranking word", unsourced("The top 25 are:") == ["25"])
check("skipped: ids, list markers and inline code", unsourced("1. E001 used `LIMIT 10`\n2. Q3") == [])
check("skipped: fenced code", unsourced("Done.\n```text\nLIMIT 10\n```") == [])
check("numbers: each missing number listed once", unsourced("5, 5 and 6") == ["5", "6"])
# Step 6 eval: "employees with no expenses treated as 0" was sent back, though
# 0 there is how missing totals were counted, not a measured value.
AVG_ONLY = ["1 row (all shown)\n| avg |\n|---|\n| 3613.11 |"]
check("exempt: a zero stated as a convention", unsourced_numbers(
    "3613.11, with no expenses treated as 0 (or as $0).", "Average spend?", AVG_ONLY) == [])
check("not exempt: a zero that is a claim",
      unsourced_numbers("3613.11, and there were 0 refunds.", "Average spend?", AVG_ONLY) == ["0"])
check("not exempt: another number after 'as'",
      unsourced_numbers("3613.11, counting none as 7.", "Average spend?", AVG_ONLY) == ["7"])

# ---------------------------------------------------------------------------
# 6. SqlAgent, with a scripted model and the real sandbox
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Stands in for the model: plays back prepared replies, records what it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def sql(query: str) -> str:
    """A reply with one sql block, the way the model writes it."""
    return f"Let me query that.\n\n```sql\n{query}\n```"


def new_sql_agent(replies, **kwargs):
    llm = ScriptedLLM(replies)
    agent = SqlAgent(llm=llm, executor=real, system_prompt="(test)", **kwargs)
    agent.new_analysis()
    return agent, llm


def kinds(events, kind):
    return [e for e in events if isinstance(e, kind)]


# The basic turn: question -> sql -> result -> answer, with no approval pause.
kernel_before = real.kernel()
agent, llm = new_sql_agent([sql("SELECT 6 * 7 AS answer"), "The answer is 42."])
check("agent: new_analysis leaves the cleaning kernel alone", real.kernel() == kernel_before,
      f"{kernel_before} -> {real.kernel()}")
events = list(agent.send("What is six times seven?"))
results = kinds(events, CodeResult)
check("agent: runs without asking for approval", not kinds(events, NeedsApproval) and agent.pending_code is None)
check("agent: the query ran in the sandbox", len(results) == 1 and results[0].result["rows"] == [[42]],
      str(results[0].result if results else events))
check("agent: the model read a [Query result]", llm.seen[1][-1]["content"].startswith("[Query result]\n1 row")
      and "| 42 |" in llm.seen[1][-1]["content"], llm.seen[1][-1]["content"])
check("agent: the turn ends answered, with no notice",
      events[-1] == TurnDone("answered") and not kinds(events, Notice), str(events[-1]))

# A python block is not a query: the SQL agent doesn't run it.
agent, llm = new_sql_agent(["```python\nprint(1)\n```"])
events = list(agent.send("Hi"))
check("agent: a python block is not run", not kinds(events, CodeResult) and events[-1] == TurnDone("answered"))

# A DuckDB error goes back in full; the fixed query then runs.
agent, llm = new_sql_agent([sql("SELECT amout FROM (SELECT 3 AS amount)"),
                            sql("SELECT amount FROM (SELECT 3 AS amount)"), "The amount is 3."])
events = list(agent.send("What is the amount?"))
results = kinds(events, CodeResult)
check("error: the first query fails", results[0].result["status"] == "error", str(results[0].result))
check("error: the model sees DuckDB's message and hints",
      results[0].model_text.startswith("Error:") and "Candidate bindings" in results[0].model_text,
      results[0].model_text)
check("error: the fixed query runs", results[1].result["status"] == "ok" and results[1].result["rows"] == [[3]])
check("error: then the turn ends answered", events[-1] == TurnDone("answered") and agent.steps == 2)

# The agent sends its file list with every query (AskSession sets it).
agent, llm = new_sql_agent([sql("SELECT SUM(n) AS total FROM zz_test_sql_agent"), "The total is 6."])
check("files: empty by default", agent.files == [])
agent.files = list(TEST_FILES)
events = list(agent.send("What is the total?"))
results = kinds(events, CodeResult)
check("files: the query sees the listed table", results and results[0].result["rows"] == [[6]],
      str(results[0].result if results else events))
agent, llm = new_sql_agent([sql("SELECT SUM(n) AS total FROM zz_test_sql_agent"), "No such table."])
events = list(agent.send("What is the total?"))
check("files: without the list, the table doesn't exist", kinds(events, CodeResult)[0].result["status"] == "error")
recorded = []


def recording_server(request: httpx.Request) -> httpx.Response:
    recorded.append(json.loads(request.content))
    return httpx.Response(200, json=ok(["n"], [[1]]))


agent = SqlAgent(llm=ScriptedLLM([sql("SELECT 1 AS n"), sql("SELECT 1 AS n"), "n is 1."]),
                 executor=fake(recording_server), system_prompt="(test)")
agent.files = ["/output/clean/a.csv", "/data/uploads/s/ask/b.csv"]
list(agent.send("n?"))
check("files: sent with every query", len(recorded) == 2
      and all(r["files"] == ["/output/clean/a.csv", "/data/uploads/s/ask/b.csv"] for r in recorded), str(recorded))
agent.new_analysis()
check("files: kept by new_analysis (the tables don't change)", agent.files == ["/output/clean/a.csv", "/data/uploads/s/ask/b.csv"])

# A write is refused, and the data is unchanged.
target = "zz_test_sql_agent"
before = real.sql(f"SELECT count(*) FROM {target}", TEST_FILES)
agent, llm = new_sql_agent([sql(f"DELETE FROM {target}"), "I can't change data: the tables are read-only."])
agent.files = list(TEST_FILES)
events = list(agent.send(f"Delete everything in {target}"))
results = kinds(events, CodeResult)
check("refused: the write is refused", results[0].result["status"] == "refused"
      and results[0].model_text.startswith("Refused:"), results[0].model_text)
check("refused: the table is unchanged", real.sql(f"SELECT count(*) FROM {target}", TEST_FILES) == before
      and before["rows"] == [[3]], str(before))
check("refused: the turn ends answered", events[-1] == TurnDone("answered"))

# The same error again and again: nudge on the 3rd, stop on the 5th.
bad = sql("SELECT nope FROM (SELECT 1 AS yes)")
agent, llm = new_sql_agent([bad] * 5)
events = list(agent.send("go"))
results = kinds(events, CodeResult)
check("repeat: the signature is DuckDB's first line, not the ^^^ marker",
      list(agent.error_counts) == [sql_error_signature(results[0].result)]
      and "nope" in list(agent.error_counts)[0], str(agent.error_counts))
check("repeat: nudged on the 3rd, with the SQL hint", "schema" in results[2].model_text
      and "DataFrame" not in results[2].model_text, results[2].model_text[-300:])
check("repeat: stopped after the 5th", events[-1] == TurnDone("stuck") and len(results) == 5)
agent, llm = new_sql_agent([sql("SELECT a FROM (SELECT 1 AS x)"), sql("SELECT b FROM (SELECT 1 AS x)"),
                            "Neither column exists."])
list(agent.send("go"))
check("repeat: different errors are counted apart", len(agent.error_counts) == 2, str(agent.error_counts))
agent, llm = new_sql_agent([sql("DELETE FROM t")] * 5)
events = list(agent.send("go"))
check("repeat: refused writes count too, so the turn stops", events[-1] == TurnDone("stuck"))

# The step limit.
agent, llm = new_sql_agent([sql("SELECT 1"), sql("SELECT 2"), "Here is what I found."], max_steps=2)
events = list(agent.send("go"))
check("limit: stops at max_steps with a summary", events[-1] == TurnDone("step_limit") and agent.steps == 2
      and any("Step limit" in n.text for n in kinds(events, Notice)))

# Two blocks in one reply: only the first runs, and the model is told.
agent, llm = new_sql_agent([sql("SELECT 1 AS a") + "\n```sql\nSELECT 2 AS b\n```", "a is 1."])
events = list(agent.send("go"))
results = kinds(events, CodeResult)
check("trim: only the first sql block runs", len(results) == 1 and results[0].code == "SELECT 1 AS a")
check("trim: the note names sql blocks and [Query result]", "2 sql blocks" in results[0].model_text
      and agent.history[2]["content"].endswith("```"), results[0].model_text[-200:])

# The number check inside the loop.
agent, llm = new_sql_agent([sql("SELECT 1520.43 AS total"), "**Answer:** $1,520.4 in total."])
events = list(agent.send("What is the total?"))
check("check: a number from the result passes", events[-1] == TurnDone("answered")
      and not kinds(events, Notice) and len(llm.seen) == 2)

agent, llm = new_sql_agent([sql("SELECT 1520.43 AS total"), "The total is 1,600.", "The total is 1,520.43."])
events = list(agent.send("What is the total?"))
notices = kinds(events, Notice)
check("check: an invented number sends the reply back", len(notices) == 1 and "1,600" in notices[0].text
      and len(llm.seen) == 3, str(notices))
check("check: the model is told which number",
      "Your reply states 1,600, which no query returned. Run a query for it or correct the reply."
      in llm.seen[2][-1]["content"], llm.seen[2][-1]["content"])
check("check: the corrected reply ends the turn", events[-1] == TurnDone("answered"))

agent, llm = new_sql_agent([sql("SELECT 1520.43 AS total"), "The total is 1,600.", "Still 1,600."])
events = list(agent.send("What is the total?"))
check("check: at most once per turn", len(kinds(events, Notice)) == 1 and events[-1] == TurnDone("answered")
      and not llm.replies)
llm.replies = ["It is 1,700.", "I would need to run a query for that."]
events = list(agent.send("And the other one?"))
check("check: a new turn checks again, and last turn's results don't count",
      len(kinds(events, Notice)) == 1 and "1,700" in kinds(events, Notice)[0].text, str(events))

agent, llm = new_sql_agent([sql("SELECT 77 AS x FROM nowhere"), "x is 77.", "I couldn't find it."])
events = list(agent.send("What is x?"))
check("check: numbers in a failed query's message don't count", len(kinds(events, Notice)) == 1)

agent, llm = new_sql_agent([sql("SELECT 4 AS n"), "Of the expenses over 1,000 in 2024, there are 4."])
events = list(agent.send("How many expenses over 1000 are there?"))
check("check: question numbers and years are exempt in the loop", not kinds(events, Notice))

TEST_CSV.unlink(missing_ok=True)
print(f"\n{PASSED} passed, {FAILED} failed")
raise SystemExit(1 if FAILED else 0)
