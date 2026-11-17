"""
Checks for the Ask questions tab's logic: the SQL prompt (sql_prompts.py) and
AskSession (sql_session.py), without a browser or a real model.

The prompt builder is checked with made-up schemas and dictionaries. AskSession
is checked like test_session.py: a scripted model plays back fixed replies, and
the sandbox is real (docker compose up -d). A few checks use a fake server
(httpx.MockTransport) for an empty or unreachable schema. Run from agent/:
    uv run python test_sql_session.py
"""

import shutil
import tempfile
from pathlib import Path

import httpx

from agent import config
from agent.executor import ExecutorClient
from agent.loop import Agent, SqlAgent
from agent.session import ChatDisplay, Session
from agent.sql_prompts import SQL_SYSTEM_PROMPT, build_sql_prompt, parse_dictionary
from agent.sql_session import AskSession, make_ask_session

PASSED, FAILED = 0, 0


def check(name: str, passed, detail: str = "") -> None:
    global PASSED, FAILED
    passed = bool(passed)
    PASSED, FAILED = PASSED + passed, FAILED + (not passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. The fixed part of the prompt
# ---------------------------------------------------------------------------

p = SQL_SYSTEM_PROMPT
check("prompt: read-only role", "DuckDB SQL" in p and "cannot change" in p)
check("prompt: one sql block per reply, then stop", "```sql" in p and "ONE sql block" in p and "[Query result]" in p)
check("prompt: Answer / How template", "**Answer:**" in p and "**How:**" in p)
check("prompt: no sql block in the final answer", "never repeat the SQL" in p)
check("prompt: arithmetic in SQL", "Never compute a number yourself" in p and "ROUND(AVG(" in p)
check("prompt: re-query on follow-ups", "follow-up" in p and "new query" in p)
check("prompt: a chosen cut-off goes in the query", "AS cutoff" in p)
check("prompt: how missing values were treated", "NULL" in p and "AVG" in p and "skip" in p)
check("prompt: joins use the Links to notes", "Links to" in p)
check("prompt: missing column -> say so, no guessing", "doesn't exist" in p and "closest columns" in p)
check("prompt: ask only when the readings differ materially", "materially different" in p)
# Step 6 eval: "Which department pays the most?" was once answered from the
# wrong table (summed expenses) with no word that it was a choice.
check("prompt: a word that could mean different columns or measures",
      "different columns or measures" in p and "state the reading in How" in p)
# Step 6 eval: an all-customers average came with How "sum divided by the
# count of rows", the formula only: the reader can't tell who was counted.
check("prompt: How says who was counted in words, not the formula",
      "in words, not as a formula" in p)
check("prompt: no LaTeX", "Never LaTeX" in p)
check("prompt: DuckDB notes", all(x in p for x in ["date_trunc", "strftime", "::DOUBLE", '"Order Total"']))
# Examples use made-up tables, never the eval data (the prompt must not
# hand the model the answers).
eval_words = ["employee", "expense", "contractor", "timesheet", "ticket", "salary", "shipment", "bank"]
check("prompt: examples use orders/customers", "orders" in p and "customers" in p)
check("prompt: no eval table names", not [w for w in eval_words if w in p.lower()],
      str([w for w in eval_words if w in p.lower()]))


# ---------------------------------------------------------------------------
# 2. Reading the data dictionary
# ---------------------------------------------------------------------------

# The exact header executor/report.py write_dictionary() writes. If that
# changes, the parser below must change with it.
HEADER = "| Column | Type | Missing | Distinct | Examples | Changes made | Links to | Description |"
check("dictionary: report.py still writes the header the parser expects",
      HEADER in (config.REPO_ROOT / "executor" / "report.py").read_text(encoding="utf-8"))

DICTIONARY = f"""# Data dictionary

Generated 2026-11-16 10:00 from the cleaned files in output/clean/. The Description column is left blank.

## customers

File: clean/customers.csv (from customers.csv), 3 rows x 3 columns.

{HEADER}
| --- | --- | --- | --- | --- | --- | --- | --- |
| customer_id | text | 0 | 3 | 'C1', 'C2', 'C3' | - | orders.customer_id (one row per ID here) |  |
| name | text | 0 | 3 | 'Ann', 'Bo' | strip_spaces (2 cells changed in 'name') | - |  |
| region | text | 1 | 2 | 'North \\| East', 'South' | merge_variants (1 cell changed in 'region'); placeholders_to_missing (1 cell changed in 'region') | - |  |

Changes to whole rows: drop_duplicates: 1 row removed (exact duplicate rows).

## orders

File: clean/orders.csv (from orders.csv), 5 rows x 3 columns.

{HEADER}
| --- | --- | --- | --- | --- | --- | --- | --- |
| order_id | whole number | 0 | 5 | 1, 2, 3 | - | - |  |
| customer_id | text | 0 | 3 | 'C1', 'C2' | - | customers.customer_id (many rows per ID here) |  |
| total | number | 1 | 4 | 10.5, 20.0 | set_missing (1 cell changed in 'total') | - |  |

## old_table

{HEADER}
| --- | --- | --- | --- | --- | --- | --- | --- |
| x | text | 0 | 1 | 'a' | strip_spaces (1 cell changed in 'x') | - |  |
"""

notes = parse_dictionary(DICTIONARY)
check("dictionary: one entry per table", sorted(notes) == ["customers", "old_table", "orders"], str(sorted(notes)))
check("dictionary: links", notes["customers"]["columns"]["customer_id"]["links"]
      == "orders.customer_id (one row per ID here)", str(notes["customers"]["columns"]["customer_id"]))
check("dictionary: changes", notes["orders"]["columns"]["total"]["changes"] == "set_missing (1 cell changed in 'total')")
check("dictionary: '-' means nothing", notes["orders"]["columns"]["order_id"] == {"changes": "", "links": ""},
      str(notes["orders"]["columns"]["order_id"]))
check("dictionary: whole-row changes", notes["customers"]["rows"]
      == "drop_duplicates: 1 row removed (exact duplicate rows)", repr(notes["customers"]["rows"]))
check("dictionary: no whole-row line -> empty", notes["orders"]["rows"] == "")
check("dictionary: an escaped | stays inside its cell",
      notes["customers"]["columns"]["region"]["changes"].startswith("merge_variants")
      and notes["customers"]["columns"]["region"]["links"] == "", str(notes["customers"]["columns"]["region"]))
check("dictionary: empty or missing text -> no notes", parse_dictionary("") == {})
# Columns are found by their header name, not their position.
moved = DICTIONARY.replace(HEADER, "| Column | Links to | Type | Missing | Distinct | Examples | Changes made | Description |")
moved = moved.replace("| customer_id | text | 0 | 3 | 'C1', 'C2', 'C3' | - | orders.customer_id (one row per ID here) |  |",
                      "| customer_id | orders.customer_id (one row per ID here) | text | 0 | 3 | 'C1' | - |  |")
check("dictionary: columns found by header name",
      parse_dictionary(moved)["customers"]["columns"]["customer_id"]["links"] == "orders.customer_id (one row per ID here)")


# ---------------------------------------------------------------------------
# 3. The schema block
# ---------------------------------------------------------------------------

SCHEMA = {"folder": "/output/clean", "problems": [], "tables": [
    {"name": "customers", "file": "customers.csv", "rows": 1,
     "columns": [{"name": "customer_id", "type": "VARCHAR"}, {"name": "region", "type": "VARCHAR"}]},
    {"name": "orders", "file": "orders.csv", "rows": 1200,
     "columns": [{"name": "order_id", "type": "BIGINT"}, {"name": "customer_id", "type": "VARCHAR"},
                 {"name": "total", "type": "DOUBLE"}, {"name": "Order Date", "type": "DATE"}]}]}

full = build_sql_prompt(SCHEMA, DICTIONARY)
check("schema: starts with the fixed prompt", full.startswith(SQL_SYSTEM_PROMPT))
check("schema: a heading per table with its row count",
      "### customers (1 row)" in full and "### orders (1,200 rows)" in full)
check("schema: every column with its type", "- order_id: BIGINT" in full and '- Order Date: DATE' in full)
check("schema: links from the dictionary",
      "- customer_id: VARCHAR. Links to customers.customer_id (many rows per ID here)" in full)
check("schema: changes from the dictionary",
      "- total: DOUBLE. Changes: set_missing (1 cell changed in 'total')" in full)
check("schema: whole-row changes",
      "Changes to whole rows: drop_duplicates: 1 row removed (exact duplicate rows)" in full)
check("schema: a table only in the dictionary is left out", "old_table" not in full)
check("schema: a column only in the dictionary is left out", "- name:" not in full)
check("schema: the tables block is at the end", full.index("## Tables") > full.index("## DuckDB notes"))

bare = build_sql_prompt(SCHEMA, "")
check("schema: no dictionary -> plain column lines",
      "- customer_id: VARCHAR\n" in bare and "Links to" not in bare.split("## Tables")[1])

empty = build_sql_prompt({"tables": [], "problems": []}, DICTIONARY)
tables_part = empty.split("## Tables")[1]
check("schema: no tables -> ask the user to clean and save first",
      "Clean and save files in the Clean tab first" in tables_part and "###" not in tables_part, tables_part)

broken = build_sql_prompt({"tables": [], "problems": [], "error": "The sandbox could not be reached (ConnectError)"}, "")
tables_part = broken.split("## Tables")[1]
check("schema: an error is passed on", "could not be read" in tables_part and "ConnectError" in tables_part, tables_part)

problems = build_sql_prompt(SCHEMA | {"problems": ["bad.csv: could not be read as CSV"]}, "")
check("schema: files that failed to load are listed", "bad.csv: could not be read as CSV" in problems)


# ---------------------------------------------------------------------------
# 4. AskSession with a scripted model and the real sandbox
# ---------------------------------------------------------------------------

class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


class BrokenLLM:
    def chat(self, messages):
        raise ConnectionError("Ollama is not running")


real = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
real.check_sandboxed()


def ask_session(replies, executor=real, output_dir=None) -> AskSession:
    agent = SqlAgent(llm=ScriptedLLM(replies), executor=executor, system_prompt="(test)")
    s = AskSession(agent, output_dir=output_dir)
    s.refresh_tables()
    return s


def sql(query: str) -> str:
    return f"Let me query that.\n\n```sql\n{query}\n```"


def working(s) -> list:
    return [m["metadata"]["title"] for m in s.messages if m.get("metadata", {}).get("status") == "pending"]


def titled(s, title) -> list:
    return [m for m in s.messages if m.get("metadata", {}).get("title") == title]


check("shared: both sessions use the same event-display code",
      issubclass(Session, ChatDisplay) and issubclass(AskSession, ChatDisplay)
      and AskSession._show is Session._show and AskSession._say is Session._say)

# Tables come from the real /sql/schema.
s = ask_session([])
real_tables = real.schema()["tables"]
check("tables: read from /sql/schema", s.tables == real_tables and s.schema_error is None,
      str([t["name"] for t in s.tables]))
check("tables: the system prompt lists them",
      all(f"### {t['name']} (" in s.agent.system_prompt for t in real_tables)
      and s.agent.history[0] == {"role": "system", "content": s.agent.system_prompt})

# A basic question: spinner, query, result, answer.
kernel_before = real.kernel()
s = ask_session([sql("SELECT 6 * 7 AS answer"), "**Answer:** 42.\n**How:** one multiplication."])
steps = s.send("What is six times seven?")
next(steps)
check("send: the question is drawn first", s.messages == [{"role": "user", "content": "What is six times seven?"}])
next(steps)
check("send: spinner while the model thinks", working(s) == ["Thinking…"], str(working(s)))
next(steps)
check("send: spinner while the query runs", working(s) == ["Running the query…"], str(working(s)))
list(steps)
check("send: no spinner at the end", working(s) == [])
check("send: the reply is shown without its sql block", s.messages[1] == {"role": "assistant", "content": "Let me query that."},
      str(s.messages[1]))
ran = titled(s, "Ran a query")
check("send: one 'Ran a query' section", len(ran) == 1, str([m.get("metadata") for m in s.messages]))
check("send: it holds the sql block", ran and "```sql\nSELECT 6 * 7 AS answer\n```" in ran[0]["content"])
check("send: and the rows as a Markdown table",
      ran and "1 row (all shown)\n\n| answer |\n|---|\n| 42 |" in ran[0]["content"], ran and ran[0]["content"])
check("send: the answer is shown last", s.messages[-1]["content"].startswith("**Answer:** 42."))
check("send: the query is shown once", sum("SELECT 6 * 7" in m["content"] for m in s.messages) == 1)
check("send: last_result has the columns and rows",
      s.last_result == {"query": "SELECT 6 * 7 AS answer", "columns": ["answer"], "rows": [[42]],
                        "row_count": 1, "truncated": False}, str(s.last_result))
check("send: the cleaning kernel is left alone", real.kernel() == kernel_before)

# Only the first 20 rows go in the chat; last_result keeps them all.
s = ask_session([sql("SELECT range AS n FROM range(30)"), "**Answer:** 30 numbers.\n**How:** range."])
list(s.send("List the numbers"))
shown = titled(s, "Ran a query")[0]["content"]
check("rows: the chat shows the first 20", "30 rows (first 20 shown)" in shown and "| 19 |" in shown
      and "| 20 |" not in shown, shown[-120:])
check("rows: last_result has every returned row", len(s.last_result["rows"]) == 30)

# A failing query, then a fixed one; a failure doesn't replace last_result.
s = ask_session([sql("SELECT 5 AS five"), "**Answer:** 5.\n**How:** a constant.",
                 sql("SELECT amout FROM (SELECT 3 AS amount)"), sql("SELECT amount FROM (SELECT 3 AS amount)"),
                 "**Answer:** 3.\n**How:** a constant."])
list(s.send("Five?"))
steps = s.send("What is the amount?")
for _ in steps:
    if titled(s, "The query failed"):
        break
check("error: titled 'The query failed'", len(titled(s, "The query failed")) == 1)
failed = titled(s, "The query failed")[0]["content"]
check("error: shows the query and DuckDB's message",
      "```sql\nSELECT amout" in failed and "```text\nError:" in failed and "amout" in failed.split("```text")[1], failed)
check("error: last_result still the last good result", s.last_result["rows"] == [[5]], str(s.last_result))
list(steps)
check("error: the fixed query runs", len(titled(s, "Ran a query")) == 2 and s.last_result["rows"] == [[3]])

# A write is refused.
s = ask_session([sql("DELETE FROM some_table"), "I can't change data: queries are read-only."])
list(s.send("Delete everything"))
refused = titled(s, "The query was refused")
check("refused: titled 'The query was refused'", len(refused) == 1, str([m.get("metadata") for m in s.messages]))
check("refused: says why", refused and "Refused" in refused[0]["content"], refused and refused[0]["content"])
check("refused: no last_result", s.last_result is None)

# The number check's note reaches the chat.
s = ask_session([sql("SELECT 4 AS n"), "**Answer:** There are 17.\n**How:** counted.",
                 "**Answer:** There are 4.\n**How:** counted."])
list(s.send("How many?"))
notes_shown = [m["content"] for m in s.messages if m["content"].startswith("*Note:")]
check("number check: shown as a note", len(notes_shown) == 1 and "17" in notes_shown[0], str(notes_shown))
check("number check: the corrected answer comes last", s.messages[-1]["content"].startswith("**Answer:** There are 4."))

# A failure (Ollama down) is shown, and the session stays usable.
s = ask_session([])
s.agent.llm = BrokenLLM()
list(s.send("Anything?"))
check("error: shown in the chat", s.messages[-1]["content"].startswith("**Error:** ConnectionError"),
      s.messages[-1]["content"])
check("error: no spinner left", working(s) == [])
s.agent.llm = ScriptedLLM(["Hello."])
list(s.send("Hi"))
check("error: still usable afterwards", s.messages[-1]["content"] == "Hello.")

# Blank messages are ignored.
s = ask_session([])
check("send: blank text does nothing", list(s.send("   ")) == [] and s.messages == [])

# new() clears the conversation only.
s = ask_session([sql("SELECT 1 AS one"), "**Answer:** 1.\n**How:** a constant."])
list(s.send("One?"))
tables_before, prompt_before, kernel_before = s.tables, s.agent.system_prompt, real.kernel()
s.new()
check("new: empty chat and no last result", s.messages == [] and s.last_result is None)
check("new: the model's history is only the system prompt",
      s.agent.history == [{"role": "system", "content": prompt_before}])
check("new: tables and prompt kept", s.tables == tables_before and s.agent.system_prompt == prompt_before)
check("new: the cleaning kernel is left alone", real.kernel() == kernel_before)


# ---------------------------------------------------------------------------
# 5. Refreshing the tables (a fake server for the schema)
# ---------------------------------------------------------------------------

current = {"schema": {"folder": "/output/clean", "problems": [], "tables": []}}


def server(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/sql/schema":
        return httpx.Response(200, json=current["schema"])
    return httpx.Response(404)


fake = ExecutorClient("http://127.0.0.1:8000", 60, transport=httpx.MockTransport(server))

s = ask_session(["Clean and save some files in the Clean tab first."], executor=fake)
check("no tables: tables empty, no error", s.tables == [] and s.schema_error is None)
check("no tables: the prompt says to clean and save first",
      "Clean and save files in the Clean tab first" in s.agent.system_prompt)

# The dictionary is read from <output_dir>/clean/data_dictionary.md.
out = Path(tempfile.mkdtemp())
(out / "clean").mkdir()
(out / "clean" / "data_dictionary.md").write_text(DICTIONARY, encoding="utf-8")
current["schema"] = SCHEMA
s = ask_session(["Hi."], executor=fake, output_dir=out)
check("dictionary: read from output/clean", "Links to customers.customer_id (many rows per ID here)"
      in s.agent.system_prompt and s.tables == SCHEMA["tables"])
s = ask_session([], executor=fake, output_dir=Path(tempfile.mkdtemp()))
check("dictionary: missing file -> plain schema", "### orders (1,200 rows)" in s.agent.system_prompt
      and "Links to customers" not in s.agent.system_prompt)

# Tables that change mid-conversation: new prompt, and the model is told.
s = ask_session(["**Answer:** none yet.\n**How:** -", "ok"], executor=fake, output_dir=out)
list(s.send("What tables are there?"))
history_before = len(s.agent.history)
s.refresh_tables()
check("refresh: unchanged tables -> no note", len(s.agent.history) == history_before)
current["schema"] = SCHEMA | {"tables": SCHEMA["tables"] + [
    {"name": "returns", "file": "returns.csv", "rows": 2, "columns": [{"name": "order_id", "type": "BIGINT"}]}]}
s.refresh_tables()
check("refresh: the system prompt is rebuilt", "### returns (2 rows)" in s.agent.history[0]["content"]
      and s.agent.history[0]["content"] == s.agent.system_prompt)
check("refresh: the model is told the tables changed",
      len(s.agent.history) == history_before + 1 and s.agent.history[-1]["role"] == "user"
      and s.agent.history[-1]["content"].startswith("[Tables]"), str(s.agent.history[-1]))

# An unreachable sandbox: no tables, and why.
down = ExecutorClient("http://127.0.0.1:8000", 60, transport=httpx.MockTransport(
    lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused"))))
s = ask_session([], executor=down)
check("schema error: kept for the UI", s.tables == [] and "could not be reached" in (s.schema_error or ""),
      str(s.schema_error))
check("schema error: the prompt says so", "could not be read" in s.agent.system_prompt)


# ---------------------------------------------------------------------------
# 6. make_ask_session: same model and sandbox as the cleaning session
# ---------------------------------------------------------------------------

llm = ScriptedLLM([])
clean = Session(Agent(llm=llm, executor=real, system_prompt="(test)"), output_dir=out)
ask = make_ask_session(clean)
check("make: an AskSession with a SqlAgent", isinstance(ask, AskSession) and isinstance(ask.agent, SqlAgent))
check("make: the same model and sandbox", ask.agent.llm is llm and ask.agent.executor is real)
check("make: the same output folder", ask.output_dir == out)
check("make: tables already read", ask.tables == real.schema()["tables"]
      and ask.agent.history[0]["content"] == ask.agent.system_prompt)
check("make: limits from config", ask.agent.max_steps == config.MAX_STEPS)

check("make: linked to the cleaning session", ask.clean is clean)


# ---------------------------------------------------------------------------
# 7. Step 5b: Ask-tab uploads, session-scoped tables, reset
# ---------------------------------------------------------------------------

def csv_file(name: str, text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(text, encoding="utf-8")
    return path


CLEAN_DIR = config.OUTPUT_DIR / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
# A table "saved by this session" (saved_tables is what refresh() would set),
# and one planted by another session / a test, which must never appear.
(CLEAN_DIR / "zz_ask_cleaned.csv").write_text("total\n1\n2\n", encoding="utf-8")
(CLEAN_DIR / "zz_planted.csv").write_text("secret\n99\n", encoding="utf-8")

clean = Session(Agent(llm=ScriptedLLM([]), executor=real, system_prompt="(test)"),
                data_dir=config.DATA_DIR, output_dir=config.OUTPUT_DIR)
clean.saved_tables = ["zz_ask_cleaned"]
ask = make_ask_session(clean)
check("scope: only this session's cleaned table", [t["name"] for t in ask.tables] == ["zz_ask_cleaned"],
      str([t["name"] for t in ask.tables]))
check("scope: a CSV planted in output/clean never appears", "zz_planted" not in ask.agent.system_prompt
      and all(t["name"] != "zz_planted" for t in ask.tables))
check("scope: the agent queries the same files", ask.agent.files == ["/output/clean/zz_ask_cleaned.csv"],
      str(ask.agent.files))
check("origin: a cleaned table is labelled cleaned", ask.table_origin == {"zz_ask_cleaned": "cleaned"},
      str(ask.table_origin))
check("uploads: the Ask folder is inside the session's upload folder", ask.upload_dir == clean.upload_dir / "ask")

# Upload a CSV on the Ask tab.
sales = csv_file("my sales.csv", "region,total\nNorth,10\nSouth,32\n")
added = ask.add_files([sales])
check("upload: copied into uploads/<id>/ask/ with a safe name", added == ["my_sales.csv"]
      and (clean.upload_dir / "ask" / "my_sales.csv").exists(), str(added))
check("upload: Gradio's temporary copy deleted", not sales.exists())
check("upload: listed", ask.files == ["my_sales.csv"] and ask.file_info == {"my_sales.csv": (2, 2)},
      f"{ask.files} {ask.file_info}")
check("upload: said in the chat", ask.messages[-1]["content"] == "*Uploaded my_sales.csv.*", ask.messages[-1]["content"])
check("upload: becomes a table, labelled uploaded",
      ask.table_origin == {"zz_ask_cleaned": "cleaned", "my_sales": "uploaded"}, str(ask.table_origin))
check("upload: the agent's files, uploads last",
      ask.agent.files == ["/output/clean/zz_ask_cleaned.csv", f"/data/uploads/{clean.session_id}/ask/my_sales.csv"],
      str(ask.agent.files))
check("upload: the prompt lists it", "### my_sales (2 rows)" in ask.agent.system_prompt)
check("upload: the cleaning tab's file list is untouched", clean.files == [])

# A question about the uploaded table runs against it.
ask.agent.llm = ScriptedLLM([sql("SELECT SUM(total) AS total FROM my_sales"), "**Answer:** 42.\n**How:** summed."])
list(ask.send("What is the total?"))
check("question on an upload: answered from it", ask.last_result and ask.last_result["rows"] == [[42]],
      str(ask.last_result))

# The same name again: replaced.
ask.add_files([csv_file("my_sales.csv", "region,total\nNorth,1\n")])
check("replace: listed once, new counts", ask.files == ["my_sales.csv"] and ask.file_info["my_sales.csv"] == (1, 2))
check("replace: said in the chat", ask.messages[-1]["content"] == "*Replaced my_sales.csv with a new version.*",
      ask.messages[-1]["content"])
check("replace: the model is told the tables changed", ask.agent.history[-1]["content"].startswith("[Tables]"))

# Non-CSV files are ignored.
check("upload: a non-CSV is ignored", ask.add_files([csv_file("notes.txt", "hi")]) == [] and ask.files == ["my_sales.csv"])

# A clash with a cleaned table: the upload wins, and the chat says so.
ask.add_files([csv_file("zz_ask_cleaned.csv", "total\n100\n")])
check("clash: said in the chat",
      ask.messages[-1]["content"] == "*zz_ask_cleaned.csv replaces the cleaned table zz_ask_cleaned for questions.*",
      ask.messages[-1]["content"])
check("clash: labelled uploaded", ask.table_origin.get("zz_ask_cleaned") == "uploaded", str(ask.table_origin))
check("clash: not listed as a problem", ask.problems == [], str(ask.problems))
check("clash: queries see the upload", real.sql("SELECT SUM(total) FROM zz_ask_cleaned", ask.agent.files)["rows"]
      == [[100]])

# Remove: the cleaned table comes back.
check("remove: an unknown name is refused", "isn't" in (ask.remove_file("nope.csv") or ""))
said = ask.remove_file("zz_ask_cleaned.csv")
check("remove: returns None when done", said is None, str(said))
check("remove: deleted from disk", not (ask.upload_dir / "zz_ask_cleaned.csv").exists())
check("remove: gone from the list", ask.files == ["my_sales.csv"] and "zz_ask_cleaned.csv" not in ask.file_info)
check("remove: the cleaned table is back", ask.table_origin.get("zz_ask_cleaned") == "cleaned"
      and real.sql("SELECT SUM(total) FROM zz_ask_cleaned", ask.agent.files)["rows"] == [[3]])
check("remove: said in the chat", ask.messages[-1]["content"] == "*Removed zz_ask_cleaned.csv.*")

# The cleaning session's table list drives the Ask tab: remove it there, gone here.
clean.saved_tables = []
ask.refresh_tables()
check("clean side removed: the cleaned table leaves", list(ask.table_origin) == ["my_sales"]
      and ask.agent.files == [f"/data/uploads/{clean.session_id}/ask/my_sales.csv"], str(ask.agent.files))

# reset(): New session empties both the chat and the uploads.
old_ask_dir = ask.upload_dir
clean.new()      # what New session does first: deletes data/uploads/<id>/, ask/ included
ask.reset()
check("reset: the old uploads are deleted", not old_ask_dir.exists())
check("reset: no uploads, no tables, no result", ask.files == [] and ask.file_info == {} and ask.tables == []
      and ask.table_origin == {} and ask.last_result is None, f"{ask.files} {ask.tables}")
check("reset: empty chat, history only the prompt", ask.messages == [] and len(ask.agent.history) == 1)
check("reset: the agent has no files", ask.agent.files == [])
check("reset: the prompt says there are no tables", "Clean and save files in the Clean tab first" in ask.agent.system_prompt)
check("reset: the next upload goes to the new session's folder",
      ask.upload_dir == clean.upload_dir / "ask" and ask.upload_dir != old_ask_dir)
clean.close()

# Dictionary notes only for cleaned tables: an upload named like an old
# cleaned table must not inherit its notes.
current["schema"] = {"problems": [], "tables": [
    {"name": "orders", "file": "/output/clean/orders.csv", "rows": 5,
     "columns": [{"name": "total", "type": "DOUBLE"}]},
    {"name": "customers", "file": "/data/uploads/abc/ask/customers.csv", "rows": 3,
     "columns": [{"name": "customer_id", "type": "VARCHAR"}, {"name": "name", "type": "VARCHAR"}]}]}
s = ask_session([], executor=fake, output_dir=out)
check("notes: on the cleaned table", "- total: DOUBLE. Changes: set_missing" in s.agent.system_prompt)
check("notes: none on the uploaded table", "- customer_id: VARCHAR\n" in s.agent.system_prompt
      and "Changes to whole rows" not in s.agent.system_prompt, s.agent.system_prompt.split("## Tables")[1])
check("origin: from the file path", s.table_origin == {"orders": "cleaned", "customers": "uploaded"}, str(s.table_origin))
from agent.sql_prompts import keep_tables
check("keep_tables: only the named sections", sorted(parse_dictionary(keep_tables(DICTIONARY, {"orders"}))) == ["orders"])

for f in ["zz_ask_cleaned.csv", "zz_planted.csv"]:
    (CLEAN_DIR / f).unlink(missing_ok=True)
shutil.rmtree(out, ignore_errors=True)
print(f"\n{PASSED} passed, {FAILED} failed")
raise SystemExit(1 if FAILED else 0)
