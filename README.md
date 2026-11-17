# Cleanup bot

A local data-cleaning agent. Upload CSV files and it checks them for problems, proposes numbered fixes in plain language, and applies **only the fixes you approve**. It then saves the cleaned files with a PDF report of every change and a data dictionary.

Then ask questions of the cleaned tables in plain English on the **Ask questions** tab: a second agent writes SQL, runs it read-only, and answers with the number, how it got it, and the query it ran.

Everything runs on your own machine: a local model through [Ollama](https://ollama.com), and every line of model-written code runs in a locked-down Docker sandbox. No data leaves the computer.

## What it does

1. **Checks** each file and every link between files: stray spaces, numbers stored as text, mixed and impossible dates, future dates, placeholder values like `-` or `unknown`, spelling variants, possible abbreviations, duplicate rows, repeated IDs, rows pointing at IDs that don't exist, and outliers, each labelled *likely an error* or *plausible*.
2. **Proposes** a numbered list of fixes and asks when a choice is yours (day or month first? drop the row or set the value to missing?).
3. **Applies** only what you approve, and shows you each block of code before it runs.
4. **Saves** the cleaned CSVs, a `change_log.json`, a PDF cleaning report and a `data_dictionary.md`.

## How it works

```mermaid
flowchart LR
    U[You, in the browser] -->|upload, chat, approve| UI[ui container<br/>Gradio app + agent loop]
    UI -->|prompts| M[Ollama on the host<br/>gemma4:12b]
    M -->|reply with a python block| UI
    UI -->|code you approved,<br/>read-only SQL| EX[executor container<br/>FastAPI + Jupyter kernel<br/>+ DuckDB]
    EX -->|output| UI
    EX -->|cleaned files, report| O[(output/clean)]
    D[(data, read-only)] --> EX
```

The model writes Python in fenced code blocks; the agent runs each block in the sandbox and feeds the output back, until the model replies without code.

### Design decisions

- **Judgement lives in tested code, not in the model.** The model doesn't write its own checks. It calls `check()`, which returns a numbered checklist with verdicts and ready-made conditions (e.g. `salary: likely an error ... selecting it with the condition (salary > 306600).fillna(False)`). The model's job is to relay the checklist, ask your questions, and call the right fix helper.
- **No change without a log entry.** Every fix goes through a helper that applies the change and logs exactly what changed (rows removed, cells changed, before → after examples). `save_clean()` refuses to write a table that differs from its last logged version, so an unlogged change can't reach the output.
- **Two layers of approval.** You approve the numbered fixes in plain language, then each block of code before it runs (this can be switched off).
- **Guards around the model.** Replies are cut after the first code block, so invented results never reach the history. Long reports drop detail lines but never checklist lines. A reply that claims files were saved when `save_clean()` never ran is caught. After an error, the model is told which fixes already ran.
- **The report is written by code.** The PDF and the data dictionary are generated from the change log, never by the model, so they can't contain an invented number.

## Asking questions (the SQL agent)

The **Ask questions** tab answers questions like *"What is the average money spent per employee?"* over this session's tables. The model writes a `sql` block, the query runs at once (no approval needed, since it can't change anything) and appears in the chat as a collapsible "Ran a query" section with its first rows. The model then replies with:

- **Answer**: the number, copied from the query result;
- **How**: what it counted and any assumption, in words (e.g. "employees with no expenses count as zero").

The full result of the last successful query is shown in **Last result** below the chat.

**Which tables.** Only this session's: tables cleaned and saved on the Clean tab (and not removed), plus CSVs uploaded on the Ask tab, which are used as they are and marked *uploaded*. The UI sends that file list with every query, so tables from earlier sessions or tests in `output/clean/` never show up. Cleaned tables get notes from `data_dictionary.md` (what was changed, which columns link tables). Removing a saved file on the Clean tab removes its table here too; **New session** resets both tabs.

**Read-only, in four layers.** Queries run in DuckDB inside the executor container (`POST /sql`), never in the UI:

1. Only one statement, and only `SELECT` or `WITH`. `INSERT`, `DELETE`, `CREATE`, `SET`, `ATTACH` and `;`-joined statements are refused.
2. The listed CSVs are copied into an in-memory database, then file access is switched off and the configuration locked, so `COPY ... TO` or `read_csv('/etc/passwd')` fail. Only files under `output/clean/` and `data/uploads/` can be loaded at all.
3. Even a write could only change the in-memory copy, never the CSV.
4. The sandbox container itself (below).

Queries time out after 30 seconds and return at most 200 rows. **A number check** runs on every final reply: a number that no query returned sends the reply back once with a note, so the model re-queries instead of inventing a figure.

### The sandbox

The executor container has no network route to the internet or the host, mounts `data/` read-only, has a read-only filesystem, runs as an unprivileged user with all Linux capabilities dropped, and is limited to 4 GB of memory, 2 CPUs and 256 processes. Its port and the UI are published on `127.0.0.1` only. The agent refuses to send code to an executor that doesn't report it is running in the sandbox.

## Running it

**Requirements:** Docker, and Ollama with the model pulled:

```bash
ollama pull gemma4:12b
```

Start the sandbox and the UI from the repo root, then open http://127.0.0.1:7860:

```bash
docker compose up -d --build
```

Upload CSV files and ask, for example, *"Check these files and tell me what you would fix."* The cleaned files and reports appear under **Downloads** and in `output/clean/`. Then open the **Ask questions** tab and ask about them. **New session** deletes the uploaded files and resets both tabs.

A terminal version is also available (it uses the CSV files in `data/`):

```bash
cd agent
uv run python -m agent.cli
```

Settings such as the model, context size and step limit are in `agent/src/agent/config.py`, and each can be overridden with an `ANALYST_` environment variable (e.g. `ANALYST_MODEL`).

## Tests and evaluation

Unit and integration tests need the containers running (`docker compose up -d`). The executor's checks and fix helpers run inside the sandbox image:

```bash
docker compose run --rm --no-deps -v "${PWD}/executor/test_helpers.py:/app/test_helpers.py:ro" executor python test_helpers.py
```

(The same for `test_cleaning.py` and `test_sql.py`. From Git Bash, prefix `MSYS_NO_PATHCONV=1` and use `$(pwd -W)`.) The others run from their own folders, e.g. `cd agent && uv run python test_loop.py`.

| Suite | Tests | What it covers |
|---|---|---|
| `executor/test_helpers.py` | 111 | every check reports every planted problem, with the right counts |
| `executor/test_cleaning.py` | 76 | fix helpers, the change log, the save check, report and dictionary |
| `executor/test_sql.py` | 83 | the DuckDB engine: SELECT only, writes and file reads refused, allowed folders, timeout, row cap, reload on change |
| `executor/test_executor.py` | 23 | the kernel server and the `/sql` routes |
| `executor/test_sandbox.py` | 16 | that the sandbox really contains code and queries |
| `agent/test_loop.py` | 63 | the agent loop and its guards, with a scripted model |
| `agent/test_session.py` | 75 | the Clean tab's logic: uploads, remove, approval, downloads |
| `agent/test_sql_agent.py` | 95 | the SQL agent: query results for the model, errors, refused writes, the number check |
| `agent/test_sql_session.py` | 128 | the Ask tab's logic and the SQL prompt (schema, dictionary notes) |
| `agent/evals/test_scoring.py`, `test_harness.py` | 36, 14 | the cleaning eval's scorer and harness |
| `agent/evals/test_sql_scoring.py`, `test_sql_harness.py` | 45, 24 | the SQL eval's scorer and harness |

The **evaluation** (`agent/evals/`) replays six scenarios with the real model and scores each run: facts the first reply must state, and checks on the cleaned tables. The scenarios cover linked tables, partial approval (only some fixes allowed), ambiguous dates the model must ask about, a clean file that must not be changed, and a set of traps (outliers hidden in text, a future date among same-year dates, a repeated key, an abbreviation).

```bash
cd agent
uv run python evals/make_datasets.py
uv run python evals/run_eval.py --runs 3
```

Latest result for `gemma4:12b`: **309/309 checks, 6 scenarios × 3 runs.**

The **SQL evaluation** asks six questions of three known tables (written by `make_datasets.py` to `output/clean/sql_eval/`) and checks each answer against the same answer computed with pandas in the sandbox: an average with two possible readings (the How must name the one used), totals per group, a top 3 in order, counts per month, a column that doesn't exist (no invented number), and an ambiguous question (ask, or state the reading). Every query must be a SELECT, and no number in the reply may lack a source.

```bash
cd agent
uv run python evals/make_datasets.py
uv run python evals/run_sql_eval.py           # 6 scenarios x 3 runs
```

Latest result for `gemma4:12b`: **102/102 checks, 6 scenarios × 3 runs.**

The `expenses` scenario and some tests use sample files (`data/employees.csv`, `data/expenses.csv`) that are not included in the repository; the other five eval datasets are generated by `make_datasets.py`.

## Repository layout

```
compose.yaml             the three containers and their sandbox settings
executor/
  server.py              FastAPI server running code in a Jupyter kernel
  analyst_helpers.py     check(), profile(), compare_keys(): the checklist
  cleaning.py            load(), the fix helpers, the change log, save_clean()
  report.py              PDF cleaning report and data dictionary
  sql.py                 read-only DuckDB engine behind /sql and /sql/schema
agent/
  src/agent/loop.py      the agent loop and its guards; SqlAgent and the number check
  src/agent/prompts.py   the cleaning system prompt
  src/agent/sql_prompts.py  the SQL prompt: schema plus data-dictionary notes
  src/agent/session.py   Clean tab state: uploads, chat messages, change log
  src/agent/sql_session.py  Ask tab state: tables, uploads, last result
  src/agent/ui.py        the Gradio app (Clean and Ask questions tabs)
  src/agent/cli.py       terminal version
  evals/                 scenarios, scoring, dataset generator, harness
```

## Built with

Python 3.14, pandas 3, DuckDB, FastAPI, Jupyter, Gradio, fpdf2, Ollama, Docker, uv.
