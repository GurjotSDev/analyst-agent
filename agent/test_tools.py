"""Does gpt-oss call a run_python tool correctly, with JSON arguments?"""

import ast
import ollama
import re
from test_models import TASKS, OPTIONS, client, OUTDATED_PATTERNS  # reuse what we already wrote


def run_python(code: str) -> str:
    """Run Python code in the analysis environment and return its output.

    Args:
        code: The Python code to execute.
    """
    return ""  # never called here; we only inspect the model's request


# Same idea as before, but the prompt points at the tool instead of code blocks.
TOOL_PROMPT = """You are a careful data analyst.
To run Python code, call the run_python tool. Call it once per step, then wait for the output.
Data files are in /data (read-only). Keep printed output short.
pandas 3 is installed: assign with df.loc[...], never chain assignments or use inplace=True."""

# The same follow-up scenario, but recorded the way a tool-using agent would.
TOOL_TASKS = {
    "explore": TASKS["explore"],  # no history, so it's the same as before
    "follow-up": [
        {"role": "user",
         "content": "Two files are in /data: employees.csv and expenses.csv. "
                    "Explore them before we analyze anything."},
        # The model's earlier step, stored as a tool call instead of a code block.
        # "content" is empty because the step was the call itself, not text.
        {"role": "assistant",
         "content": "",
         "tool_calls": [{
             "function": {
                 "name": "run_python",
                 "arguments": {"code": "import pandas as pd\n"
                                       "emp = pd.read_csv('/data/employees.csv')\n"
                                       "exp = pd.read_csv('/data/expenses.csv')\n"
                                       "emp.info()\nexp.info()"},
             },
         }]},
        # The executor's output, returned in a "tool" message.
        # tool_name says which tool this result belongs to.
        {"role": "tool",
         "tool_name": "run_python",
         "content": "emp: 412 rows. columns: employee_id (int64), name (str), "
                    "department (str), salary (int64). No missing values.\n"
                    "exp: 5,830 rows. columns: expense_id (int64), employee_id (int64), "
                    "category (str), amount (float64), date (str). 12 missing in category."},
        # Now the user's actual request, as a separate user message.
        {"role": "user",
         "content": "Now calculate the total expense cost per employee, and add a column "
                    "to emp that marks employees earning under $100k."},
    ],
}

RUNS = 10

for task_name, messages in TOOL_TASKS.items():
    failures = 0
    for run in range(1, RUNS + 1):
        try:
            response = client.chat(
                model="gpt-oss:20b",
                messages=[{"role": "system", "content": TOOL_PROMPT}, *messages],
                tools=[run_python],
                think="medium",
                options=OPTIONS,
            )
        except ollama.ResponseError as e:
            failures += 1
            print(f"{task_name} #{run}: ERROR {e.error[:80]}")
            continue  # skip to the next run; nothing below applies to a failed request

        calls = response.message.tool_calls or []
        if not calls:
            failures += 1
            print(f"{task_name} #{run}: no tool call. Reply: {response.message.content[:80]!r}")
            continue

        for call in calls:
            code = call.function.arguments.get("code", "")

            # Both variables are assigned here, right before the print that uses them.
            try:
                ast.parse(code)
                ok = True
            except SyntaxError:
                ok = False
            issues = [reason for pattern, reason in OUTDATED_PATTERNS
                      if re.search(pattern, code)]

            print(f"{task_name} #{run}: tool={call.function.name} "
                  f"syntax_ok={ok} issues={issues}")

    print(f"== {task_name}: {failures}/{RUNS} failed ==\n")