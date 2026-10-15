import ast
import re
import time
import ollama

client = ollama.Client(host="http://127.0.0.1:11434")

MODELS = {
    "gpt-oss:20b": "medium",
    "qwen3.5:9b": True,
}

RUNS_PER_TASK = 3

OPTIONS = {"num_ctx": 32768}

SYSTEM_PROMPT = """You are a careful data analyst working in a Python environment.

How to run code:
- To run code, write it in a fenced block that starts with ```python.
- Only ```python blocks are executed. Use another label (e.g. ```text) for anything that should not run.
- Write ONE python block per reply, then stop. you will receive its output in the next message.
- Variables persist between blocks, like a Jupyter notebook.
- Data files are in /data (read-ony). Save charts and files to /output.
- Keep printed output short: use .head(), .describe(), or summaries, never whle tables.

How to work:
- Before analyzing, explore each file: columns, dtypes, row counts, missing values, a few sample rows.
- If something is ambiguous (for example, two possible join columns), ask the user instead of guessing.

Library notes (the installed versions are new than most examples online):
- pandas 3: Copy-on-Write is always on. Assign with df.loc[rows, "col"] = value. Never chain like df[mask]["col"] = value; it silently does nothing.
- pandas 3: text clumns have a dtype "str", not "object". Use select_dtypes(include="str").
- pandas 3: never use inplace=True. Assign the result instead: df["col"] = df["col"].fillna(0).
- NumPy 2: np.float_, np.Nan and np.Inf were removed. Use np.float64, np.nan and np.inf.
"""

TASKS = {
    "explore": [
        {
            "role": "user",
            "content": "Two files are in /data: employees.csv and expenses.csv. "
                       "Explore them before we analyze anything",
        },
    ],

    "follow-up": [
        {
            "role": "user",
            "content": "Two files are in /data: employees.csv and expenses.csv. "
                        "Explore them before we analyze anything",
        },
        {
            "role": "assistant",
            "content": "Let me load both files and look a their structure.\n\n"
                       "```python\nimport pandas as pd\n"
                       "emp = pd.read_csv('/data/employees.csv')\n"
                       "exp = pd.read_csv('/data/expenses.csv')\n"
                       "emp.info()\nexp.info()\n```",
        },
        {
            "role": "user",
            "content": "Output:\n"
            "emp: 412 rows. columns: employee_id (int64), name (str), "
            "department (str), salary (int64). No missing values.\n"
            "exp: 5,830 rows. columns: expense_id (int64), employee_id (int64), "
            "category (str), amount (float64), date (str). 12 missing in category.\n\n"
            "Now calculate the total expense cost per employee, add a column "
            "to emp that marks employees earning under $100k."
        },
    ],
}

OUTDATED_PATTERNS = [
    (r"np\.(float_|NaN|Inf|PINF|NINF)\b",
     "NumPy alias removed in NumPy 2"),
    (r"select_dtypes\([^)]*['\"]object['\"]",
     "pandas 3 text columns are 'str', so 'object' misses them"),
    (r"\][ \t]*\[[^\]\n]+\][ \t]*=(?!=)",
     "chained assignment df[...][...] = ... silently fails in pandas 3"),
    (r"\][ \t]*\.\w+\([^)]*inplace\s*=\s*True",
     "inplace=True on a column does nothing in pandas 3"),
]

CODE_BLOCK = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)

def check_reply(content: str) -> dict:
    blocks = CODE_BLOCK.findall(content)

    syntax_ok = True
    for code in blocks:
        try:
            ast.parse(code)
        except SyntaxError:
            syntax_ok = False

    issues = [reason for pattern, reason in OUTDATED_PATTERNS if any(re.search(pattern, code) for code in blocks)]

    return {
        "blocks": len(blocks),
        "syntax_ok": syntax_ok,
        "issues": issues,
    }

def run_once(model: str, think, messages: list[dict]) -> dict:
    start = time.perf_counter()
    response = client.chat(
        model=model, 
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        think=think,
        options=OPTIONS,
    )
    seconds = time.perf_counter() - start

    tokens_per_sec = 0.0
    if response.eval_duration:
        tokens_per_sec = response.eval_count / (response.eval_duration / 1e9)

    result = check_reply(response.message.content or "")
    result.update(seconds=seconds, tokens_per_sec=tokens_per_sec, content=response.message.content or "")
    return result

def gpu_share(model: str) -> str:
    for loaded in client.ps().models:
        if loaded.model == model and loaded.size:
            return f"{loaded.size_vram / loaded.size:.0%} on GPU"
        return "not loaded"

def main() -> None:
    for model, think in MODELS.items():
        if "cloud" in model:
            raise SystemExit(f"Refusing to test {model}: cloud models are not allowed.")

        print(f"\n=== {model} ===")
        for task_name, messages in TASKS.items():
            for run in range(1, RUNS_PER_TASK + 1):
                try:
                    r = run_once(model, think, messages)
                except ollama.ResponseError as e:
                    hint = f"  (run: ollama pull {model})" if e.status_code == 404 else ""
                    print(f"  {task_name} #{run}: ERROR {e.error}{hint}")
                    break

                status = "OK " if r["blocks"] == 1 and r["syntax_ok"] and not r["issues"] else "BAD"
                print(f"  {task_name} #{run}: {status} blocks={r['blocks']} "
                      f"syntax_ok={r['syntax_ok']} {r['seconds']:.1f}s "
                      f"{r['tokens_per_sec']:.0f} tok/s")
                for issue in r["issues"]:
                    print(f"      outdated: {issue}")
 
                if run == 1:
                    print("  --- first reply ---")
                    print("  " + r["content"].replace("\n", "\n  "))
                    print("  -------------------")
 
        print(f"  VRAM: {gpu_share(model)}")
 
 
if __name__ == "__main__":
    main()