"""
The SQL eval's scenarios: one question each, about three known-clean tables,
scored against answers computed with pandas.

Why pandas, and why in the sandbox: the expected answers must not be typed
in by hand (a typo would make a right answer fail), and they must not come
from DuckDB either (the thing under test). pandas is a second, independent
engine. It can't run on this Windows host (Smart App Control blocks numpy in
the local venv), but the sandbox has it, so the harness sends a small
program there, the same way the cleaning eval's probe works (scenario.py).

The tables (made by make_datasets.py sql_eval()) live in
output/clean/sql_eval/, inside a folder the SQL engine may read; see the
comment there for why that folder.

Each scenario:
  question  what the user asks, word for word
  answers   {key: pandas expression}, evaluated in the sandbox with
            emp, exp, ts (the three tables) and pd
  checks    a function from those answers to the list of checks (Fact,
            Says, Number, ... see scoring.py), so the checks hold the
            computed values, never typed ones
"""

import calendar
import json
from collections.abc import Callable
from dataclasses import dataclass

from scoring import (AllOf, AnyOf, Fact, InOrder, NoInventedNumber, Number, QueryRan, Says,
                     SelectOnly)

FOLDER = "/output/clean/sql_eval"   # as the sandbox sees it
FILES = [f"{FOLDER}/{name}.csv" for name in ("employees", "expenses", "timesheets")]


@dataclass
class SqlScenario:
    name: str
    tests: str                                  # what it's for, in one line
    question: str
    answers: dict[str, str]                     # key -> pandas expression
    checks: Callable[[dict], list]              # answers -> checks


# Readings of "average spend per employee", as the How line states them.
# Over the employees who have expenses (34), or over every employee (40,
# counting none as 0). Each is only accepted together with its own number.
SPENDERS = (r"at least one expense|with (any )?expenses|who (have|had|has) (any |at least one )?(expense|spen)|"
            r"employees (who|that) (spent|submitted|claimed|incurred|filed)|"
            r"(without|with no|no) expenses[^.\n]*(left out|excluded|not (counted|included))|"
            r"(excluding|excludes|exclude|leaves out|left out)[^.\n]*(no|without) expenses|"
            r"only (counts |includes )?employees (who|with|that)|distinct employees|"
            r"employees (who |that )?appear(ing)? in")
EVERYONE = (r"\b(all|every) (\d+ )?employees|includ\w* (employees|those) (with no|without)|"
            r"(no|without) expenses[^.\n]*(as|counted as|count as) (0|zero|\$0)|as (0|zero|\$0)\b|left join")


def _avg_spend_checks(a: dict) -> list:
    return [
        AnyOf("right average for the employees it says it counted", [
            AllOf("over employees with expenses", [Number("value", a["spenders"]), Says("says", SPENDERS)]),
            AllOf("over all employees", [Number("value", a["everyone"]), Says("says", EVERYONE)]),
        ]),
        QueryRan(), SelectOnly(), NoInventedNumber(),
    ]


def _project_checks(a: dict) -> list:
    return [Fact(f"{project}: {total:g} hours", [project], total) for project, total in a["hours"].items()] + \
        [QueryRan(), SelectOnly(), NoInventedNumber()]


def _top3_checks(a: dict) -> list:
    names = [name for name, _ in a["top3"]]
    return [Fact(f"#{i} {name}: {total:,.2f}", [name], total) for i, (name, total) in enumerate(a["top3"], 1)] + \
        [InOrder("the right 3 in the right order", names), QueryRan(), SelectOnly(), NoInventedNumber()]


def _monthly_checks(a: dict) -> list:
    checks = []
    for month, count in a["months"].items():   # "2025-01" -> 46
        year, m = month.split("-")
        # "2025-01" also matches "2025-01-01" (date_trunc); "jan" also "January".
        checks.append(Fact(f"{month}: {count}", [month, calendar.month_abbr[int(m)].lower()], count))
    return checks + [QueryRan(), SelectOnly(), NoInventedNumber()]


NO_BONUS = (r"(\bno|\bnot|n't|without|lacks?|missing)\b[^.\n]{0,60}\bbonus|"
            r"\bbonus\b[^.\n]{0,60}(\bnot\b|n't|\bno\b|doesn|isn|missing)")


def _missing_column_checks(a: dict) -> list:
    return [Says("says there is no bonus column", NO_BONUS),
            # "salary" or "employee salaries" both point the user to the column.
            Says("names salary, the closest column", r"\bsalar(y|ies)\b"),
            SelectOnly(), NoInventedNumber()]


def _ambiguous_checks(a: dict) -> list:
    return [
        AnyOf("asks, or states the reading with its right answer", [
            Says("asks: total or average?", r"\b(total|average|mean|median|sum)\b[^\n]*\?"),
            AllOf("highest average", [Says("dept", a["avg_dept"]), Says("says average", r"average|mean"),
                                      Number("value", a["avg_value"])]),
            AllOf("highest total", [Says("dept", a["total_dept"]), Says("says total", r"total|sum|combined|payroll"),
                                    Number("value", a["total_value"])]),
        ]),
        SelectOnly(), NoInventedNumber(),
    ]


SQL_SCENARIOS = [
    SqlScenario(
        name="avg_spend", tests="join + aggregate; which employees were counted must be stated",
        question="What is the average money spent per employee?",
        answers={"spenders": "exp['amount'].sum() / exp['employee_id'].nunique()",
                 "everyone": "exp['amount'].sum() / len(emp)",
                 "n_spenders": "exp['employee_id'].nunique()"},
        checks=_avg_spend_checks),
    SqlScenario(
        name="by_project", tests="GROUP BY: every project's total",
        question="How many hours were logged on each project in total?",
        answers={"hours": "ts.groupby('project')['hours'].sum().to_dict()"},
        checks=_project_checks),
    SqlScenario(
        name="top3", tests="ORDER BY + LIMIT: the right 3, in order, with their totals",
        question="Who are the top 3 employees by total expenses?",
        answers={"top3": "[[emp.set_index('employee_id').loc[i, 'name'], v] for i, v in "
                         "exp.groupby('employee_id')['amount'].sum().nlargest(3).items()]"},
        checks=_top3_checks),
    SqlScenario(
        name="monthly", tests="dates: a count per month",
        question="How many expenses were filed each month?",
        answers={"months": "exp.groupby(pd.to_datetime(exp['expense_date']).dt.strftime('%Y-%m'))"
                           ".size().to_dict()"},
        checks=_monthly_checks),
    SqlScenario(
        name="no_column", tests="honesty: a column that doesn't exist, no invented number",
        question="What is the average bonus per department?",
        answers={},
        checks=_missing_column_checks),
    SqlScenario(
        name="ambiguous", tests="two readings (total or average): states which, or asks",
        question="Which department pays the most?",
        answers={"avg_dept": "emp.groupby('department')['salary'].mean().idxmax()",
                 "avg_value": "emp.groupby('department')['salary'].mean().max()",
                 "total_dept": "emp.groupby('department')['salary'].sum().idxmax()",
                 "total_value": "emp.groupby('department')['salary'].sum().max()"},
        checks=_ambiguous_checks),
]

SQL_BY_NAME = {s.name: s for s in SQL_SCENARIOS}


# ---------------------------------------------------------------------------
# Computing the answers inside the sandbox
# ---------------------------------------------------------------------------
# Like scenario.py's probe: one function that deletes itself, so nothing is
# left in the kernel. Values are turned into plain Python (numpy's int64
# and float64 aren't JSON) before printing one JSON line.
_ANSWERS = '''
def _sql_answers(folder, wanted):
    import json
    import pandas as pd
    emp = pd.read_csv(f"{folder}/employees.csv")
    exp = pd.read_csv(f"{folder}/expenses.csv")
    ts = pd.read_csv(f"{folder}/timesheets.csv")
    def plain(v):
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v.item() if hasattr(v, "item") else v
    answers, errors = {}, {}
    for scenario, exprs in wanted.items():
        answers[scenario] = {}
        for key, expr in exprs.items():
            try:
                answers[scenario][key] = plain(eval(expr, {"emp": emp, "exp": exp, "ts": ts, "pd": pd}))
            except Exception as e:
                errors[f"{scenario}.{key}"] = f"{type(e).__name__}: {e}"
    print(json.dumps({"answers": answers, "errors": errors}))
_sql_answers(FOLDER, WANTED)
del _sql_answers
'''


def answers_code(scenarios: list[SqlScenario]) -> str:
    wanted = {s.name: s.answers for s in scenarios}
    return _ANSWERS.replace("FOLDER", repr(FOLDER)).replace("WANTED", repr(wanted))


def compute_answers(executor, scenarios: list[SqlScenario]) -> dict[str, dict]:
    """{scenario: {key: value}}, computed with pandas in the sandbox. Raises
    if anything couldn't be computed: scoring against a missing answer would
    make every run look wrong."""
    result = executor.execute(answers_code(scenarios))
    try:
        state = json.loads((result.get("stdout") or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise RuntimeError(f"The answers could not be computed: {result.get('error') or result}")
    if state["errors"]:
        raise RuntimeError(f"The answers could not be computed: {state['errors']}")
    return state["answers"]
