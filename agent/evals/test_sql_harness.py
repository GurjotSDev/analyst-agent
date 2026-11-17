"""
End-to-end check of the SQL eval harness, with fake models.

The fake model runs correct SQL, then answers by repeating the query result
under a How line. So if every scenario scores 100%:
  1. the plumbing works (agent, /sql, scoring, the run checks), and
  2. the answers computed with pandas in the sandbox agree with DuckDB
     running correct SQL on the same files: two independent engines.
Then fake models make specific mistakes, to prove each one is marked down.

Needs the sandbox running (docker compose up -d) and the tables
(uv run python evals/make_datasets.py). Run from agent/:
    uv run python evals/test_sql_harness.py
"""

import sys

from agent import config
from agent.executor import ExecutorClient
from run_sql_eval import make_agent, run_question
from sql_scenario import FILES, SQL_BY_NAME, SQL_SCENARIOS, compute_answers

failures = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global failures
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


class HonestSqlLLM:
    """Runs the given queries one per reply, then answers with `how` plus
    the last query result, copied as it came. With no queries, answers with
    `how` alone."""

    def __init__(self, queries: list[str], how: str):
        self.queries, self.how, self.last = list(queries), how, ""

    def chat(self, messages: list[dict]) -> str:
        last = messages[-1]["content"]
        if last.startswith("[Query result]"):
            self.last = last.split("\n", 1)[-1]
        if self.queries:
            return f"```sql\n{self.queries.pop(0)}\n```"
        return f"{self.how}\n\n{self.last}".strip()


AVG_SPENDERS = "SELECT ROUND(SUM(amount) / COUNT(DISTINCT employee_id), 2) AS avg_spend FROM expenses"
TOP3 = ("SELECT e.name, SUM(x.amount) AS total FROM expenses x JOIN employees e USING (employee_id) "
        "GROUP BY e.name ORDER BY total DESC LIMIT 3")
BY_PROJECT = "SELECT project, SUM(hours) AS total_hours FROM timesheets GROUP BY project ORDER BY project"
MONTHLY = ("SELECT strftime(expense_date, '%Y-%m') AS month, COUNT(*) AS expenses FROM expenses "
           "GROUP BY month ORDER BY month")
BEST_AVG = ("SELECT department, ROUND(AVG(salary), 2) AS average_salary FROM employees "
            "GROUP BY department ORDER BY average_salary DESC LIMIT 1")

# What a correct bot does for each scenario: (queries, How line).
CORRECT = {
    "avg_spend": ([AVG_SPENDERS], "**How:** expenses only, averaged over employees with at least one expense."),
    "by_project": ([BY_PROJECT], "**How:** timesheets grouped by project."),
    "top3": ([TOP3], "**How:** expenses joined to employees on employee_id, ranked by total."),
    "monthly": ([MONTHLY], "**How:** expenses counted per month of expense_date."),
    "no_column": ([], "There is no bonus column in these tables, so I can't answer that. "
                      "The closest column is salary in employees."),
    "ambiguous": ([BEST_AVG], "**How:** I read 'pays the most' as the highest average salary."),
}

executor = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
executor.check_sandboxed()
schema = executor.schema(FILES)
answers = compute_answers(executor, SQL_SCENARIOS)


def run(scenario: str, queries: list[str], how: str) -> list[str]:
    """The checks missed when a fake model runs these queries and says how."""
    agent = make_agent(HonestSqlLLM(queries, how), executor, schema)
    result = run_question(agent, SQL_BY_NAME[scenario], answers[scenario])[0]
    return [k for k, v in result["checks"].items() if not v]


# 0. The tables load, and the answers are what the data was built to give.
check("all 3 tables load", len(schema.get("tables", [])) == 3 and not schema.get("problems"), str(schema.get("problems")))
check("6 employees have no expenses", answers["avg_spend"]["n_spenders"] == 34)
check("the two readings of 'pays the most' disagree",
      answers["ambiguous"]["avg_dept"] != answers["ambiguous"]["total_dept"])

# 1. Correct work scores 100% on every scenario.
for scenario in SQL_SCENARIOS:
    queries, how = CORRECT[scenario.name]
    missed = run(scenario.name, queries, how)
    check(f"correct work: {scenario.name}", not missed, str(missed))

# 2. Other right answers: the all-employees reading, and asking back.
check("right: averaged over all 40, said so", not run("avg_spend", [
    "SELECT ROUND(SUM(amount) / (SELECT COUNT(*) FROM employees), 2) AS avg_spend FROM expenses"],
    "**How:** averaged over all employees, counting those with no expenses as zero."))
check("right: total salary, said so", not run("ambiguous", [
    "SELECT department, SUM(salary) AS total_salary FROM employees GROUP BY department "
    "ORDER BY total_salary DESC LIMIT 1"], "**How:** I read 'pays the most' as the largest total payroll."))
check("right: asks total or average", not run("ambiguous", [],
      "Do you mean the highest total salary bill, or the highest average salary?"))


# 3. Mistakes, each caught by the right check.
def mistake(name: str, scenario: str, queries: list[str], how: str, expected: set[str]) -> None:
    missed = run(scenario, queries, how)
    check(name, expected <= set(missed), str(missed))


AVG_CHECK = "right average for the employees it says it counted"
mistake("all-employee number, but says 'with expenses'", "avg_spend",
        ["SELECT ROUND(SUM(amount) / (SELECT COUNT(*) FROM employees), 2) AS avg_spend FROM expenses"],
        "**How:** averaged over employees with at least one expense.", {AVG_CHECK})
mistake("average expense row, not per employee", "avg_spend",
        ["SELECT ROUND(AVG(amount), 2) AS avg_spend FROM expenses"],
        "**How:** averaged over employees with at least one expense.", {AVG_CHECK})
mistake("right number, no word on who was counted", "avg_spend", [AVG_SPENDERS],
        "**How:** from the expenses table.", {AVG_CHECK})
mistake("bottom 3 instead of top 3", "top3", [TOP3.replace("DESC", "ASC")], "",
        {"the right 3 in the right order"})
mistake("right 3, wrong order", "top3", [f"SELECT * FROM ({TOP3}) ORDER BY name"], "",
        {"the right 3 in the right order"})
mistake("counted employees, not expenses", "monthly",
        [MONTHLY.replace("COUNT(*)", "COUNT(DISTINCT employee_id)")], "", {"2025-04: 61"})
mistake("hours averaged, not summed", "by_project", [BY_PROJECT.replace("SUM(", "AVG(")], "",
        {"Apollo: 1088 hours"})
mistake("invented a bonus figure", "no_column", [],
        "There is no bonus column. A typical bonus would be about 5,000 per department; salary is the closest.",
        {"no invented number"})
mistake("tried to change data", "by_project", ["DELETE FROM timesheets WHERE hours > 40", BY_PROJECT], "",
        {"every query was a SELECT"})
mistake("no query ever worked", "by_project", ["SELECT project, SUM(hourz) FROM timesheets GROUP BY project"],
        "Apollo: 1088 hours.", {"a query ran OK", "no invented number"})
mistake("silent pick: no word on average or total", "ambiguous",
        [BEST_AVG.replace("AS average_salary", "AS pay").replace("ORDER BY average_salary", "ORDER BY pay")],
        "The department that pays the most:", {"asks, or states the reading with its right answer"})
mistake("highest average, but says total", "ambiguous", [BEST_AVG.replace("AS average_salary", "AS pay")
        .replace("ORDER BY average_salary", "ORDER BY pay")], "**How:** the largest total salary.",
        {"asks, or states the reading with its right answer"})

print(f"\n{failures} failed")
sys.exit(1 if failures else 0)
