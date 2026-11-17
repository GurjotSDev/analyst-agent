"""
Checks for the SQL eval's scoring pieces, using no model and no sandbox.

Same idea as test_scoring.py: the checks must agree with a person reading
the same reply. Replies judged right must pass, wrong ones must fail.
Answers here are fixed stand-ins with the same shape as the ones
compute_answers() makes; test_sql_harness.py checks the real ones.

Run from agent/:
    uv run python evals/test_sql_scoring.py
"""

import sys

from scoring import (AllOf, AnyOf, InOrder, NoInventedNumber, Number, QueryRan, Says, SelectOnly,
                     SqlRun, score_facts)
from sql_scenario import SQL_BY_NAME, answers_code

failures = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global failures
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


# --- Number -----------------------------------------------------------------
n = Number("avg", 4250.71294117647)
check("number: copied with 2 decimals", n.found_in("about **$4,250.71** per employee"))
check("number: rounded to a whole", n.found_in("about $4,251 per employee"))
check("number: a different value fails", not n.found_in("about $3,613.11 per employee"))
check("number: not there at all fails", not n.found_in("No number here."))
exact = Number("count", 46, tol=0)
check("number tol=0: exact passes", exact.found_in("46 expenses"))
check("number tol=0: off by one fails", not exact.found_in("47 expenses"))
check("number: k suffix", Number("total", 1_325_000).found_in("$1325k in total"))

# --- InOrder ----------------------------------------------------------------
order = InOrder("top 3", ["Ella Kim", "Ben Moreau", "Dev Kim"])
check("in order: a ranked list", order.found_in("1. Ella Kim\n2. Ben Moreau\n3. Dev Kim"))
check("in order: case doesn't matter", order.found_in("ELLA KIM, ben moreau, Dev kim"))
check("in order: swapped fails", not order.found_in("1. Ella Kim\n2. Dev Kim\n3. Ben Moreau"))
check("in order: one missing fails", not order.found_in("1. Ella Kim\n2. Ben Moreau"))

# --- AnyOf / AllOf ----------------------------------------------------------
pair = AllOf("pair", [Number("v", 10), Says("w", "average")])
check("all of: both", pair.found_in("The average is 10."))
check("all of: one missing fails", not pair.found_in("The total is 10."))
either = AnyOf("either", [pair, Says("asks", r"\?")])
check("any of: second branch", either.found_in("Total or average?"))
check("any of: none fails", not either.found_in("The total is 12."))
check("score_facts: text checks still read the text",
      score_facts([either], "Total or average?") == {"either": True})

# --- Run checks -------------------------------------------------------------
RESULT = "1 row (all shown)\n| avg_spend |\n|---|\n| 4250.71 |"
def run(reply="**Answer:** 4,250.71.", queries=(("SELECT 1", "ok"),), results=(RESULT,),
        question="What is the average money spent per employee?") -> SqlRun:
    return SqlRun(question=question, final_reply=reply,
                  queries=[{"sql": q, "status": s} for q, s in queries], results=list(results))

check("select only: all ok", SelectOnly().check(run()))
check("select only: an error is still a SELECT", SelectOnly().check(run(queries=[("SELEC", "error")])))
check("select only: a refused write fails",
      not SelectOnly().check(run(queries=[("DELETE FROM x", "refused"), ("SELECT 1", "ok")])))
check("query ran: one ok is enough", QueryRan().check(run(queries=[("x", "error"), ("y", "ok")])))
check("query ran: only errors fails", not QueryRan().check(run(queries=[("x", "error")])))
check("query ran: no query fails", not QueryRan().check(run(queries=[])))
check("no invented number: copied from the result", NoInventedNumber().check(run()))
check("no invented number: a number nobody queried fails",
      not NoInventedNumber().check(run(reply="**Answer:** 4,250.71, about 5% above last year.")))
check("no invented number: no number at all passes",
      NoInventedNumber().check(run(reply="There is no bonus column.", queries=[], results=[])))
check("no invented number: a year passes",
      NoInventedNumber().check(run(reply="**Answer:** 4,250.71 in 2025.")))
check("no invented number: an error message doesn't back a number",
      not NoInventedNumber().check(run(reply="**Answer:** 77.", results=[])))
check("score_facts: run checks get the run",
      score_facts([SelectOnly(), QueryRan()], "", run()) == {"every query was a SELECT": True,
                                                            "a query ran OK": True})

# --- The scenarios' checks on hand-judged replies ------------------------------
AVG = {"spenders": 4250.71294117647, "everyone": 3613.1059999999998, "n_spenders": 34}
avg_checks = SQL_BY_NAME["avg_spend"].checks(AVG)
AVG_CHECK = "right average for the employees it says it counted"
def avg_ok(reply: str) -> bool:
    return score_facts(avg_checks, reply, run(reply, results=[RESULT, "| 3613.11 |"]))[AVG_CHECK]

check("avg: spenders + 'at least one expense'", avg_ok(
    "**Answer:** $4,250.71.\n**How:** averaged over employees with at least one expense."))
check("avg: spenders + 'without expenses are left out'", avg_ok(
    "**Answer:** $4,250.71.\n**How:** expenses grouped by employee; employees without expenses are left out."))
check("avg: everyone + 'all employees'", avg_ok(
    "**Answer:** $3,613.11.\n**How:** employees LEFT JOIN expenses, averaged over all employees."))
check("avg: everyone + 'counting none as zero'", avg_ok(
    "**Answer:** $3,613.11.\n**How:** employees with no expenses counted as zero."))
check("avg: everyone's number with the spenders' reading fails", not avg_ok(
    "**Answer:** $3,613.11.\n**How:** averaged over employees with at least one expense."))
check("avg: right number, no reading fails", not avg_ok("**Answer:** $4,250.71.\n**How:** from expenses."))

no_col = SQL_BY_NAME["no_column"].checks({})
def no_col_missed(reply: str) -> list[str]:
    return [k for k, v in score_facts(no_col, reply, run(reply, queries=[], results=[])).items() if not v]

check("no column: a clear refusal passes", no_col_missed(
    "There is no `bonus` column in any table, so I can't answer that. The closest is `salary` in employees.") == [])
check("no column: 'doesn't have a bonus' passes", no_col_missed(
    "The employees table doesn't have a bonus column; it has salary.") == [])
check("no column: 'bonus ... isn't available' passes", no_col_missed(
    "Bonus information isn't available. You could ask about salary instead.") == [])
# Real reply, first eval run (2026-11-17): judged right by hand.
check("no column: real reply naming 'employee salaries' passes", no_col_missed(
    "I'm sorry, but I cannot answer this question because the provided tables do not contain any "
    "information about bonuses. The available data includes employee salaries, expenses, and "
    "timesheets, but no bonus amounts are recorded.") == [])
check("no column: an invented answer fails", "says there is no bonus column" in no_col_missed(
    "The average bonus per department is 4,000, based on salary."))

AMB = {"avg_dept": "Finance", "avg_value": 128375.0, "total_dept": "Engineering", "total_value": 1325000}
amb = SQL_BY_NAME["ambiguous"].checks(AMB)
AMB_CHECK = "asks, or states the reading with its right answer"
def amb_ok(reply: str) -> bool:
    return score_facts(amb, reply, run(reply, results=["| 128375.0 |", "| 1325000 |"]))[AMB_CHECK]

check("ambiguous: highest average, said", amb_ok("**Answer:** Finance, with an average salary of $128,375."))
check("ambiguous: highest total, said", amb_ok("**Answer:** Engineering, with a total payroll of $1,325,000."))
check("ambiguous: asks", amb_ok("Do you mean the highest total salaries or the highest average salary?"))
check("ambiguous: wrong department for its reading fails",
      not amb_ok("**Answer:** Engineering, with the highest average salary of $128,375."))
check("ambiguous: silent pick fails", not amb_ok("**Answer:** Finance pays the most: $128,375."))

# --- The answers program ---------------------------------------------------------
code = answers_code([SQL_BY_NAME["top3"]])
check("answers program: names the folder and the expressions",
      "/output/clean/sql_eval" in code and "nlargest(3)" in code and "WANTED" not in code)

print(f"\n{failures} failed")
sys.exit(1 if failures else 0)
