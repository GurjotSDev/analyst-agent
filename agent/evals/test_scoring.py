"""
Checks for the scorer, using no model and no sandbox.

The important part is the bottom half: real model replies. A scorer is only
useful if it agrees with a human reading the same reply, so replies judged
correct must pass and the wrong ones must fail.

Run from agent/:
    uv run python evals/test_scoring.py
"""

import json

from scenario import BY_NAME, score_probe
from scoring import Fact, NeverProposes, Says, numbers_in, score_facts

failures = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global failures
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


steps = {s.name: s for s in BY_NAME['expenses'].steps}

# --- Reading numbers --------------------------------------------------------
check("plain and comma numbers", numbers_in("6 rows, 5,833 rows") == [6, 5833])
check("dollars and decimals", numbers_in("$12,974.15") == [12974.15])
check("k and M suffixes", numbers_in("$306k and $1.25M") == [306_000, 1_250_000])
check("words: million, thousand", numbers_in("$1.25 million, 306 thousand") == [1_250_000, 306_000])
check("'12 meals' is not 12 million", numbers_in("12 meals") == [12])
check("IDs inside words ignored", numbers_in("row_006 and x2") == [])
check("table cells", numbers_in("| Sales | $12,818.79 | 87 |") == [12818.79, 87])

# --- Matching facts ---------------------------------------------------------
f = Fact("1,250,000 salary", ["salar"], 1_250_000)
check("same line matches", f.found_in("| Extreme salary | $1,250,000 |"))
check("written as 1.25M matches", f.found_in("Salary of $1.25M is likely an error"))
check("another number fails", not f.found_in("Salary: $306,600"))
check("number on another line fails", not f.found_in("Salary\n$1,250,000"))
check("keyword is case-insensitive", f.found_in("SALARY 1,250,000"))
check("exact count must be exact", not Fact("dup", ["duplicate"], 3).found_in("4 duplicate rows"))

# --- Real model replies -------------------------------------------------------
# Step 1 (explore): the model found all 8 problems. Must score 8/8.
GOOD_EXPLORE = """
| Missing salary values | 6 | None specified |
| Department spelling variants | 3 | 'engineering' vs 'Engineering' (123 vs 3 rows) |
| Extreme salaries (far from typical) | 5 | Most extreme: $1,250,000 (employee_id=1018), also $306,600 (1032) |
| Exact duplicate rows | 3 | Duplicate expense_id values |
| Missing category | 12 | NaN values in category column |
| Category spelling variants | 15 | 'meals' vs 'Meals' |
| Negative amounts | 9 | Refunds or data errors? |
- Employee ID **9999** appears in expenses (5 rows) but has no matching row in employees table
"""
scores = score_facts(steps["explore"].facts, GOOD_EXPLORE)
check("good explore summary: 8/8 counts", all(v for k, v in scores.items()
      if isinstance(next(f for f in steps["explore"].facts if f.name == k), Fact)))
# It never says which salary is an error, so the verdict check rightly fails
# it, and only that one.
check("summary without a verdict: only that check fails", [k for k, v in scores.items() if not v]
      == ["1,250,000 called a likely error"], str([k for k, v in scores.items() if not v]))

# The same summary with two numbers wrong must lose exactly those two.
bad = GOOD_EXPLORE.replace("| Missing category | 12 |", "| Missing category | 0 |") \
                  .replace("(5 rows)", "(4 rows)")
scores = score_facts(steps["explore"].facts, bad)
check("wrong counts are caught", [k for k, v in scores.items() if not v]
      == ["12 missing categories", "employee 9999: 5 rows", "1,250,000 called a likely error"])

# --- Says: a phrase instead of a number --------------------------------------
asks = Says("asks about order", r"day[ -]?first|month[ -]?first|day/month")
check("Says: finds the phrase", asks.found_in("Are these dates Day-first or month first?"))
check("Says: absent phrase fails", not asks.found_in("I converted the dates."))

# --- NeverProposes: a fix that must not be on the numbered list -------------
# A real proposal that breaks two rules:
bad = """I propose the following fixes:
5. Drop the 5 salary outliers in employees (values 1,250,000, 306,600, and 289,000) [drop_rows]
4. Fill 6 missing salary values in employees with 92400 (the median) [fill_missing]
8. Drop the 53 amount outliers in expenses (e.g., 6,372.55) [drop_rows]"""
good = """The other 53 large expenses look plausible, so I would not drop them.
1. Set the salary of 1,250,000 to missing (1 value), a likely error.
Should I fill the 6 missing salaries with the median? I'd leave them missing."""
no53 = NeverProposes("no 53", r"(drop|remov|delet|exclud)\w*\b.*\b53\b")
no5 = NeverProposes("no 5 salaries", r"(drop|remov|delet|exclud)\w*\b.*\b(5|five)\b.*salar")
nofill = NeverProposes("no guessed fill", r"fill\w*\b.*\b(median|mean|average|'other'|\"other\")")
check("NeverProposes: catches dropping 53", not no53.found_in(bad))
check("NeverProposes: catches dropping 5 salaries", not no5.found_in(bad))
check("NeverProposes: catches a median fill", not nofill.found_in(bad))
check("NeverProposes: unnumbered sentences don't count", no53.found_in(good) and nofill.found_in(good))
check("NeverProposes: a good proposal passes", no5.found_in(good))
check("NeverProposes: bold numbers count", not no53.found_in("**2.** Remove the 53 outliers"))
nothing = BY_NAME["clean"].steps[0].facts[0]
check("clean data: 'no issues' wordings accepted", all(nothing.found_in(t) for t in [
    "I found no data quality issues.", "The catalog looks clean.", "There is nothing to fix.",
    "Issues to report: none"]))
check("clean data: proposing fixes is not 'nothing to fix'",
      not nothing.found_in("1. Convert added_date to dates (80 rows)"))

# --- The probe's scorer -----------------------------------------------------
expenses = BY_NAME["expenses"]
def probe_output(results=None, saved=None, rows=None, save_dir="/output/evals/run1") -> str:
    """What the probe prints: everything correct unless told otherwise."""
    return json.dumps({
        "results": results if results is not None else {e.name: True for e in expenses.expects},
        "errors": {},
        "saved_rows": saved if saved is not None else {"employees": 412, "expenses": 5816},
        "table_rows": rows if rows is not None else {"employees": 412, "expenses": 5816},
        "save_dir": save_dir})

def failed(stdout: str) -> list[str]:
    checks, _ = score_probe(stdout, expenses)
    return [k for k, v in checks.items() if not v]

check("perfect cleaning: all probe checks pass", failed(probe_output()) == [])
one_wrong = {e.name: e.name != "refunds removed (not NaN)" for e in expenses.expects}
check("a failed expectation is reported", failed(probe_output(one_wrong)) == ["refunds removed (not NaN)"])
check("no output: everything fails", len(failed("")) == len(expenses.expects) + 1)
check("not saving is caught", failed(probe_output(saved={})) == ["cleaned files saved"])
check("saved file with the wrong rows is caught",
      failed(probe_output(saved={"employees": 412, "expenses": 5825})) == ["cleaned files saved"])
check("files outside the run's folder don't count",
      failed(probe_output(save_dir="/output")) == ["cleaned files saved"])
check("no save check when a scenario saves nothing",
      "cleaned files saved" not in score_probe(probe_output(), BY_NAME["clean"])[0])



# --- Real banking replies --------------------------------------------------
banking = BY_NAME["banking"].steps[0].facts
RUN2 = """1. Merge 'savings' into 'Savings' (4 rows) (fix: `merge_variants`)
2. Drop 3 exact duplicate rows in `transactions` (fix: `drop_duplicates`)
3. Regarding the 5 values in `transactions["amount"]` that are far beyond typical values (all > 2498.65), would you like to drop these rows or set those specific values to missing?
4. Regarding the future dates in `accounts["opened_date"]` (1 row) and `transactions["timestamp"]` (2 rows), are future dates allowed in these columns?"""
RUN3 = """3. For the 5 values in `amount` that are far beyond every other value (999,999.99), would you like to drop these rows or set the `amount` to missing?
4. Are future dates allowed in the `opened_date` (account) and `timestamp` (transaction) columns?"""
failed = lambda text: [k for k, v in score_facts(banking, text).items() if not v]
# Run 2 did report the 5 placeholders, in the report's words ("far beyond").
check("banking run 2: 'far beyond' counts as the sentinels", "5 sentinel amounts" not in failed(RUN2))
check("banking run 2: no year 2031 is still a miss", "account opened in 2031" in failed(RUN2))
# Run 3 really did skip them: no counts, no dates.
check("banking run 3: future dates without counts are a miss", {"2 future transactions",
      "account opened in 2031"} <= set(failed(RUN3)), str(failed(RUN3)))

print(f"\n{'All passed' if not failures else f'{failures} FAILED'}")