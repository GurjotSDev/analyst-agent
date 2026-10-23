"""
Checks for analyst_helpers.py: does profile() / compare_keys() report every
problem planted in the fake data, with the right numbers?

Runs directly on this PC (no Docker needed), from executor/:
    uv run python test_helpers.py
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from analyst_helpers import compare_keys, merge_variants, profile

DATA = Path(__file__).resolve().parent.parent / "data"


def captured(func, *args) -> str:
    """Run a function that prints, and return what it printed as a string.
    redirect_stdout temporarily sends print() into a StringIO buffer."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args)
    return buffer.getvalue()


def returned_and_printed(func, *args):
    """Like captured(), but for functions that print AND return something
    (merge_variants): gives back both, as a pair."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args)
    return result, buffer.getvalue()


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


emp = pd.read_csv(DATA / "employees.csv")
exp = pd.read_csv(DATA / "expenses.csv")
p_emp = captured(profile, emp, "employees")
p_exp = captured(profile, exp, "expenses")
keys = captured(compare_keys, emp, exp, "employee_id", "employees", "expenses")

# --- The eight planted problems (the answer key) ---
check("6 missing salaries", "salary       float64      6 missing" in p_emp)
check("$1.25M outlier named", "1,250,000 (employee_id=1018)" in p_emp)
check("'engineering' variant", "'Engineering' 123 / 'engineering' 3" in p_emp)
check("12 missing categories", "NaN (missing) 12" in p_exp)
check("'meals' variant", "'Meals' 1,702 / 'meals' 15" in p_exp)
check("9 negative amounts", "negative 9" in p_exp)
check("3 duplicate rows", "Exact duplicate rows: 3" in p_exp)
check("employee 9999: 1 ID, 5 rows", "1 ID in 5 rows (IDs: 9999)" in keys)

# --- Things that must NOT be reported ---
check("no false duplicates in employees", "Exact duplicate rows: 0" in p_emp)
check("no spelling variants in remote", "variants in remote" not in p_emp)
check("no min/max for ID columns", "employee_id: min" not in p_emp)

# --- The "Issues to report" checklists: every planted problem, with its count ---
def checklist(report: str) -> str:
    return report[report.index("Issues to report"):]

issues = checklist(p_emp) + checklist(p_exp) + checklist(keys)
for name, text in [("missing salaries", "salary: 6 missing values"),
                   ("salary outlier", "1,250,000 (employee_id=1018)"),
                   ("engineering", "'engineering' 3"),
                   ("missing categories", "category: 12 missing values"),
                   ("meals", "'meals' 15"),
                   ("negatives", "amount: 9 negative values"),
                   ("duplicates", "3 exact duplicate rows"),
                   ("orphan IDs", "1 ID in 5 rows (IDs: 9999)")]:
    check(f"checklist has {name}", text in issues)
check("employees checklist numbered", "Issues to report (3):" in p_emp)
check("expenses checklist numbered", "Issues to report (5):" in p_exp)
clean = pd.DataFrame({"x_id": [1, 2], "v": [1.0, 2.0]})
check("clean table says none", "Issues to report: none" in captured(profile, clean, "clean"))

# --- Size: all three together must fit in what the model sees per step (5,000) ---
total = len(p_emp) + len(p_exp) + len(keys)
check("output fits the model's limit", total < 4500, f"{total} chars")

# --- merge_variants(): fixes only the misspelled rows ---
dept = pd.Series(["Engineering"] * 5 + ["engineering"] * 2 + ["HR"] * 3 + ["Office Supplies", None],
                 name="department")
fixed, msg = returned_and_printed(merge_variants, dept)
check("variants merged to majority", (fixed == "Engineering").sum() == 7)
check("HR untouched", (fixed == "HR").sum() == 3)
check("other values untouched", (fixed == "Office Supplies").sum() == 1)
check("missing stays missing", fixed.isna().sum() == 1)
check("reports what changed", "'engineering' -> 'Engineering': 2 rows" in msg, msg.strip().splitlines()[0])

same, msg = returned_and_printed(merge_variants, pd.Series(["a", "b"], name="remote"))
check("no variants: nothing changed", "nothing changed" in msg and list(same) == ["a", "b"])

spaces, _ = returned_and_printed(merge_variants, pd.Series(["Meals"] * 3 + ["meals "], name="category"))
check("trailing space variant fixed", (spaces == "Meals").sum() == 4)

# --- Edge cases the fake data doesn't cover ---
odd = pd.DataFrame({
    "item_id": [1, 2, 3, 4],
    "label": ["Meals", "meals ", " Meals", None],      # spaces + capitals
    "empty": [np.nan] * 4,                             # all missing
    "flag": [True, False, True, True],                 # booleans
    "kind": pd.Series(["a", "b", "a", "b"], dtype=object),  # older object dtype
})
out = captured(profile, odd, "odd")
check("whitespace variants found", "' Meals' 1" in out and "'meals ' 1" in out)
check("all-missing column handled", "empty: all values missing" in out)
check("booleans don't crash", "flag" in out)
check("object text columns work", "kind: a 2, b 2" in out)

no_ids = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
check("no ID column: uses row numbers", "(row 5)" in captured(profile, no_ids, "t"))

left = pd.DataFrame({"k": [1, 2, None]})
right = pd.DataFrame({"k": [2, 3, 3]})
out = captured(compare_keys, left, right, "k", "L", "R")
check("missing keys counted, not matched", "L: 2 distinct IDs, one row per ID, 1 missing" in out)
check("missing keys in checklist", "L: 1 row with no k" in out)
check("rows vs IDs counted separately", "In R but not L: 1 ID in 2 rows (IDs: 3)" in out)

# Show the real reports, so you can see what the model will see.
print("\n" + p_emp + "\n" + p_exp + "\n" + keys)