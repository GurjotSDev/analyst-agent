r"""
Checks for analyst_helpers.py: does profile() / compare_keys() report every
problem planted in the fake data, with the right numbers? (The fixes are
tested in test_cleaning.py.)

Run inside the sandbox image, from the repo root:
    docker compose run --rm -v "${PWD}\executor\test_helpers.py:/app/test_helpers.py:ro" executor python test_helpers.py
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from analyst_helpers import check as check_tables  # test_helpers has its own check()
from analyst_helpers import compare_keys, profile

DATA = Path(__file__).resolve().parent.parent / "data"


def captured(func, *args) -> str:
    """Run a function that prints, and return what it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args)
    return buffer.getvalue()


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
                   ("salary outlier", "1,250,000; at employee_id=1018"),
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
check("object text columns work", "kind: 'a' 2, 'b' 2" in out)

no_ids = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
check("no ID column: uses row numbers", "(row 5)" in captured(profile, no_ids, "t"))

left = pd.DataFrame({"k": [1, 2, None]})
right = pd.DataFrame({"k": [2, 3, 3]})
out = captured(compare_keys, left, right, "k", "L", "R")
check("missing keys counted, not matched", "L: 2 distinct IDs, one row per ID, 1 missing" in out)
check("missing keys in checklist", "L: 1 row with no k" in out)
check("rows vs IDs counted separately", "In R but not L: 1 ID in 2 rows (IDs: 3)" in out)

# =============================================================================
# Each check on a small table with one of each problem planted
# =============================================================================
orders = pd.DataFrame({
    "order_id": range(1, 11),
    # stray spaces x3, placeholders x2 ('-', 'unknown'), a real missing value
    "region ": ["North", "North ", "North", "South", " South", "East  Coast", "-", "unknown",
                None, "North"],
    # numbers stored as text: 9 numbers, 1 not
    "amount": ["$1,200", "45", "3.50", "1,000", "abc", "12", "7", "8", "9", "10"],
    # mixed formats: ISO x6 (one impossible: Feb 30), month-first slash x2,
    # written month x1, not a date x1
    "order_date": ["2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-02-30",
                   "2025-01-08", "01/23/2025", "02/14/2025", "Mar 3, 2025", "soon"],
    # all ISO, but one in the future and one before 1900
    "ship_date": ["2025-01-05", "2099-05-01", "1850-01-01"] + ["2025-01-06"] * 7,
})
out = captured(profile, orders, "orders")
issues = checklist(out)

check("A: column name with stray space", "column names with stray spaces: 'region '" in issues)
check("A: stray spaces counted", "region : 3 values with stray spaces" in issues)
check("A: spaces shown in examples", "'North '" in issues and "'East  Coast'" in issues)
check("A: placeholders found", "region : 2 placeholder values" in issues
      and "'-' 1" in issues and "'unknown' 1" in issues)
check("A: placeholders not called spelling variants", "spelling variants" not in issues)
check("A: numbers stored as text", "amount: numbers stored as text: 9 of 10 values" in issues)
check("A: formatted examples first", "e.g. '$1,200', '1,000');" in issues)
check("A: non-numbers named", "not numbers: 1 value (e.g. 'abc')" in issues)
check("A: mixed date formats", "order_date: mixed date formats: YYYY-MM-DD 6, MM/DD/YYYY 2, "
      "written month 1" in issues)
check("A: non-date value", "order_date: not dates: 1 value (e.g. 'soon')" in issues)
check("A: impossible date", "impossible dates: 1 value (e.g. '2025-02-30')" in issues)
check("A: future date", "ship_date: 1 date in the future, latest 2099-05-01 (order_id=2)" in issues)
check("A: date before 1900", "ship_date: 1 date before 1900, earliest 1850-01-01 (order_id=3)" in issues)
check("A: one clean format not flagged", "ship_date: mixed" not in out and "ship_date: dates, all YYYY-MM-DD" in out)
check("A: written month parsed", "range 2025-01-03 to 2025-03-03" in out)

# Day/month order: decided when it can be, flagged when it can't.
def date_issues(values):
    return checklist(captured(profile, pd.DataFrame({"d_date": values}), "t"))
check("A: day-first detected", "mixed" not in date_issues(["23/01/2025", "05/02/2025"])
      and "order unknown" not in date_issues(["23/01/2025", "05/02/2025"]))
check("A: order unknown flagged", "day/month order unknown in 2 dates like '03/04/2025'"
      in date_issues(["03/04/2025", "05/06/2025"]))
check("A: conflicting orders flagged", "only fit day first ('23/01/2025')" in
      date_issues(["23/01/2025", "01/23/2025"]))
check("A: 2-digit years read", "range 2025-01-23 to 2025-01-24" in
      captured(profile, pd.DataFrame({"d": ["01/23/25", "01/24/25"]}), "t"))
check("A: times allowed in ISO", "Issues to report: none" in
      captured(profile, pd.DataFrame({"at": ["2025-01-03 14:00", "2025-01-04T09:30:00"]}), "t"))
real_dates = pd.DataFrame({"when": pd.to_datetime(["2025-01-01", "2099-01-01"])})
check("A: real date columns checked too", "1 date in the future" in captured(profile, real_dates, "t"))

# Future and pre-1900 dates come with a ready-made condition: the cut-off the
# check itself used, so it selects exactly the dates it counted.
today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
check("A: future date: condition is today", f"with the condition ship_date > '{today}'" in issues, issues)
check("A: future date: text column converted first", "(suggest: if not, after to_dates, select it" in issues)
check("A: real dates: no conversion step", f"(suggest: if not, select it with the condition when > '{today}'"
      in checklist(captured(profile, real_dates, "t")))
check("A: before 1900: condition given", "with the condition ship_date < '1900-01-01'" in issues)
# The condition really selects only the future date, once the dates are real.
shipped = pd.to_datetime(orders["ship_date"])
check("A: future condition selects only the future date", (shipped > today).sum() == 1)

# Things these checks must NOT flag
fine = pd.DataFrame({
    "zip": ["02134", "10001", "94105"],                   # codes: leading zeros
    "name": ["Ana Silva", "Bo Chen", "Cy Diaz"],          # ordinary text
    "notes": ["call 2025-01-03", "ok", "follow up"],      # mentions a date, isn't one
    "status": ["None", "Done", "Done"],                   # 'None' here is read by pandas as NaN anyway
})
fine_out = captured(profile, fine.iloc[:, :3], "fine")
check("A: ZIP codes not numbers-as-text", "numbers stored as text" not in fine_out)
check("A: notes column not a date column", "notes: dates" not in fine_out)
check("A: clean text says none", "Issues to report: none" in fine_out, fine_out.strip().splitlines()[-1])

print("\n" + out)

# check(): every profile plus every shared-ID comparison, in one call
emp.attrs["table"], exp.attrs["table"] = "employees", "expenses"   # as load() sets them
both = captured(check_tables, emp, exp)
check("check(): profiles + compare_keys", "=== employees: 412 rows" in both and
      "=== expenses: 5,833 rows" in both and "1 ID in 5 rows (IDs: 9999)" in both)
check("check(): compares only shared ID columns", both.count("=== employees vs expenses on") == 1)
unrelated = captured(check_tables, pd.DataFrame({"a_id": [1]}), pd.DataFrame({"b_id": [1]}))
check("check(): unrelated tables not compared", "no shared ID column" in unrelated)

# Which direction of a key mismatch is an issue
customers = pd.DataFrame({"customer_id": [1, 2, 3]})
orders = pd.DataFrame({"customer_id": [1, 1, 2, 9, 9]})
out = captured(compare_keys, customers, orders, "customer_id", "customers", "orders")
check("customer with no orders: normal, not an issue",
      "(normal: not every customers row has orders rows)" in out
      and "customers: 1 ID in 1 row" not in checklist(out), checklist(out).splitlines()[0])
check("order with unknown customer: an issue", "orders: 1 ID in 2 rows (IDs: 9) not in customers"
      in checklist(out))
one_to_one = captured(compare_keys, pd.DataFrame({"k": [1, 2]}), pd.DataFrame({"k": [2, 3]}), "k", "A", "B")
check("one-to-one tables: both directions are issues", "Issues to report (2)" in one_to_one)

# Numbers with a unit
units = captured(profile, pd.DataFrame({"weight_kg": ["12.5 kg", "3 kg", "40.0 kg", "7.1 kg"]}), "t")
check("numbers with a unit found", "weight_kg: numbers stored with a unit: 4 of 4 values like "
      "'12.5 kg', '3 kg' (unit 'kg')" in units, checklist(units).splitlines()[-1][:70])
check("unit fix points to custom_fix", "propose a custom_fix that removes 'kg', then to_numbers" in units)
mixed_units = captured(profile, pd.DataFrame({"w": ["12 kg", "3 lb", "40 kg", "7 lb"]}), "t")
check("mixed units not treated as one unit", "numbers stored with a unit" not in mixed_units)
check("plain words not a unit", "numbers stored with a unit" not in
      captured(profile, pd.DataFrame({"c": ["red", "blue", "red", "green"]}), "t"))

# Singular and plural wording
one = captured(profile, pd.DataFrame({"x": ["a", "b", "-"]}), "t")
check("one placeholder: 'means'", "1 placeholder value that probably means missing" in one)

# Incomplete example lists are labelled as incomplete
check("example list says it is partial", "5 values; most extreme 3 of 5:" in p_emp)
check("fix hint: select with a condition", "with the condition (salary > 306600).fillna(False)" in checklist(p_emp))
# salary has missing values: the plain condition gives <NA> for them (nullable
# columns), which the fix helpers refuse. The ready-made one never does.
check("fix hint: condition is safe with missing values",
      not (emp["salary"].astype("Float64") > 306600).fillna(False).isna().any())

# Outlier verdicts: which values far from typical are likely errors
check("lone huge salary: likely an error", "salary: likely an error: 1 value (1,250,000; at "
      "employee_id=1018) far beyond every other value" in checklist(p_emp))
check("other high salaries: plausible", "the other 4 values far from typical look plausible "
      "(suggest: fix only the likely error" in checklist(p_emp))
check("spread-out large expenses: plausible", "amount: plausible: 53 values far from typical" in checklist(p_exp)
      and "none stands apart from the rest" in checklist(p_exp) and "likely" not in checklist(p_exp))
sentinel = captured(profile, pd.DataFrame({"t": [4.0, 5, 6, 5, 4, 6, 7, 5, -999, -999]}), "t")
check("repeated sentinel: likely errors, low side", "t: likely errors: 2 values (-999; most extreme at row 8) far beyond"
      in sentinel and "with the condition t < 4" in sentinel, checklist(sentinel))
two_sizes = captured(profile, pd.DataFrame({"v": list(range(100, 160)) + [10_000, 1_000_000]}), "t")
check("two bad values of different sizes: both caught", "2 values (1,000,000, 10,000;" in two_sizes
      and "v > 159" in two_sizes, checklist(two_sizes))
both = captured(profile, pd.DataFrame({"v": list(range(100, 160)) + [-50_000, 90_000]}), "t")
check("errors on both sides: one condition with or", "v > 159 or v < 100" in both, checklist(both))
check("verdict comes first, before any detail", all(line.split(": ")[1] in ("likely an error", "likely errors",
      "plausible") for line in (checklist(p_emp) + checklist(p_exp)).splitlines()
      if "far " in line and line.strip()[:1].isdigit()))
check("missing values: suggest leaving them", "salary: 6 missing values (suggest: leave missing" in p_emp)
two = captured(profile, pd.DataFrame({"v": [1, 2, 2, 3, 3, 3, 4, 4, 5, 500, 900]}), "t")
check("short example list says it is complete", "2 values; all of them:" in two)

# Repeated keys: the first ID column should name one row each
tickets = pd.DataFrame({"ticket_id": ["T1", "T2", "T2", "T3", "T3", "T4"],
                        "agent_id": [7, 7, 8, 9, 9, 7],
                        "hours": [1.0, 2.0, 5.0, 3.0, 3.0, 4.0]})
rk = captured(profile, tickets, "tickets")
check("repeated key reported", "ticket_id: repeated key: 1 ID on more than one different row (T2 x2)"
      in checklist(rk), checklist(rk))
check("repeated key: exact duplicates not counted twice", "T3" not in checklist(rk).split("repeated key")[1]
      and "1 exact duplicate row" in checklist(rk))
check("repeated key: ready-made condition", "drop_rows with the condition ticket_id.duplicated()" in rk)
check("repeated key: only the first ID column", "agent_id: repeated" not in rk)
check("repeated key: none in the fake data (expenses' employee_id may repeat)",
      "repeated key" not in p_emp + p_exp)
check("repeated key: numeric IDs shown plainly", "(12 x2)" in captured(
      profile, pd.DataFrame({"id": [11.0, 12.0, 12.0], "v": [1, 2, 3]}), "t"))

# Numbers stored as text are checked as the numbers they'll become, so an
# outlier shows in the FIRST check, not only after to_numbers.
rates = pd.DataFrame({"contractor_id": [f"C{i}" for i in range(12)],
                      "day_rate": [f"${r}" for r in (640, 890, 870, 810, 690, 580, 730, 910, 920, 590, 660, 880)]})
rates.loc[4, "day_rate"] = "$75,000"
tr = captured(profile, rates, "contractors")
check("text numbers: outlier found before converting", "day_rate: likely an error: 1 value (75,000; at "
      "contractor_id=C4)" in checklist(tr), checklist(tr))
check("text numbers: told to convert first", "(suggest: after to_numbers, fix only the likely error" in tr)
check("text numbers: conversion still proposed", "day_rate: numbers stored as text" in checklist(tr))
check("text numbers: range shown as numbers", "day_rate (as numbers): min 580" in tr)
check("text numbers: none flagged in plain data", "far from typical" not in captured(
      profile, pd.DataFrame({"id": [1, 2, 3, 4], "v": ["$10", "$12", "$11", "$13"]}), "t"))

# Spelling variants say how many values a merge changes, and into what
teams = captured(profile, pd.DataFrame({"team": ["Cloud Ops"] * 6 + ["cloud ops", "CLOUD OPS", "Security"]}), "t")
check("variants: merge count given", "team: spelling variants 'Cloud Ops' 6 / 'cloud ops' 1 / 'CLOUD OPS' 1; "
      "merging changes 2 values to 'Cloud Ops'" in checklist(teams), checklist(teams))
text_hours = captured(profile, pd.DataFrame({"id": range(12), "hours": [str(h) for h in
                      (40, 36, 42, 38, 40, 32, 40, 36, 400, 40, 42, 38)]}), "t")
check("text numbers: condition safe once converted", "(hours > 42).fillna(False)" in text_hours, checklist(text_hours))

# Abbreviations: a rare value that starts exactly one common value
def abbrev(values):
    return checklist(captured(profile, pd.DataFrame({"c": values}), "t"))
check("abbreviation found", "c: possible abbreviation: 'Med' 1 may mean 'Medium' 11; ask the user"
      in abbrev(["Medium"] * 11 + ["High"] * 9 + ["Med"]), abbrev(["Medium"] * 11 + ["High"] * 9 + ["Med"]))
check("abbreviation: trailing dot ok", "'Med.' 1 may mean 'Medium'" in abbrev(["Medium"] * 5 + ["Med."]))
check("abbreviation: custom_fix suggested", "(suggest: if they agree, a custom_fix that replaces 'Med' with "
      "'Medium')" in abbrev(["Medium"] * 5 + ["Med"]))
check("abbreviation: not when a word follows", "abbreviation" not in abbrev(["North East"] * 6 + ["North"]))
check("abbreviation: not when ambiguous", "abbreviation" not in abbrev(["Data Lake"] * 5 + ["Data Eng"] * 5 + ["Da"]))
check("abbreviation: not when about as common", "abbreviation" not in abbrev(["Medium"] * 2 + ["Med"]))
check("abbreviation: not for capital variants", "abbreviation" not in abbrev(["Medium"] * 6 + ["medium"]))
check("abbreviation: not a single letter", "abbreviation" not in abbrev(["Medium"] * 6 + ["M"]))
check("abbreviation: not in free text", "abbreviation" not in abbrev([f"Name {i}" for i in range(20)] + ["Na"]))

# Orphan rows come with a ready-made condition to drop them
orphan_out = captured(compare_keys, pd.DataFrame({"cid": ["A", "B"], "n": [1, 2]}),
                      pd.DataFrame({"cid": ["A", "A", "Z"], "h": [1, 2, 3]}), "cid", "people", "hours")
check("orphans: drop condition suggested", "(suggest: ask whether to drop these rows; to drop them, drop_rows "
      "with the condition ~hours['cid'].isin(people['cid']))" in orphan_out, orphan_out)

# check() with no tables = every loaded table; a table left out is named
import cleaning
for records in (cleaning._SOURCES, cleaning._ORIGINALS, cleaning._HISTORY):
    records.clear()
with contextlib.redirect_stdout(io.StringIO()):          # hide load()'s own printout
    e2 = cleaning.load(DATA / "employees.csv")
    x2 = cleaning.load(DATA / "expenses.csv")
everything = captured(check_tables)
check("check() with no tables checks all loaded", "=== employees: 412" in everything
      and "=== expenses: 5,833" in everything and "1 ID in 5 rows (IDs: 9999)" in everything)
partial = captured(check_tables, e2)
check("profile() uses the name from load()", captured(profile, e2).startswith("=== employees: 412"))
check("profile() still accepts a name", captured(profile, e2, "staff").startswith("=== staff:"))
check("table left out is named", "WARNING: loaded but not checked: expenses" in partial)
check("no warning when all are checked", "WARNING" not in captured(check_tables, e2, x2))

# Show the real reports, so you can see what the model will see.
print("\n" + p_emp + "\n" + p_exp + "\n" + keys)