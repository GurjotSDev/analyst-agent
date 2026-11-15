"""
The evaluation scenarios: what the bot is asked, and how each answer is scored.

Each scenario is a short conversation about one set of files:
  1. "Check these files": the reply must state each planted problem with its
     count (Fact) or say something specific (Says), e.g. ask about day order.
  2. "Apply these fixes": then a probe looks at the cleaned tables INSIDE the
     sandbox and checks each Expect, a pandas expression that must be True.

Expectations are written against the original files (orig[...]) wherever
possible, e.g. "every converted date equals the original text parsed day
first", so they don't depend on random values in the fake data.

Datasets: data/employees.csv + data/expenses.csv (the sample data), and
data/eval/* from make_datasets.py. All fake; the counts below are what was
planted.
"""

import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath as Path   # the sandbox's paths are Linux paths

from scoring import Fact, NeverProposes, Says


@dataclass
class Step:
    name: str
    message: str
    facts: list = field(default_factory=list)   # Fact / Says, checked in the final reply
    probe: bool = False                          # check the cleaned tables after this message


@dataclass
class Expect:
    name: str     # shown in the results
    expr: str     # a Python expression that must be True. Available names:
                  #   t     {table: last logged version}   orig {table: as loaded}
                  #   log   the change log (list of dicts)  pd, np


@dataclass
class Scenario:
    name: str
    tests: str                 # what this scenario is for, in one line
    files: list[str]           # relative to /data
    steps: list[Step]
    expects: list[Expect]
    saved: list[str] = field(default_factory=list)   # tables that must be saved


def _check_message(n: int) -> str:
    return (f"Here {'is a file' if n == 1 else f'are {n} files'}. Check {'it' if n == 1 else 'them'} "
            "for problems and tell me what you would fix.")


SCENARIOS = [
    # -------------------------------------------------------------------------
    Scenario(
        name="expenses",
        tests="the original fake data: variants, outliers, refunds, an unknown employee",
        files=["employees.csv", "expenses.csv"],
        steps=[
            Step("explore", _check_message(2), facts=[
                Fact("6 missing salaries", ["salar"], 6),
                Fact("3 'engineering' variants", ["engineering"], 3),
                Fact("$1.25M salary outlier", ["salar", "1018", "outlier", "extreme"], 1_250_000),
                Fact("12 missing categories", ["categor"], 12),
                Fact("15 'meals' variants", ["meal"], 15),
                Fact("9 negative amounts", ["negative", "refund"], 9),
                Fact("3 duplicate rows", ["duplicate"], 3),
                Fact("employee 9999: 5 rows", ["9999"], 5),
                # The proposal itself: only the lone salary is an error.
                Says("1,250,000 called a likely error", r"likely (an )?error"),
                NeverProposes("never proposes dropping the 53 expenses", r"(drop|remov|delet|exclud)\w*\b.*\b53\b"),
                NeverProposes("never proposes dropping 5 salaries", r"(drop|remov|delet|exclud)\w*\b.*\b(5|five)\b.*salar"),
                NeverProposes("never proposes a guessed fill", r"fill\w*\b.*\b(median|mean|average|'other'|\"other\")"),
            ]),
            Step("clean", probe=True, message=(
                "Apply these fixes: merge the spelling variants, drop the 3 duplicates, exclude "
                "negative amounts as refunds, exclude the employee 9999 rows, set employee 1018's "
                "salary to missing as a data error (keep the employee), and label missing categories "
                "\"Uncategorized\". Keep the other salaries and the large expenses; they're plausible.")),
        ],
        expects=[
            Expect("412 employees kept", "len(t['employees']) == 412"),
            Expect("departments merged, HR intact", "sorted(t['employees']['department'].unique()) == "
                   "['Engineering', 'Executive', 'Finance', 'HR', 'Marketing', 'Operations', 'Sales']"),
            Expect("7 missing salaries", "t['employees']['salary'].isna().sum() == 7"),
            Expect("1018 kept, salary removed", "t['employees'].loc[t['employees']['employee_id'] == 1018, "
                   "'salary'].isna().tolist() == [True]"),
            Expect("5,816 expense rows", "len(t['expenses']) == 5816"),
            Expect("refunds removed (not NaN)", "(t['expenses']['amount'] < 0).sum() == 0 "
                   "and t['expenses']['amount'].isna().sum() == 0"),
            Expect("no duplicates", "t['expenses'].duplicated().sum() == 0"),
            Expect("9999 removed", "not (t['expenses']['employee_id'] == 9999).any()"),
            Expect("12 Uncategorized, none missing", "(t['expenses']['category'] == 'Uncategorized').sum() "
                   "== 12 and t['expenses']['category'].isna().sum() == 0"),
            Expect("categories merged", "sorted(t['expenses']['category'].unique()) == ['Equipment', "
                   "'Meals', 'Office Supplies', 'Software', 'Training', 'Travel', 'Uncategorized']"),
            # 7 fixes were asked for; the two drops may be combined into one call.
            Expect("fixes logged by helpers", "len(log) >= 6"),
        ],
        saved=["employees", "expenses"],
    ),
    # -------------------------------------------------------------------------
    Scenario(
        name="retail",
        tests="three linked tables, money as text, two date formats, retried-load duplicates, orphan rows",
        files=["eval/retail/customers.csv", "eval/retail/products.csv", "eval/retail/orders.csv"],
        steps=[
            Step("explore", _check_message(3), facts=[
                Fact("6 duplicate orders", ["duplicate"], 6),
                Fact("customer 999: 4 orders", ["999"], 4),
                Fact("product P099: 2 orders", ["p099"], 2),
                # The formats must be named; the count of 50 is optional. Replies
                # count the fix by rows changed (all 1,206), which is also right.
                Says("mixed date formats named", r"mm/dd/yyyy"),
                Fact("6 region placeholders", ["placeholder", "region"], 6),
                Fact("3 'north' variants", ["north"], 3),
                Fact("5 customer names with spaces", ["name"], 5),
                Fact("4 product names with spaces", ["product"], 4),
                Fact("39 prices stored as text", ["price"], 39),
                Says("'TBD' price found", r"TBD"),
            ]),
            Step("clean", probe=True, message=(
                "Apply these fixes: remove the 6 duplicate orders, remove the orders for customer 999 and "
                "for product P099 (they don't exist), convert order_date to real dates (the slash dates "
                "are month first), turn the 'TBD' price into a missing value and convert prices to numbers, "
                "turn the '-' and 'unknown' regions into missing values, merge 'north' into 'North', and "
                "remove the stray spaces in customer and product names.")),
        ],
        expects=[
            Expect("200 customers kept", "len(t['customers']) == 200"),
            Expect("customer names trimmed", "(t['customers']['name'] != t['customers']['name'].str.strip())"
                   ".sum() == 0"),
            Expect("6 regions now missing", "t['customers']['region'].isna().sum() == 6"),
            Expect("regions merged", "sorted(t['customers']['region'].dropna().unique()) == "
                   "['East', 'North', 'South', 'West']"),
            Expect("40 products kept", "len(t['products']) == 40"),
            Expect("product names trimmed", "t['products']['product_name'].str.contains(r'^\\s|\\s$|\\s\\s', "
                   "regex=True).sum() == 0"),
            Expect("prices are numbers, TBD missing", "pd.api.types.is_numeric_dtype(t['products']['price']) "
                   "and t['products']['price'].isna().sum() == 1"),
            Expect("prices correct", "np.allclose(t['products']['price'].dropna(), pd.to_numeric("
                   "orig['products']['price'].str.replace(r'[$,]', '', regex=True), errors='coerce').dropna())"),
            Expect("1,194 orders kept", "len(t['orders']) == 1194"),
            Expect("no duplicate orders", "t['orders'].duplicated().sum() == 0"),
            Expect("orphans removed", "not (t['orders']['customer_id'] == 999).any() "
                   "and not (t['orders']['product_id'] == 'P099').any()"),
            Expect("order dates converted correctly", "pd.api.types.is_datetime64_any_dtype(t['orders']"
                   "['order_date']) and (t['orders']['order_date'] == pd.to_datetime(orig['orders']"
                   "['order_date'], format='ISO8601', errors='coerce').fillna(pd.to_datetime(orig['orders']"
                   "['order_date'], format='%m/%d/%Y', errors='coerce')).loc[t['orders'].index]).all()"),
            Expect("fixes logged by helpers", "len(log) >= 8"),
        ],
        saved=["customers", "products", "orders"],
    ),
    # -------------------------------------------------------------------------
    Scenario(
        name="banking",
        tests="PARTIAL approval: only 3 fixes approved; normal negatives, leading zeros, future dates must stay",
        files=["eval/banking/accounts.csv", "eval/banking/transactions.csv"],
        steps=[
            Step("explore", _check_message(2), facts=[
                Fact("3 duplicate transactions", ["duplicate"], 3),
                Fact("5 sentinel amounts", ["999,999.99", "999999.99", "far from typical", "far beyond"], 5),
                Fact("4 'savings' variants", ["savings"], 4),
                Fact("2 future transactions", ["future", "2027"], 2),
                Says("account opened in 2031", r"2031"),
            ]),
            Step("clean", probe=True, message=(
                "Only apply these three fixes: remove the 3 duplicate transactions, set the 999999.99 "
                "amounts to missing (an old system's placeholder; keep the rows), and merge 'savings' "
                "into 'Savings'. Leave everything else exactly as it is: the negative amounts are normal "
                "withdrawals, and I'll check the future dates with the bank.")),
        ],
        expects=[
            Expect("3,000 transactions kept", "len(t['transactions']) == 3000 "
                   "and t['transactions'].duplicated().sum() == 0"),
            Expect("sentinels missing, rows kept", "t['transactions']['amount'].isna().sum() == 5 "
                   "and (t['transactions']['amount'] > 999999).sum() == 0"),
            Expect("withdrawals untouched (not approved)", "(t['transactions']['amount'] < 0).sum() == "
                   "(orig['transactions'].drop_duplicates()['amount'] < 0).sum()"),
            Expect("timestamps untouched (not approved)", "t['transactions']['timestamp'].equals("
                   "orig['transactions'].loc[t['transactions'].index, 'timestamp'])"),
            Expect("opened dates untouched (not approved)", "t['accounts']['opened_date'].equals("
                   "orig['accounts']['opened_date'])"),
            Expect("account types merged", "sorted(t['accounts']['account_type'].unique()) == "
                   "['Checking', 'Savings']"),
            Expect("account IDs keep leading zeros", "t['accounts']['account_id'].str.len().eq(8).all() "
                   "and t['accounts']['account_id'].equals(orig['accounts']['account_id'])"),
            # Exactly the three approved fixes: anything more is an unapproved change.
            Expect("exactly 3 changes logged", "len(log) == 3"),
        ],
        saved=["accounts", "transactions"],
    ),
    # -------------------------------------------------------------------------
    Scenario(
        name="shipments",
        tests="dates that read both ways (must ask), weights with a unit (custom fix), sensor glitches",
        files=["eval/shipments/shipments.csv"],
        steps=[
            Step("explore", _check_message(1), facts=[
                Says("asks about day/month order", r"day[ -]?first|month[ -]?first|day/month|"
                                                   r"month/day|dd/mm|mm/dd|which order"),
                Fact("3 origins with spaces", ["origin", "hamburg"], 3),
                Fact("4 sensor glitches", ["-999", "temperature"], 4),
                Says("impossible date found", r"2025-02-30"),
                Fact("500 weights with a unit", ["unit", "kg"], 500),
            ]),
            Step("clean", probe=True, message=(
                "The ship dates are day first. Please: convert ship_date to real dates (day first), turn "
                "weight_kg into plain numbers by removing ' kg', set the -999 temperatures to missing (sensor "
                "errors, keep the rows), and remove the stray spaces in origin. Leave delivered_date as it "
                "is; I'll ask the warehouse about the impossible date.")),
        ],
        expects=[
            Expect("500 shipments kept", "len(t['shipments']) == 500"),
            Expect("ship dates read day first", "pd.api.types.is_datetime64_any_dtype(t['shipments']"
                   "['ship_date']) and (t['shipments']['ship_date'] == pd.to_datetime(orig['shipments']"
                   "['ship_date'], format='%d/%m/%Y')).all()"),
            Expect("weights are numbers in kg", "pd.api.types.is_numeric_dtype(t['shipments']['weight_kg']) "
                   "and np.allclose(t['shipments']['weight_kg'], orig['shipments']['weight_kg']"
                   ".str.replace(' kg', '').astype(float))"),
            Expect("sensor glitches missing", "t['shipments']['temperature_c'].isna().sum() == 4 "
                   "and t['shipments']['temperature_c'].min() >= 2"),
            Expect("origins trimmed", "sorted(t['shipments']['origin'].unique()) == "
                   "['Hamburg', 'Lyon', 'Milan', 'Rotterdam']"),
            Expect("delivered_date untouched (not approved)", "t['shipments']['delivered_date'].equals("
                   "orig['shipments']['delivered_date'])"),
            Expect("custom fix logged", "any(e['action'] == 'custom_fix' for e in log)"),
        ],
        saved=["shipments"],
    ),
    # -------------------------------------------------------------------------
    Scenario(
        name="clean",
        tests="nothing is wrong: no invented problems, nothing changed",
        files=["eval/clean/catalog.csv"],
        steps=[
            Step("explore", _check_message(1), facts=[
                Says("says there is nothing to fix", r"no (data[- ]quality )?(issues|problems)|nothing to "
                     r"(fix|clean|change)|issues to report: none|(looks|is|are) (already )?clean|"
                     r"no (fixes|changes) (are )?(needed|required)"),
            ]),
            Step("apply", probe=True, message="Apply all the fixes you proposed."),
        ],
        expects=[
            Expect("nothing changed", "len(log) == 0"),
            Expect("catalog unchanged", "t['catalog'].equals(orig['catalog'])"),
        ],
    ),
    # -------------------------------------------------------------------------
    Scenario(
        name="staffing",
        tests="outliers hidden in text numbers, a future date among same-year dates, repeated key, abbreviation",
        files=["eval/staffing/contractors.csv", "eval/staffing/timesheets.csv"],
        steps=[
            Step("explore", _check_message(2), facts=[
                Says("repeated contractor C012 found", r"C012"),
                Fact("75,000 day rate found before converting", ["day_rate", "day rate"], 75_000),
                Fact("400 hours found before converting", ["hours"], 400),
                Says("both outliers called likely errors", r"likely (an )?error[\s\S]*likely (an )?error"),
                Fact("C099: 2 orphan rows", ["c099"], 2),
                Says("abbreviation 'Sec' raised", r"\bSec\b"),
                Says("impossible date found", r"2025-02-30"),
                Says("future week found", r"2030-03-02"),
                # 15.03.2025 and 03/16/2026 can each be read only one way.
                NeverProposes("never asks a needless day/month question",
                              r"\?.*(day[ -]?first|month[ -]?first|day/month|month/day)"),
            ]),
            Step("clean", probe=True, message=(
                "Apply these fixes: drop the exact duplicate rows in both tables; for contractor C012 keep "
                "the first row and drop the later one; strip the spaces in names; merge the team variants, "
                "and 'Sec' means 'Security'; convert day_rate and hours to numbers, with the '-' hours as a "
                "missing value; set the 75,000 day rate and the 400 hours to missing, keeping the rows; "
                "convert start_date and week_start to dates, with the impossible date 2025-02-30 as missing; "
                "week starts can't be in the future, so set that week to missing too; turn the '?' country "
                "into a missing value; merge the project variants; and drop the timesheet rows for "
                "contractor C099, who doesn't exist. Leave the other missing values as they are, then save "
                "both tables.")),
        ],
        expects=[
            Expect("24 contractors, one row each", "len(t['contractors']) == 24 "
                   "and t['contractors']['contractor_id'].is_unique"),
            Expect("C012: the first row kept", "t['contractors'].loc[t['contractors']['contractor_id'] == 'C012', "
                   "'team'].tolist() == [orig['contractors'].loc[orig['contractors']['contractor_id'] == 'C012', "
                   "'team'].iloc[0]]"),
            Expect("teams merged, Sec expanded", "sorted(t['contractors']['team'].unique()) == "
                   "['AI Research', 'Cloud Ops', 'Data Engineering', 'Security']"),
            Expect("names trimmed", "(t['contractors']['name'] != t['contractors']['name'].str.strip()).sum() == 0"),
            Expect("day rates numbers, typo missing", "pd.api.types.is_numeric_dtype(t['contractors']['day_rate']) "
                   "and t['contractors']['day_rate'].isna().sum() == 1 and t['contractors']['day_rate'].max() < 1000"),
            Expect("15.03.2025 read as 15 March", "t['contractors'].loc[t['contractors']['contractor_id'] == "
                   "'C004', 'start_date'].tolist() == [pd.Timestamp('2025-03-15')]"),
            Expect("impossible start date missing", "t['contractors']['start_date'].isna().sum() == 1"),
            Expect("2 countries missing", "t['contractors']['country'].isna().sum() == 2"),
            Expect("72 timesheet rows, one per entry", "len(t['timesheets']) == 72 "
                   "and t['timesheets']['entry_id'].is_unique"),
            Expect("every contractor exists", "t['timesheets']['contractor_id'].isin("
                   "t['contractors']['contractor_id']).all()"),
            Expect("hours numbers, 3 missing", "pd.api.types.is_numeric_dtype(t['timesheets']['hours']) "
                   "and t['timesheets']['hours'].isna().sum() == 3 and t['timesheets']['hours'].max() <= 42"),
            # The trap: every other week is in 2026, so a guessed cut-off blanks them all.
            Expect("ONLY the future week missing", "t['timesheets']['week_start'].isna().sum() == 1 "
                   "and t['timesheets'].loc[t['timesheets']['week_start'].isna(), 'entry_id'].tolist() == [5026]"),
            Expect("03/16/2026 read as March 16", "t['timesheets'].loc[t['timesheets']['entry_id'] == 5031, "
                   "'week_start'].tolist() == [pd.Timestamp('2026-03-16')]"),
            Expect("projects merged", "t['timesheets']['project'].nunique() == 4"),
            Expect("fixes logged by helpers", "len(log) >= 14"),
        ],
        saved=["contractors", "timesheets"],
    ),
]

BY_NAME = {s.name: s for s in SCENARIOS}


# ---------------------------------------------------------------------------
# The probe: evaluates a scenario's expectations inside the sandbox kernel
# ---------------------------------------------------------------------------
# Tables come from the change log's records (cleaning._HISTORY): each table's
# last logged version, whatever the model named its variables. Everything is
# inside a function that deletes itself, so the probe leaves nothing behind.
_PROBE = '''
def _eval_probe(expects, saved_tables):
    import json
    import numpy as np
    import pandas as pd
    import cleaning
    t = {name: versions[-1] for name, versions in cleaning._HISTORY.items()}
    orig = dict(cleaning._ORIGINALS)
    log = cleaning._LOG
    results, errors = {}, {}
    for name, expr in expects:
        try:
            results[name] = bool(eval(expr, {"t": t, "orig": orig, "log": log, "pd": pd, "np": np}))
        except Exception as e:   # e.g. a table never loaded, or a column missing
            results[name] = False
            errors[name] = f"{type(e).__name__}: {e}"
    saved = {}
    for name in saved_tables:
        path = cleaning.OUTPUT_DIR / "clean" / f"{name}.csv"
        if path.exists():
            saved[name] = int(len(pd.read_csv(path)))
    print(json.dumps({"results": results, "errors": errors, "saved_rows": saved,
                      "table_rows": {n: int(len(df)) for n, df in t.items()},
                      "save_dir": str(cleaning.OUTPUT_DIR)}))
_eval_probe(EXPECTS, SAVED)
del _eval_probe
'''


def probe_code(scenario: Scenario) -> str:
    expects = [(e.name, e.expr) for e in scenario.expects]
    # repr() writes the lists as Python literals for the kernel to read.
    return _PROBE.replace("EXPECTS", repr(expects)).replace("SAVED", repr(scenario.saved))


def score_probe(stdout: str, scenario: Scenario) -> tuple[dict[str, bool], dict[str, str]]:
    """The probe's JSON as (checks, errors). Saving is checked here: each
    table that must be saved has a file with its logged row count, in this
    run's own folder under evals (never files from somewhere else)."""
    try:
        state = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        state = {}
    results = state.get("results", {})
    checks = {e.name: bool(results.get(e.name, False)) for e in scenario.expects}
    if scenario.saved:
        saved, rows = state.get("saved_rows", {}), state.get("table_rows", {})
        in_run_folder = "evals" in Path(state.get("save_dir", "")).parts
        checks["cleaned files saved"] = in_run_folder and all(
            name in saved and saved[name] == rows.get(name) for name in scenario.saved)
    return checks, state.get("errors", {})