r"""
Checks for cleaning.py: do the fix helpers change exactly what they should,
log it correctly, and refuse every change that isn't logged?

Run inside the sandbox image, from the repo root:
    docker compose run --rm -v "${PWD}\executor\test_cleaning.py:/app/test_cleaning.py:ro" executor python test_cleaning.py
"""

import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

# Saved files go to a throwaway folder, never your real output/clean/.
# OUTPUT_DIR must be set BEFORE cleaning is imported: it reads it on import.
OUT = Path(tempfile.mkdtemp())
os.environ["OUTPUT_DIR"] = str(OUT)

import numpy as np
import pandas as pd

import cleaning
from cleaning import (CleaningError, current, custom_fix, drop_duplicates, drop_rows,
                      fill_missing, load, merge_variants, placeholders_to_missing, save_clean,
                      set_missing, show_log, strip_spaces, to_dates, to_numbers, undo)

DATA = Path(__file__).resolve().parent.parent / "data"


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


def quiet(func, *args, **kwargs):
    """Run a helper without its printout; return (result, printed text)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


def refused(func, *args, **kwargs) -> str:
    """The CleaningError message if the call is refused, else ''."""
    try:
        quiet(func, *args, **kwargs)
    except CleaningError as e:
        return str(e)
    return ""


def fresh() -> None:
    """Forget every table and log entry, like a new kernel."""
    for records in (cleaning._SOURCES, cleaning._ORIGINALS, cleaning._HISTORY, cleaning._SAVED):
        records.clear()
    cleaning._LOG.clear()


def csv(name: str, df: pd.DataFrame) -> Path:
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


# =============================================================================
# 1. The fake data, cleaned exactly as the eval's message 2 asks
# =============================================================================
fresh()
emp, _ = quiet(load, DATA / "employees.csv")
exp, _ = quiet(load, DATA / "expenses.csv")
emp, _ = quiet(merge_variants, emp, "department")
exp, _ = quiet(merge_variants, exp, "category")
exp, _ = quiet(drop_duplicates, exp)
exp, _ = quiet(drop_rows, exp, exp["amount"] < 0, reason="refunds")
exp, _ = quiet(drop_rows, exp, exp["employee_id"] == 9999, reason="unknown employee")
emp, _ = quiet(set_missing, emp, emp["employee_id"] == 1018, "salary", reason="data error")
exp, _ = quiet(fill_missing, exp, "category", "Uncategorized")

check("table names from file names", emp.attrs["table"] == "employees" and exp.attrs["table"] == "expenses")
check("5,816 expense rows", len(exp) == 5816, f"{len(exp):,}")
check("exact total", round(exp["amount"].sum(), 2) == 3092192.37, f"{exp['amount'].sum():,.2f}")
check("412 employees, 7 missing salaries", len(emp) == 412 and emp["salary"].isna().sum() == 7)
check("departments merged, HR intact", sorted(emp["department"].unique()) ==
      ["Engineering", "Executive", "Finance", "HR", "Marketing", "Operations", "Sales"])
check("12 Uncategorized", (exp["category"] == "Uncategorized").sum() == 12)

log = cleaning._LOG
check("7 changes logged", len(log) == 7, str(len(log)))
by_reason = {e["reason"]: e for e in log}
refunds = by_reason["refunds"]
check("refunds: 9 rows removed", refunds["rows_removed"] == 9 and refunds["cells_changed"] == 0)
check("refunds: IDs of removed rows kept", len(refunds["removed"]) == 9
      and refunds["removed"][0].startswith("expense_id="), refunds["removed"][0])
e1018 = by_reason["data error"]
check("1018: one cell, before -> after", e1018["cells_changed"] == 1 and
      e1018["examples"][0] == {"row": "employee_id=1018", "column": "salary",
                               "before": 1250000.0, "after": None}, str(e1018["examples"]))
variants = by_reason["spelling variants"]
check("merge mapping logged", log[0]["details"]["mapping"] == {"engineering": "Engineering"}
      and log[0]["cells_changed"] == 3, str(log[0]["details"]))
check("dups: 3 removed", by_reason["exact duplicate rows"]["rows_removed"] == 3)

_, msg = quiet(save_clean, emp)
_, msg = quiet(save_clean, exp)
saved = pd.read_csv(OUT / "clean" / "expenses.csv")
report = json.loads((OUT / "clean" / "change_log.json").read_text())
check("clean CSV written", len(saved) == 5816 and (OUT / "clean" / "employees.csv").exists())
check("saved file matches the table", round(saved["amount"].sum(), 2) == 3092192.37)
check("change log JSON written", len(report["changes"]) == 7 and
      report["tables"]["expenses"]["original_rows"] == 5833, str(report["tables"]["expenses"]))
check("save message says what was saved", "5,816 rows x 5 columns (was 5,833 x 5), 5 logged changes" in msg, msg.strip())

# The report and data dictionary, written by save_clean()
pdf = (OUT / "clean" / "cleaning_report.pdf").read_bytes()
check("PDF report written", pdf.startswith(b"%PDF") and pdf.count(b"/Type /Page\n") >= 1, f"{len(pdf):,} bytes")
dictionary = (OUT / "clean" / "data_dictionary.md").read_text()
check("dictionary: a section per saved table", "## employees" in dictionary and "## expenses" in dictionary)
check("dictionary: changes per column", "| salary | decimal number | 7 |" in dictionary
      and "set_missing (1 cell changed in 'salary')" in dictionary)
check("dictionary: links between tables", "expenses.employee_id (one row per ID here)" in dictionary
      and "employees.employee_id (many rows per ID here)" in dictionary)
check("dictionary: dates stored as text are flagged", "| hire_date | text (holds dates) |" in dictionary)
check("dictionary: whole-row changes listed", "drop_rows: 9 rows removed (refunds)" in dictionary)

# =============================================================================
# 2. The chain of custody: every unlogged change is caught
# =============================================================================
fresh()
exp, _ = quiet(load, DATA / "expenses.csv")
exp, _ = quiet(drop_duplicates, exp)

sneaky = exp.copy()
sneaky.loc[sneaky["amount"] < 0, "amount"] = 0      # not through a helper
msg = refused(fill_missing, sneaky, "category", "Uncategorized")
check("unlogged cell change refused", "not in the change log: 9 cells changed in 'amount'" in msg, msg[:90])
check("refusal says how to recover", "expenses = current('expenses')" in msg)
check("current() gives the logged version", refused(fill_missing, current("expenses"), "category", "x") == "")
fresh()
exp, _ = quiet(load, DATA / "expenses.csv")

msg = refused(save_clean, exp[exp["amount"] > 0])    # filtered, never logged
check("unlogged filter refused at save", "9 rows removed" in msg and "not in the change log" in msg, msg[:80])

relabelled = quiet(drop_rows, exp, exp["amount"] < 0, reason="refunds")[0].reset_index(drop=True)
msg = refused(save_clean, relabelled)
check("reset_index refused", "rows removed" in msg and "rows added" in msg, msg[:90])

fresh()
exp, _ = quiet(load, DATA / "expenses.csv")
quiet(drop_rows, exp, exp["amount"] < 0, reason="refunds")    # result never assigned!
msg = refused(fill_missing, exp, "category", "Uncategorized")
check("forgotten assignment caught", "9 rows added" in msg, msg[:80])

plain = pd.read_csv(DATA / "expenses.csv")                     # not via load()
check("table not from load() refused", "wasn't loaded with load()" in refused(drop_duplicates, plain))

exp = current("expenses")
other = pd.read_csv(DATA / "employees.csv")
check("condition on another table refused",
      "condition on this same table" in refused(drop_rows, exp, other["salary"] > 0, reason="x"))
undecided = (exp["amount"] > 0).astype(object)
undecided.iloc[0] = None                                      # neither True nor False
check("condition with missing values refused",
      "has missing values" in refused(drop_rows, exp, undecided, reason="x"))
check("reason required", "Give a reason" in refused(drop_rows, exp, exp["amount"] < 0, reason=" "))
check("unknown column refused", "No column 'amnt'" in refused(fill_missing, exp, "amnt", 0))

# =============================================================================
# 3. The new fix helpers, on the Stage A problems
# =============================================================================
fresh()
orders_csv = csv("orders", pd.DataFrame({
    "order_id": range(1, 9),
    "region": ["North", "North ", " South", "East  Coast", "-", "unknown", None, "North"],
    "total": ["$1,200", "45", "3.50", "1,000", "abc", "12", "7", "8"],
    "units": ["1,200", "45", "3", "1,000", "2", "12", "7", "8"],
    "order_date": ["2025-01-03", "01/23/2025", "Mar 3, 2025", "2025-01-06",
                   "2025-01-07", "02/14/2025", "2025-01-09", "2025-01-10"],
    "slash_date": ["03/04/2025"] * 8,
}))
orders, _ = quiet(load, orders_csv)

orders, msg = quiet(strip_spaces, orders)
check("strip_spaces: all text columns", orders["region"].tolist()[:4] == ["North", "North", "South", "East Coast"])
check("strip_spaces: only changed columns logged", [e["column"] for e in cleaning._LOG] == ["region"], msg.strip())
check("strip_spaces: missing stays missing", orders["region"].isna().sum() == 1)

orders, msg = quiet(placeholders_to_missing, orders)
check("placeholders -> missing", orders["region"].isna().sum() == 3 and
      cleaning._LOG[-1]["details"]["values"] == {"-": 1, "unknown": 1}, str(cleaning._LOG[-1]["details"]))

msg = refused(to_numbers, orders, "total")
check("to_numbers: refuses non-numbers", "total: not numbers: 1 value (e.g. 'abc')" in msg, msg[:70])
orders, _ = quiet(to_numbers, orders, "total", invalid="missing")
check("to_numbers: invalid='missing'", orders["total"].tolist()[:4] == [1200.0, 45.0, 3.5, 1000.0]
      and pd.isna(orders["total"][4]))
orders, _ = quiet(to_numbers, orders, "units")
check("to_numbers: whole numbers -> Int64", str(orders["units"].dtype) == "Int64"
      and orders["units"].tolist()[:2] == [1200, 45], str(orders["units"].dtype))

orders, _ = quiet(to_dates, orders, "order_date")
check("to_dates: mixed formats converted", str(orders["order_date"].dtype).startswith("datetime64")
      and orders["order_date"][1] == pd.Timestamp("2025-01-23")
      and orders["order_date"][2] == pd.Timestamp("2025-03-03"))
msg = refused(to_dates, orders, "slash_date")
check("to_dates: ambiguous order refused", "could be day first or month first" in msg, msg[:70])
orders, _ = quiet(to_dates, orders, "slash_date", day_first=True)
check("to_dates: day_first=True", orders["slash_date"][0] == pd.Timestamp("2025-04-03"))

n_before = len(cleaning._LOG)
orders, msg = quiet(undo, orders)
check("undo: last change removed", len(cleaning._LOG) == n_before - 1
      and orders["slash_date"][0] == "03/04/2025", msg.strip())
orders, _ = quiet(to_dates, orders, "slash_date", day_first=False)
check("redo with the other order", orders["slash_date"][0] == pd.Timestamp("2025-03-04"))
check("log numbering continuous", [e["n"] for e in cleaning._LOG] == list(range(1, len(cleaning._LOG) + 1)))

impossible, _ = quiet(load, csv("dates", pd.DataFrame({"d": ["2025-02-30", "2025-01-01"]})))
check("to_dates: impossible date refused", "not real dates in the given order: 1 value (e.g. '2025-02-30')" in refused(to_dates, impossible, "d"))

# custom_fix: logs exactly what changed; an error inside logs nothing.
with contextlib.redirect_stdout(io.StringIO()):
    with custom_fix(orders, "cap units at 100") as fix:
        fix.df.loc[fix.df["units"] > 100, "units"] = 100
orders = fix.df
last = cleaning._LOG[-1]
check("custom_fix logged", last["action"] == "custom_fix" and last["reason"] == "cap units at 100"
      and last["cells_changed"] == 2, last["summary"])
n_before = len(cleaning._LOG)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        with custom_fix(orders, "broken fix") as fix:
            fix.df.loc[:, "units"] = 0
            raise ValueError("model code failed halfway")
except ValueError:
    pass
check("failed custom_fix logs nothing", len(cleaning._LOG) == n_before)
check("table still the logged version", refused(cleaning._check_unchanged, orders, "orders") == "")
try:
    with contextlib.redirect_stdout(io.StringIO()):
        with custom_fix(orders, "double rows") as fix:
            fix.df = pd.concat([fix.df, fix.df])   # rows can no longer be told apart
    msg = ""
except CleaningError as e:
    msg = str(e)
check("duplicate row labels refused", "duplicate row labels" in msg, msg[:60])

# =============================================================================
# 4. Smaller behaviours
# =============================================================================
fresh()
t, _ = quiet(load, csv("t", pd.DataFrame({"item_id": [1, 2, 3], "qty": [5, 6, 7],
                                          "dept": ["HR", "Sales", "sales "]})))
t, _ = quiet(set_missing, t, t["item_id"] == 2, "qty", reason="typo")
check("set_missing on whole numbers -> Int64", str(t["qty"].dtype) == "Int64"
      and t["qty"].isna().sum() == 1 and t["qty"][0] == 5)
t, _ = quiet(merge_variants, t, "dept")
check("merge_variants: spaces-only variant merged", t["dept"].tolist() == ["HR", "Sales", "Sales"])
_, msg = quiet(merge_variants, t, "dept")
check("nothing to change: not logged", "nothing to change; not logged" in msg
      and len(cleaning._LOG) == 2)
_, msg = quiet(drop_rows, t, t["item_id"] == 999, reason="unknown item")
check("no match: warned, not logged", "WARNING: no rows in t match" in msg and
      "another table" in msg and len(cleaning._LOG) == 2, msg.strip()[:60])
_, msg = quiet(set_missing, t, t["item_id"] == 999, "qty", reason="x")
check("no match in set_missing: warned too", "WARNING: no rows in t match" in msg)
t2, msg = quiet(load, csv("t", pd.DataFrame({"item_id": [1], "qty": [1], "dept": ["x"]})))
check("loading again clears that table's log", len(cleaning._LOG) == 0
      and "2 earlier log entries cleared" in msg, msg.strip())

zeros, msg = quiet(load, csv("zeros", pd.DataFrame({"account_id": ["00012345", "00099999"],
                                                     "zip": ["02134", "10001"], "amount": ["0.5", "12"]})))
check("load keeps leading zeros", zeros["account_id"].tolist() == ["00012345", "00099999"]
      and zeros["zip"].tolist() == ["02134", "10001"], str(zeros["account_id"].tolist()))
check("load still reads '0.5' as a number", pd.api.types.is_numeric_dtype(zeros["amount"]))
_, date_msg = quiet(load, csv("zdates", pd.DataFrame({"d": ["01/23/2025", "02/14/2025"]})))
check("dates starting with 0 are not 'leading zeros'", "leading zeros" not in date_msg, date_msg.strip())
check("load says which columns kept zeros", "keep leading zeros: 'account_id', 'zip'" in msg, msg.strip())

missing_file = refused(load, "/data/orders.csv")
check("missing file: says so and lists real files", "No file /data/orders.csv" in missing_file
      and "Files available" in missing_file, missing_file[:60])

dated, _ = quiet(load, csv("dated", pd.DataFrame({"id": [1, 2], "d": ["01/23/2025", "2025-02-01"]})))
dated, _ = quiet(to_dates, dated, "d")
quiet(save_clean, dated)
check("dates saved as ISO text", (OUT / "clean" / "dated.csv").read_text().splitlines()[1:]
      == ["1,2025-01-23", "2,2025-02-01"])
check("log JSON readable with Int64 and dates",
      json.loads((OUT / "clean" / "change_log.json").read_text())["changes"][0]["action"] == "to_dates")

check("dictionary: real dates shown without a time", "2025-01-23" in (OUT / "clean" / "data_dictionary.md")
      .read_text() and "00:00:00" not in (OUT / "clean" / "data_dictionary.md").read_text())

# The report shows remaining issues without the instructions meant for the model.
from report import _for_people
check("report: fix hint cut", _for_people("category: spelling variants 'Meals' 1,701 / 'meals' 15 "
      "(fix: merge_variants)") == "category: spelling variants 'Meals' 1,701 / 'meals' 15")
check("report: suggestion cut", _for_people("salary: 7 missing values (suggest: leave missing, unless the "
      "user gives a fill value; then fill_missing)") == "salary: 7 missing values")
check("report: question to the model cut", _for_people("ship_date: 1 date in the future, latest 2099-05-01 "
      "(order_id=2); say if this column can hold future dates") ==
      "ship_date: 1 date in the future, latest 2099-05-01 (order_id=2)")
check("report: future-date hint and condition cut", _for_people("ship_date: 1 date in the future, latest 2099-05-01 "
      "(order_id=2); say if this column can hold future dates (suggest: if not, after to_dates, select it with "
      "the condition ship_date > '2026-11-14'; ask whether to set the dates to missing or drop the rows)") ==
      "ship_date: 1 date in the future, latest 2099-05-01 (order_id=2)")
check("report: verdict kept", _for_people("amount: plausible: 53 values far from typical, most extreme 6,372.55 "
      "(expense_id=5685), but none stands apart from the rest, so they look like real values (suggest: keep them)")
      .endswith("so they look like real values"))

# custom_fix has no single column: the dictionary files its cell edits under
# the columns it changed, and only its row removals under whole rows.
tk, _ = quiet(load, csv("tk", pd.DataFrame({"ticket_id": ["T1", "T2", "T2"],
                                            "priority": ["High", "Med", "Low"]})))
with custom_fix(tk, "Med means Medium") as fix:
    fix.df.loc[fix.df["priority"] == "Med", "priority"] = "Medium"
tk = fix.df
with custom_fix(tk, "drop second T2") as fix:
    fix.df = fix.df[~fix.df["ticket_id"].duplicated()]
tk = fix.df
check("log: cells per column recorded", cleaning._LOG[-2]["cells_by_column"] == {"priority": 1},
      str(cleaning._LOG[-2].get("cells_by_column")))
quiet(save_clean, tk)
tk_dict = (OUT / "clean" / "data_dictionary.md").read_text().split("## tk")[1].split("\n## ")[0]
check("dictionary: custom_fix cell edit under its column",
      "| priority |" in tk_dict and "custom_fix: Med means Medium (1 cell changed)" in
      next(line for line in tk_dict.splitlines() if line.startswith("| priority |")), tk_dict)
whole_line = next((line for line in tk_dict.splitlines() if line.startswith("Changes to whole rows")), "")
check("dictionary: cell-only custom_fix not under whole rows", "Med means Medium" not in whole_line, whole_line)
check("dictionary: row-removing custom_fix still under whole rows", "drop second T2" in whole_line, whole_line)

# Text the PDF's built-in font can't draw must not break the report.
odd, _ = quiet(load, csv("odd", pd.DataFrame({"id": [1, 2], "name": ["Zoë → Ünal", "東京 office"]})))
odd, _ = quiet(drop_rows, odd, odd["id"] == 2, reason="test – removal")
quiet(save_clean, odd)
check("report survives non-Latin text", (OUT / "clean" / "cleaning_report.pdf").read_bytes().startswith(b"%PDF"))

_, out = quiet(show_log)
check("show_log lists changes", "[1] dated: to_dates 'd'" in out, out.strip().splitlines()[-1])