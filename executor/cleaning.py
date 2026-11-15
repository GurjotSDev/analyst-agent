"""
Fix helpers with a change log: every change to a table is recorded, and
nothing reaches the saved file without a log entry.

The guarantee works like a chain of custody. load() records each table as
read from its file. Every fix helper first checks that the table it's given
is EXACTLY the last logged version; then it applies one change, logs what
changed (rows removed, cells changed, before -> after examples), and records
the result as the new logged version. save_clean() makes the same check
before writing. So a change made any other way (a stray df.loc[...] = ...,
a reset_index, a forgotten filter) is caught at the next helper or at save,
with a description of exactly what changed.

    orders = load("/data/orders.csv")                  # table name: "orders"
    orders = drop_duplicates(orders)
    orders = drop_rows(orders, orders["total"] < 0, reason="refunds")
    orders = set_missing(orders, orders["order_id"] == 42, "total", reason="data error")
    orders = fill_missing(orders, "region", "Unknown")
    orders = merge_variants(orders, "region")
    orders = strip_spaces(orders)                      # every text column, or one
    orders = placeholders_to_missing(orders)           # '-', '?', 'unknown' -> NaN
    orders = to_numbers(orders, "total")               # '$1,200' -> 1200
    orders = to_dates(orders, "order_date", day_first=False)
    with custom_fix(orders, "cap discounts at 100%") as fix:
        fix.df.loc[fix.df["discount"] > 1, "discount"] = 1
    orders = fix.df
    orders = undo(orders)                              # back one logged step
    show_log()
    save_clean(orders)                                 # output/clean/orders.csv

The check is against mistakes, not against code trying to cheat: model code
runs in the same Python process and could rewrite these records. The sandbox
and the user's approval of each code block cover that side.
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyst_helpers import (_as_numbers, _examples, _is_identifier, _is_placeholder, _is_text,
                             _layout_of, _parse_dates, _plural, _slash_order, _variant_groups)

# The kernel runs in the output folder; in Docker OUTPUT_DIR is /output.
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
MAX_LOGGED_IDS = 1000   # removed-row IDs kept per log entry (the count is always exact)
MAX_EXAMPLES = 5        # before -> after examples kept per log entry


class CleaningError(Exception):
    """A fix was refused. The message says why and what to do instead."""


# The records, one set per kernel. A kernel reset clears them along with
# every DataFrame, so they always describe the data that exists.
_SOURCES: dict[str, str] = {}               # table -> file it was loaded from
_ORIGINALS: dict[str, pd.DataFrame] = {}    # table -> as read from the file
_HISTORY: dict[str, list[pd.DataFrame]] = {}  # table -> every logged version, oldest first
_LOG: list[dict] = []                       # every logged change, in order
_SAVED: dict[str, pd.DataFrame] = {}        # table -> the version last saved to output/clean/


# ---------------------------------------------------------------------------
# Comparing two versions of a table
# ---------------------------------------------------------------------------

def _diff(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Everything that differs between two versions of a table.

    Rows are matched by their index (row label). Filters keep the labels of
    the rows they keep, so a dropped row shows up as a missing label. That's
    also why reset_index() counts as a change: it relabels the rows, and the
    labels are how every later check knows which row is which.
    """
    d = {
        "rows_removed": list(before.index.difference(after.index)),
        "rows_added": list(after.index.difference(before.index)),
        "columns_removed": [c for c in before.columns if c not in after.columns],
        "columns_added": [c for c in after.columns if c not in before.columns],
        "duplicate_labels": bool(after.index.duplicated().any()),
        "cells": {},          # column -> number of changed cells
        "dtypes": {},         # column -> (old dtype, new dtype)
        "examples": [],       # a few (row, column, before, after)
    }
    if d["duplicate_labels"]:
        return d   # rows can't be matched one to one; reported as a structural change
    rows = before.index.intersection(after.index)
    for col in before.columns.intersection(after.columns):
        if str(before[col].dtype) != str(after[col].dtype):
            d["dtypes"][col] = (str(before[col].dtype), str(after[col].dtype))
        # Compare as plain Python objects, so '1,200' vs 1200.0 or a string vs a
        # date compare as "different" instead of raising. Two missing values
        # count as the same.
        old = before.loc[rows, col].astype(object)
        new = after.loc[rows, col].astype(object)
        same = (old == new).fillna(False).astype(bool) | (old.isna() & new.isna())
        changed = same.index[~same.to_numpy()]
        if len(changed):
            d["cells"][col] = len(changed)
            # Examples whose text visibly changes come first: after to_dates,
            # '2025-06-27' -> 2025-06-27 only changed type, while
            # '12/18/2025' -> 2025-12-18 shows what the conversion did.
            visible = [i for i in changed if str(_plain(old[i])) != str(_plain(new[i]))]
            ordered = visible + [i for i in changed if i not in set(visible)]
            for label in ordered[:MAX_EXAMPLES - len(d["examples"])]:
                d["examples"].append({"row": _label(before, label), "column": col,
                                      "before": _plain(old[label]), "after": _plain(new[label])})
    return d


def _is_empty(d: dict) -> bool:
    return not (d["rows_removed"] or d["rows_added"] or d["columns_removed"] or
                d["columns_added"] or d["duplicate_labels"] or d["cells"])


def _describe(d: dict) -> str:
    """'3 rows removed, 12 cells changed in category' and so on."""
    parts = []
    if d["duplicate_labels"]:
        parts.append("rows have duplicate labels (e.g. after concat), so they can't be matched")
    if d["rows_removed"]:
        parts.append(f"{_plural(len(d['rows_removed']), 'row')} removed")
    if d["rows_added"]:
        parts.append(f"{_plural(len(d['rows_added']), 'row')} added")
    for col, n in d["cells"].items():
        parts.append(f"{_plural(n, 'cell')} changed in {col!r}")
    if d["columns_removed"]:
        parts.append(f"columns removed: {_examples(d['columns_removed'], 10)}")
    if d["columns_added"]:
        parts.append(f"columns added: {_examples(d['columns_added'], 10)}")
    return "; ".join(parts) or "nothing"


def _label(df: pd.DataFrame, index) -> str:
    """A row's name for people: its ID (order_id=42) if the table has one."""
    for col in df.columns:
        if _is_identifier(col):
            return f"{col}={_plain(df.at[index, col])}"
    return f"row {index}"


def _plain(x):
    """A value as plain JSON-friendly Python: numpy ints become ints, missing
    values become None, dates become text."""
    if x is None or (not isinstance(x, (list, dict)) and pd.isna(x)):
        return None
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d %H:%M:%S") if x.hour or x.minute or x.second else x.strftime("%Y-%m-%d")
    if isinstance(x, np.generic):
        return x.item()
    return x


# ---------------------------------------------------------------------------
# The chain of custody
# ---------------------------------------------------------------------------

def _table_of(df) -> str:
    if not isinstance(df, pd.DataFrame):
        raise CleaningError(f"Expected a DataFrame, got {type(df).__name__}.")
    table = df.attrs.get("table")
    if table not in _HISTORY:
        raise CleaningError("This DataFrame wasn't loaded with load(), so its changes can't be "
                            "tracked. Load the file with load('/data/<file>.csv') and clean that.")
    return table


def _check_unchanged(df: pd.DataFrame, table: str) -> None:
    """Refuse a table that differs from its last logged version."""
    d = _diff(_HISTORY[table][-1], df)
    if not _is_empty(d):
        raise CleaningError(
            f"{table} has changes that are not in the change log: {_describe(d)}. Every change "
            f"must go through a fix helper or custom_fix(). To go back to the last logged "
            f"version: {table} = current({table!r})")


def _record(table: str, action: str, column, reason: str, before: pd.DataFrame,
            after: pd.DataFrame, details: dict | None = None) -> pd.DataFrame:
    """Log one change and make `after` the table's new logged version."""
    after.attrs["table"] = table
    d = _diff(before, after)
    if _is_empty(d):
        print(f"{action}({table}{', ' + repr(column) if column else ''}): nothing to change; not logged")
        return after
    if d["duplicate_labels"]:
        raise CleaningError("The result has duplicate row labels, so its rows can't be matched to "
                            "the original. Don't concat a table with itself or with copies of its rows.")

    removed = d["rows_removed"]
    entry = {
        "n": len(_LOG) + 1,
        "table": table,
        "action": action,
        "column": column,
        "reason": reason,
        "summary": _describe(d),
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_removed": len(removed),
        "rows_added": len(d["rows_added"]),
        "cells_changed": sum(d["cells"].values()),
        # Per column, so the data dictionary can file a custom_fix (logged
        # with no single column) under each column it actually changed.
        "cells_by_column": dict(d["cells"]),
        "removed": [_label(before, i) for i in removed[:MAX_LOGGED_IDS]],
        "examples": d["examples"],
        "type_changes": {c: f"{a} -> {b}" for c, (a, b) in d["dtypes"].items()},
        "details": details or {},
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    _LOG.append(entry)
    _HISTORY[table].append(after.copy())
    shown = f" (e.g. {', '.join(entry['removed'][:3])})" if removed else ""
    print(f"[{entry['n']}] {table}: {action}{' ' + repr(column) if column else ''}: "
          f"{entry['summary']}{shown}")
    return after


def _start(df) -> tuple[str, pd.DataFrame]:
    """Every fix begins here: which table, and is it the logged version?"""
    table = _table_of(df)
    _check_unchanged(df, table)
    return table, df.copy()


def _need_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise CleaningError(f"No column {col!r}. Columns: {_examples(list(df.columns), 30)}")


def _need_reason(reason: str) -> None:
    if not isinstance(reason, str) or not reason.strip():
        raise CleaningError("Give a reason (the user's words), e.g. reason='refunds'. "
                            "It goes into the change log and the report.")


def _need_mask(df: pd.DataFrame, mask) -> pd.Series:
    """A row selection must be True/False for exactly this table's rows."""
    if not isinstance(mask, pd.Series) or not mask.index.equals(df.index):
        raise CleaningError("The row selection must be a condition on this same table, e.g. "
                            "orders['total'] < 0 (built from the table you pass in).")
    if mask.isna().any():
        raise CleaningError("The row selection has missing values, so some rows are neither "
                            "selected nor not. Decide them first, e.g. (cond).fillna(False).")
    return mask.astype(bool)


def _warn_no_match(action: str, table: str, verb: str) -> None:
    """A selection that matches no rows is usually a mistake (often the wrong
    table), not a fact about the data, so say so instead of passing silently."""
    print(f"{action}({table}): WARNING: no rows in {table} match this condition, so nothing was "
          f"{verb} or logged. If the user asked for this fix, check the condition, and check "
          f"whether these rows are in another table (compare_keys shows where each ID is).")


# ---------------------------------------------------------------------------
# Loading, and going back
# ---------------------------------------------------------------------------

def load(path, table: str | None = None) -> pd.DataFrame:
    """Read a CSV and start its change log. The table name defaults to the
    file name: load('/data/orders.csv') is table 'orders'."""
    path = Path(path)
    table = table or path.stem
    if not path.exists():
        # List the real files, so a wrong path can be corrected.
        files =sorted(str(p) for p in Path("/data").rglob("*.csv")) if Path("/data").exists() else []
        raise CleaningError(f"No file {path}. Use exactly the paths listed under 'Files available' "
                            f"in your instructions. (Every CSV in /data: {', '.join(files) or 'none'}.)")
    # pandas guesses each column's type, and reads '00012345' as the number
    # 12345: the zeros are gone before any check can see them. So first read
    # every value exactly as written (dtype=str), find columns with leading
    # zeros, and read those as text. Only values made entirely of digits
    # count: '00012345' does; '0.5' and the date '01/23/2025' don't.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    zero_cols = [c for c in raw.columns if raw[c].str.fullmatch(r"0\d+").any()]
    df = pd.read_csv(path, dtype={c: "str" for c in zero_cols})
    df.attrs["table"] = table
    cleared = sum(e["table"] == table for e in _LOG)
    # Loading again starts that table over, so its old entries no longer apply.
    _LOG[:] = [e for e in _LOG if e["table"] != table]
    _SOURCES[table] = str(path)
    _SAVED.pop(table, None)   # an earlier saved version no longer matches this log
    _ORIGINALS[table] = df.copy()
    _HISTORY[table] = [df.copy()]
    entries = f"{cleared} earlier log {'entry' if cleared == 1 else 'entries'}"
    note = f" (loaded again: its {entries} cleared)" if cleared else ""
    kept = f"; kept as text to keep leading zeros: {_examples(zero_cols, 10)}" if zero_cols else ""
    print(f"load: {table} = {path.name}, {len(df):,} rows x {df.shape[1]} columns{kept}{note}")
    return df


def current(table: str) -> pd.DataFrame:
    """The table's last logged version, e.g. to throw away unlogged changes."""
    if table not in _HISTORY:
        raise CleaningError(f"No table {table!r}. Loaded: {_examples(list(_HISTORY), 20)}")
    return _HISTORY[table][-1].copy()


def undo(df_or_table) -> pd.DataFrame:
    """Remove the table's last logged change and return the version before it.
    Use as: orders = undo(orders)"""
    table = df_or_table if isinstance(df_or_table, str) else _table_of(df_or_table)
    if table not in _HISTORY:
        raise CleaningError(f"No table {table!r}.")
    if len(_HISTORY[table]) == 1:
        raise CleaningError(f"{table} has no logged changes to undo.")
    last = max(i for i, e in enumerate(_LOG) if e["table"] == table)
    entry = _LOG.pop(last)
    _HISTORY[table].pop()
    for i, e in enumerate(_LOG, start=1):
        e["n"] = i   # keep the numbering continuous
    print(f"undo: removed [{entry['n']}] {table}: {entry['action']} ({entry['summary']})")
    return current(table)


# ---------------------------------------------------------------------------
# The fix helpers
# ---------------------------------------------------------------------------

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, keeping the first of each."""
    table, new = _start(df)
    return _record(table, "drop_duplicates", None, "exact duplicate rows", df,
                   new[~new.duplicated()])


def drop_rows(df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    """Remove the rows where mask is True, e.g. drop_rows(orders, orders['total'] < 0,
    reason='refunds')."""
    table, new = _start(df)
    _need_reason(reason)
    mask = _need_mask(df, mask)
    if not mask.any():
        _warn_no_match("drop_rows", table, "removed")
        return df
    return _record(table, "drop_rows", None, reason, df, new[~mask])


def set_missing(df: pd.DataFrame, mask: pd.Series, col: str, reason: str) -> pd.DataFrame:
    """Set ONE column to missing in the selected rows, keeping the rows:
    set_missing(emp, emp['employee_id'] == 1018, 'salary', reason='data error')."""
    table, new = _start(df)
    _need_column(df, col)
    _need_reason(reason)
    mask = _need_mask(df, mask)
    if not mask.any():
        _warn_no_match("set_missing", table, "changed")
        return df
    if pd.api.types.is_integer_dtype(new[col]) and not isinstance(new[col].dtype, pd.Int64Dtype):
        # A plain int64 column can't hold a missing value; Int64 (capital I,
        # pandas' nullable integer) can, and keeps the numbers whole.
        new[col] = new[col].astype("Int64")
    new.loc[mask, col] = np.nan
    return _record(table, "set_missing", col, reason, df, new)


def fill_missing(df: pd.DataFrame, col: str, value, reason: str = "") -> pd.DataFrame:
    """Replace missing values in one column with a value, e.g. 'Uncategorized'."""
    table, new = _start(df)
    _need_column(df, col)
    n = int(new[col].isna().sum())
    new[col] = new[col].fillna(value)
    return _record(table, "fill_missing", col, reason or f"missing values set to {value!r}",
                   df, new, {"value": _plain(value), "filled": n})


def merge_variants(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Merge spellings that differ only in capitals or spaces ('meals', ' Meals')
    into the spelling most rows already use. Nothing else changes: 'HR' stays
    'HR' when it has no variants."""
    table, new = _start(df)
    _need_column(df, col)
    mapping = {}
    for spellings in _variant_groups(new[col]):
        target = spellings.index[0]   # value_counts() lists the most common first
        for spelling in spellings.index[1:]:
            mapping[spelling] = target
    # .replace() with a dict swaps only the exact values listed in it.
    new[col] = new[col].replace(mapping)
    return _record(table, "merge_variants", col, "spelling variants", df, new,
                   {"mapping": mapping})


def strip_spaces(df: pd.DataFrame, col: str | None = None) -> pd.DataFrame:
    """Remove spaces at the start and end of text values, and turn runs of
    spaces inside them into one. One column, or every text column (col=None),
    each logged separately."""
    table, new = _start(df)
    cols = [col] if col else [c for c in new.columns if _is_text(new[c])]
    if col:
        _need_column(df, col)
    for c in cols:
        before = new
        new = before.copy()
        new[c] = before[c].str.strip().str.replace(r"\s+", " ", regex=True)
        if col is None and _is_empty(_diff(before, new)):
            new = before   # nothing to fix in this column: skip it quietly
            continue
        new = _record(table, "strip_spaces", c, "stray spaces", before, new)
    return new


def placeholders_to_missing(df: pd.DataFrame, col: str | None = None) -> pd.DataFrame:
    """Turn text that means "no value" ('-', '?', 'unknown', 'none'...) into real
    missing values. One column, or every text column (col=None)."""
    table, new = _start(df)
    cols = [col] if col else [c for c in new.columns if _is_text(new[c])]
    if col:
        _need_column(df, col)
    for c in cols:
        before = new
        new = before.copy()
        found = before[c].dropna()
        found = found[_is_placeholder(found)]
        new.loc[found.index, c] = np.nan
        if col is None and found.empty:
            new = before
            continue
        new = _record(table, "placeholders_to_missing", c, "placeholder values", before, new,
                      {"values": {str(k): int(v) for k, v in found.value_counts().items()}})
    return new


def to_numbers(df: pd.DataFrame, col: str, invalid: str = "refuse") -> pd.DataFrame:
    """Convert text like '$1,200', '45%' or ' 3.5' to numbers. '%' is dropped,
    not divided: '45%' becomes 45.

    Values that aren't numbers stop the conversion, so nothing is lost
    silently. If the user approved it, invalid='missing' turns them into
    missing values instead."""
    table, new = _start(df)
    _need_column(df, col)
    s = new[col]
    if not _is_text(s):
        print(f"to_numbers({table}, {col!r}): already numbers; nothing to change")
        return df
    values = s.dropna().astype(str)
    numbers = _as_numbers(values)
    bad = values[numbers.isna()]
    if len(bad) and invalid != "missing":
        hint = (" Some are placeholders: placeholders_to_missing() first."
                if _is_placeholder(bad).any() else "")
        raise CleaningError(
            f"{col}: not numbers: {_plural(len(bad), 'value')} (e.g. {_examples(bad)}).{hint} "
            f"Fix them first, or, if the user approved it, use "
            f"to_numbers(..., {col!r}, invalid='missing') to make them missing.")
    result = pd.Series(np.nan, index=s.index, dtype="float64")
    result[numbers.index] = numbers
    # Whole numbers become a nullable integer column (Int64), so they're
    # saved as 1200, not 1200.0, and missing values stay possible.
    if result.dropna().apply(float.is_integer).all():
        result = result.astype("Int64")
    new[col] = result
    return _record(table, "to_numbers", col, "numbers stored as text", df, new,
                   {"made_missing": [str(b) for b in bad[:MAX_EXAMPLES]] if len(bad) else [],
                    "percent_signs_dropped": bool(values.str.contains("%").any())})


def to_dates(df: pd.DataFrame, col: str, day_first: bool | None = None,
             invalid: str = "refuse") -> pd.DataFrame:
    """Convert a text column of dates, in any mix of formats, into real dates.

    Dates like 03/04/2025 need day_first=True (3 April) or day_first=False
    (March 4) unless the values themselves decide it. Values that aren't real
    dates stop the conversion unless invalid='missing' (with the user's OK)."""
    table, new = _start(df)
    _need_column(df, col)
    s = new[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        print(f"to_dates({table}, {col!r}): already dates; nothing to change")
        return df
    values = s.dropna().astype(str)
    layout = _layout_of(values)
    orders = {}
    for label, sep in (("slash", "/"), ("dash", "-")):
        rows = layout == label
        if not rows.any():
            continue
        found = _slash_order(values[rows].str.strip().str.split(sep, expand=True))
        if found == "both":
            raise CleaningError(f"{col}: some dates only fit day first and others only month first, "
                                "so one order can't be right for all. Ask the user how to handle them.")
        if day_first is None and found == "unknown":
            raise CleaningError(f"{col}: dates like {_examples(values[rows], 1)} could be day first or "
                                "month first. Ask the user, then pass day_first=True or False.")
        if day_first is not None:
            found = "day first" if day_first else "month first"
        orders[label] = found
    parsed = _parse_dates(values, layout, orders)
    bad = values[parsed.isna()]
    if len(bad) and invalid != "missing":
        raise CleaningError(
            f"{col}: not real dates in the given order: {_plural(len(bad), 'value')} "
            f"(e.g. {_examples(bad)}). Fix them first, or, if the user approved it, use "
            f"to_dates(..., {col!r}, invalid='missing') to make them missing.")
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    result[parsed.index] = parsed
    new[col] = result
    return _record(table, "to_dates", col, "text dates converted", df, new,
                   {"order": {k: v for k, v in orders.items()},
                    "made_missing": [str(b) for b in bad[:MAX_EXAMPLES]] if len(bad) else []})


class _Fix:
    """What custom_fix() hands to the with-block: change fix.df freely."""
    def __init__(self, df: pd.DataFrame):
        self.df = df


@contextmanager
def custom_fix(df: pd.DataFrame, description: str):
    """For an approved fix no helper covers. Change fix.df inside the block;
    when it ends, exactly what changed is logged with the description:

        with custom_fix(orders, "cap discounts at 100%") as fix:
            fix.df.loc[fix.df["discount"] > 1, "discount"] = 1
        orders = fix.df

    If the block raises an error, nothing is logged and the table keeps its
    last logged version."""
    table, new = _start(df)
    _need_reason(description)
    fix = _Fix(new)
    yield fix   # the with-block runs here
    if not isinstance(fix.df, pd.DataFrame):
        raise CleaningError("fix.df must still be a DataFrame at the end of the block.")
    fix.df = _record(table, "custom_fix", None, description, df, fix.df)
    print(f"  now use: {table} = fix.df")


# ---------------------------------------------------------------------------
# The log and saving
# ---------------------------------------------------------------------------

def show_log(table: str | None = None) -> None:
    """Print the change log (one table, or all)."""
    entries = [e for e in _LOG if table is None or e["table"] == table]
    if not entries:
        print("Change log: empty")
        return
    print(f"Change log ({_plural(len(entries), 'change')}):")
    for e in entries:
        print(f"  [{e['n']}] {e['table']}: {e['action']}"
              f"{' ' + repr(e['column']) if e['column'] else ''}: {e['summary']} ({e['reason']})")


def save_clean(df: pd.DataFrame) -> None:
    """Write the cleaned table to output/clean/<table>.csv, after checking that
    every change since load() is in the change log. Also (re)writes, for
    every table so far: change_log.json (the full log), cleaning_report.pdf
    and data_dictionary.md, so all three always match the saved files."""
    table = _table_of(df)
    _check_unchanged(df, table)
    out_dir = OUTPUT_DIR / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table}.csv"
    df.to_csv(path, index=False)

    original = _ORIGINALS[table]
    report = {
        "saved": datetime.now().isoformat(timespec="seconds"),
        "tables": {t: {"source": _SOURCES[t],
                       "original_rows": len(_ORIGINALS[t]),
                       "original_columns": list(_ORIGINALS[t].columns),
                       "current_rows": len(_HISTORY[t][-1]),
                       "changes": sum(e["table"] == t for e in _LOG)}
                   for t in _HISTORY},
        "changes": _LOG,
    }
    # default=_plain converts anything json can't (numpy numbers, dates).
    (out_dir / "change_log.json").write_text(json.dumps(report, indent=2, default=_plain),
                                             encoding="utf-8")
    _SAVED[table] = df.copy()
    _write_report_files(out_dir)
    n = sum(e["table"] == table for e in _LOG)
    print(f"save_clean: clean/{table}.csv, {len(df):,} rows x {df.shape[1]} columns "
          f"(was {len(original):,} x {original.shape[1]}), {_plural(n, 'logged change')}; "
          f"also updated: change_log.json, cleaning_report.pdf, data_dictionary.md")


def _write_report_files(out_dir: Path) -> None:
    # Imported here, not at the top: report.py pulls in fpdf2, which only
    # saving needs, and it keeps loading this module fast.
    from analyst_helpers import issues_in
    from report import write_dictionary, write_report
    tables = [{"table": t, "source": _SOURCES[t],
               "rows_before": len(_ORIGINALS[t]), "cols_before": _ORIGINALS[t].shape[1],
               "rows_after": len(_HISTORY[t][-1]), "cols_after": _HISTORY[t][-1].shape[1],
               "changes": sum(e["table"] == t for e in _LOG), "saved": t in _SAVED}
              for t in _HISTORY]
    remaining = {t: issues_in(_HISTORY[t][-1], t) for t in _HISTORY}
    write_report(out_dir / "cleaning_report.pdf", tables, _LOG, remaining)
    write_dictionary(out_dir / "data_dictionary.md", _SAVED, _SOURCES, _LOG, _is_identifier)