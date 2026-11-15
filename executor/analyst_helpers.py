"""
Tested data-quality helpers, preloaded into every kernel.

The model calls these instead of writing its own checking and cleaning code.
In testing, the model found most problems when asked, but its improvised code
had bugs: counting missing values as a category (hiding a misspelling), and
"fixing" capitalisation with .str.capitalize(), which also broke correct
values ('HR' -> 'Hr'). These functions are tested once against known data
(test_helpers.py), so the routine work is right every time and the model only
has to interpret and decide.

    profile(orders, "orders")              data-quality report for one table
    compare_keys(orders, customers, "customer_id", "orders", "customers")
                                           how two tables link up
    check(orders, customers)               both of the above for every table and
                                           every shared ID column, in one call

The fixes, and the change log that records them, are in cleaning.py.
"""

import re

import pandas as pd
from pandas.api import types

MAX_LISTED_VALUES = 15   # text columns with at most this many values get full counts
MISSING = "NaN (missing)"  # how missing values are shown. Not "<missing>": the model
                           # mistook that for a real text value and searched for it.

# Text that people type to mean "no value". pandas already reads '', 'NA', 'N/A',
# 'NULL', 'None' and 'nan' as missing, so these are the ones that get through
# as ordinary text. Compared after trimming spaces and lower-casing.
PLACEHOLDERS = frozenset({"", "-", "--", "---", "?", "??", "unknown", "unk", "missing",
                          "none", "nil", "null", "na", "n/a", "n.a.", "#n/a", "not available",
                          "not applicable", "tbd", "tba", "0000-00-00"})

# Date layouts we recognise, as (label, pattern). A column whose values match
# more than one of these has mixed date formats. Slash and dash dates with the
# year last are ambiguous (03/04/2025 is March 4 or 3 April), so their label is
# decided later from the values themselves.
_TIME = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"
DATE_LAYOUTS = [
    ("YYYY-MM-DD", re.compile(r"\d{4}-\d{1,2}-\d{1,2}" + _TIME)),
    ("YYYY/MM/DD", re.compile(r"\d{4}/\d{1,2}/\d{1,2}")),
    ("slash", re.compile(r"\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})")),
    ("dash", re.compile(r"\d{1,2}-\d{1,2}-\d{4}")),
    ("DD.MM.YYYY", re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}")),
    ("written month", re.compile(r"(?:[A-Za-z]{3,9}\.? \d{1,2},? \d{4})|(?:\d{1,2} [A-Za-z]{3,9}\.? \d{4})")),
]
OLDEST_PLAUSIBLE = pd.Timestamp("1900-01-01")

# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _fmt(x) -> str:
    """1250000.0 -> '1,250,000'; 83.61 -> '83.61'. Easier for a model to read."""
    if pd.isna(x):
        return MISSING
    x = float(x)
    return f"{x:,.0f}" if x.is_integer() else f"{x:,.2f}"


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _is_identifier(name: str) -> bool:
    """Columns like employee_id or id hold labels, not quantities, so a
    min/max/median of them means nothing. They get different checks."""
    name = name.lower()
    return name == "id" or name.endswith("_id")


def _repeated_key(df: pd.DataFrame, col: str) -> str | None:
    """The issue line for a key column that names more than one DIFFERENT row,
    e.g. ticket T1005 opened on two dates. Exact duplicate rows are left out:
    they have their own issue line and drop_duplicates fixes them, so counting
    them here too would report the same rows twice.

    Which copy is right is the user's call, so the line asks; it also hands
    the model a ready-made condition for the common answer, "keep the first"."""
    ids = df.drop_duplicates()[col].dropna()
    counts = ids[ids.duplicated(keep=False)].value_counts()
    if counts.empty:
        return None
    shown = ", ".join(f"{_plain_id(v)} x{n}" for v, n in counts.head(5).items())
    more = f", and {len(counts) - 5:,} more" if len(counts) > 5 else ""
    return (f"{col}: repeated key: {_plural(len(counts), 'ID')} on more than one different row "
            f"({shown}{more}); as the table's first ID column it should name one row each "
            f"(suggest: ask which row to keep for each ID; to keep the first, drop_rows with "
            f"the condition {col}.duplicated(), after drop_duplicates)")


def _plain_id(v) -> str:
    """An ID as people write it: 1018, not 1018.0 or np.int64(1018)."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v.item() if hasattr(v, "item") else v)


def _is_text(s: pd.Series) -> bool:
    # True for pandas 3's "str" dtype and for older object columns of strings.
    return types.is_string_dtype(s) and not types.is_numeric_dtype(s)


def _is_number(s: pd.Series) -> bool:
    # bool counts as numeric to pandas, but min/median of True/False is useless.
    return types.is_numeric_dtype(s) and not types.is_bool_dtype(s)


def _row_label(df: pd.DataFrame, index) -> str:
    """Name a row by its first identifier column if there is one
    (employee_id=1018), otherwise by its row number. This lets the model
    find that exact row again, e.g. df[df['employee_id'] == 1018]."""
    for col in df.columns:
        if _is_identifier(col):
            return f"{col}={df.at[index, col]}"
    return f"row {index}"


def _is_placeholder(v: pd.Series) -> pd.Series:
    """True for values like '-', '?', 'unknown': text that means "missing"."""
    return v.str.strip().str.lower().isin(PLACEHOLDERS)


def _variant_groups(s: pd.Series, spaces_only_too: bool = True) -> list[pd.Series]:
    """Groups of spellings that are the same after ignoring capitalisation and
    surrounding spaces, e.g. 'Meals' / 'meals ' / ' MEALS'. Each group is a
    value_counts() Series, most common spelling first. Only groups with more
    than one spelling are returned. Placeholders ('Unknown' / 'unknown') are
    left out: they are reported as placeholders, not as spellings.

    spaces_only_too=False skips groups that differ ONLY by spaces ('Sales' /
    'Sales '): profile() reports those as stray spaces instead, so the same
    problem isn't listed twice."""
    v = s.dropna()
    v = v[~_is_placeholder(v)]
    groups = []
    for _, group in v.groupby(v.str.strip().str.lower()):
        spellings = group.value_counts()
        if len(spellings) > 1 and (spaces_only_too or group.str.strip().nunique() > 1):
            groups.append(spellings)
    return groups


def _abbreviations(v: pd.Series) -> list[tuple[str, str, int, int]]:
    """Rare values that look like a shortened form of one common value in the
    same column: 'Med' 1 next to 'Medium' 11. Returns (short, full, count of
    short, count of full) for each.

    Deliberately narrow, because a wrong match is worse than a miss (and the
    model must ask the user about every match anyway):
    - the short form is the START of the full one ('Med' / 'Medium'), at
      least 2 letters, ignoring capitals and a trailing '.' ('Med.');
    - the full form doesn't continue with a space: 'North' and 'North East'
      are usually two real categories, not an abbreviation;
    - the full form is at least 3x as common: an abbreviation is the odd one out;
    - exactly ONE full form fits: 'Data' next to 'Data Lake' and 'Data
      Engineering' is ambiguous, so it isn't reported;
    - not values that differ only in capitals or spaces: _variant_groups
      reports those, and merge_variants fixes them.
    Only for short category columns (the caller checks), so free text such as
    names is never compared."""
    counts = v.str.strip().value_counts()
    found = []
    for short, n_short in counts.items():
        key = short.lower().rstrip(".")
        if len(key) < 2:
            continue
        fits = [(full, n_full) for full, n_full in counts.items()
                if full.lower() != short.lower()
                and full.lower().startswith(key) and len(full) > len(key)
                and full[len(key)] != " " and n_full >= 3 * n_short]
        if len(fits) == 1:
            found.append((short, fits[0][0], int(n_short), int(fits[0][1])))
    return found


def _examples(values, n: int = 3) -> str:
    """'a', 'b', 'c': repr() shows stray spaces and quotes, e.g. 'Sales '."""
    return ", ".join(repr(x) for x in list(dict.fromkeys(values))[:n])


def _layout_of(v: pd.Series) -> pd.Series:
    """For each text value, the label of the date layout it matches, or NaN."""
    layout = pd.Series(pd.NA, index=v.index, dtype="object")
    stripped = v.str.strip()
    for label, pattern in DATE_LAYOUTS:
        hit = layout.isna() & stripped.str.fullmatch(pattern)
        layout[hit] = label
    return layout


def _slash_order(parts: pd.DataFrame) -> str:
    """Which number comes first in dates like 03/04/2025. A first number over 12
    can only be a day; a second number over 12 can only be a day too. If
    neither ever happens, every value fits both readings."""
    first, second = parts[0].astype(int), parts[1].astype(int)
    day_first, month_first = bool((first > 12).any()), bool((second > 12).any())
    if day_first and month_first:
        return "both"
    if day_first:
        return "day first"
    return "month first" if month_first else "unknown"


def _parse_dates(v: pd.Series, layout: pd.Series, orders: dict) -> pd.Series:
    """Parse each value with the format of its own layout. One format for the
    whole column would turn every value in another layout into NaT."""
    parsed = pd.Series(pd.NaT, index=v.index, dtype="datetime64[ns]")
    s = v.str.strip()
    for label, rows in layout.dropna().groupby(layout.dropna()).groups.items():
        part = s[rows]
        if label == "YYYY-MM-DD":
            out = pd.to_datetime(part, format="ISO8601", errors="coerce")
        elif label == "YYYY/MM/DD":
            out = pd.to_datetime(part, format="%Y/%m/%d", errors="coerce")
        elif label == "DD.MM.YYYY":
            out = pd.to_datetime(part, format="%d.%m.%Y", errors="coerce")
        elif label in ("slash", "dash"):
            sep = "/" if label == "slash" else "-"
            day_first = orders[label] == "day first"
            a, b = ("%d", "%m") if day_first else ("%m", "%d")
            # Try 4-digit years, then 2-digit ones (%y: 25 -> 2025).
            out = pd.to_datetime(part, format=f"{a}{sep}{b}{sep}%Y", errors="coerce")
            out = out.fillna(pd.to_datetime(part, format=f"{a}{sep}{b}{sep}%y", errors="coerce"))
        else:  # written month: 'Mar 3, 2025', 'March 3 2025', '3 Mar 2025'
            clean = part.str.replace(r"[.,]", "", regex=True)
            out = pd.Series(pd.NaT, index=part.index, dtype="datetime64[ns]")
            for fmt in ("%b %d %Y", "%B %d %Y", "%d %b %Y", "%d %B %Y"):
                out = out.fillna(pd.to_datetime(clean, format=fmt, errors="coerce"))
        parsed[rows] = out.astype("datetime64[ns]")
    return parsed


def _check_date_range(df: pd.DataFrame, col: str, dates: pd.Series,
                      out: list[str], issues: list[str], text: bool = False) -> None:
    """Dates in the future, or before 1900: shared by text and real date columns.

    Each line hands the model a ready-made condition, as the outlier lines
    do. Without one, gemma wrote its own for "set the future week to
    missing": week_start > "2025-12-31", which matched EVERY week in a file
    of 2026 timesheets. The condition is the same cut-off the check used, so
    it selects exactly the dates counted here. text=True: the column is
    still text, so the condition only works after to_dates."""
    valid = dates.dropna()
    if valid.empty:
        return
    out.append(f"  range {valid.min():%Y-%m-%d} to {valid.max():%Y-%m-%d}")
    after = "after to_dates, " if text else ""
    today = pd.Timestamp.today().normalize()
    future = valid[valid > today]
    if len(future):
        last = future.idxmax()
        issues.append(f"{col}: {_plural(len(future), 'date')} in the future, latest "
                      f"{future.max():%Y-%m-%d} ({_row_label(df, last)}); say if this "
                      f"column can hold future dates (suggest: if not, {after}select "
                      f"{'it' if len(future) == 1 else 'them'} with the condition "
                      f"{col} > '{today:%Y-%m-%d}'; ask whether to set the dates to missing "
                      f"or drop the rows)")
    old = valid[valid < OLDEST_PLAUSIBLE]
    if len(old):
        first = old.idxmin()
        issues.append(f"{col}: {_plural(len(old), 'date')} before 1900, earliest "
                      f"{old.min():%Y-%m-%d} ({_row_label(df, first)}); likely typos "
                      f"(suggest: {after}select {'it' if len(old) == 1 else 'them'} with the "
                      f"condition {col} < '{OLDEST_PLAUSIBLE:%Y-%m-%d}'; ask whether to set the "
                      f"dates to missing or drop the rows)")


def _check_text_dates(df: pd.DataFrame, col: str, v: pd.Series, layout: pd.Series,
                      out: list[str], issues: list[str]) -> None:
    """Report formats, ambiguous day/month order, values that aren't dates,
    impossible dates (2025-02-30), and dates outside a plausible range."""
    orders = {}
    for label, sep in (("slash", "/"), ("dash", "-")):
        rows = layout == label
        if rows.any():
            orders[label] = _slash_order(v[rows].str.strip().str.split(sep, expand=True))
    names = {"day first": "DD{s}MM{s}YYYY", "month first": "MM{s}DD{s}YYYY",
             "unknown": "MM{s}DD{s}YYYY or DD{s}MM{s}YYYY", "both": "both MM{s}DD and DD{s}MM"}
    display = layout.copy()
    for label, sep in (("slash", "/"), ("dash", "-")):
        if label in orders:
            display[layout == label] = names[orders[label]].format(s=sep)

    counts = display.value_counts()
    if len(counts) == 1:
        out.append(f"{col}: dates, all {counts.index[0]}")
    else:
        parts = ", ".join(f"{k} {n:,} (e.g. {_examples(v[display == k], 1)})" for k, n in counts.items())
        out.append(f"{col}: mixed date formats: {parts}")
        issues.append(f"{col}: mixed date formats: " + ", ".join(f"{k} {n:,}" for k, n in counts.items())
                      + " (fix: to_dates)")
    for label, sep in (("slash", "/"), ("dash", "-")):
        if orders.get(label) == "both":
            rows = v[layout == label].str.strip()
            first = rows[rows.str.split(sep).str[0].astype(int) > 12]
            second = rows[rows.str.split(sep).str[1].astype(int) > 12]
            issues.append(f"{col}: some dates only fit day first ({_examples(first, 1)}), others only "
                          f"month first ({_examples(second, 1)}); the order can't be decided")
        elif orders.get(label) == "unknown":
            issues.append(f"{col}: day/month order unknown in {_plural(int((layout == label).sum()), 'date')} "
                          f"like {_examples(v[layout == label], 1)}; ask which order the source uses")

    not_dates = v[layout.isna()]
    if len(not_dates):
        issues.append(f"{col}: not dates: {_plural(len(not_dates), 'value')} "
                      f"(e.g. {_examples(not_dates)})")
    dates = _parse_dates(v, layout, orders)
    unreal = v[layout.notna() & dates.isna()]
    if len(unreal):
        issues.append(f"{col}: impossible dates: {_plural(len(unreal), 'value')} "
                      f"(e.g. {_examples(unreal)})")
    _check_date_range(df, col, dates, out, issues, text=True)


def _as_numbers(v: pd.Series) -> pd.Series:
    """Text like '$1,200', '45%' or ' 3.5 ' as numbers (NaN where it isn't one)."""
    return pd.to_numeric(v.str.replace(r"[\s$€£,%]", "", regex=True), errors="coerce")


def _plain(x) -> str:
    """A number as Python code: 306600.0 -> '306600', 2.5 -> '2.5'. For
    conditions the model copies, so no thousands separators."""
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(round(x, 6))


def _set_apart(v: pd.Series, extreme: pd.Series) -> pd.Index:
    """The values far from typical that ALSO stand far apart from every other
    value: likely errors, not just unusual. A typo'd salary of 1,250,000 is
    4x further from the median than the next salary; real large values come
    in a spread, with no such gap.

    Rule: sort by distance from the median; a value is set apart if there is
    a gap of at least 3x (in that distance) between it and the next value
    down, and every value above the gap is itself far from typical. The
    LARGEST such group is taken, so two bad values of different sizes
    (1,000,000 and 100,000 among hundreds) are both caught.

    Trade-off: a heuristic. An error only 2x too big has no 3x gap and is
    called plausible, so the wording says "likely" and the user decides."""
    d = (v - v.median()).abs().sort_values(ascending=False, kind="stable")   # ties: first row first
    apart = 0
    for k in range(1, min(len(extreme), len(d) - 1) + 1):
        if d.iloc[k - 1] > 0 and d.iloc[k - 1] >= 3 * d.iloc[k]:
            apart = k
    return d.index[:apart]


def _outlier_issue(df: pd.DataFrame, col: str, v: pd.Series, extreme: pd.Series,
                   text: bool = False) -> str:
    """The checklist line for values far from typical, with a verdict on
    which look like errors, so the model repeats a tested judgement instead
    of making (or dodging) its own.

    The verdict comes FIRST ("salary: likely an error: ..."). In the eval,
    gemma copied the start of each line and cut the end off, which is where
    the verdict used to be. Anything a model must repeat goes up front.

    text=True: the column is still text, so the condition only works once
    to_numbers has run; the suggestion says so."""
    apart = _set_apart(v, extreme)
    n_rest = len(extreme) - len(apart)
    if not len(apart):
        first = (extreme - v.median()).abs().idxmax()
        return (f"{col}: plausible: {_plural(len(extreme), 'value')} far from typical, most extreme "
                f"{_fmt(v[first])} ({_row_label(df, first)}), but none stands apart from the "
                f"rest, so they look like real values (suggest: keep them)")
    rest = v.drop(apart)
    values = sorted(set(v[apart].tolist()), key=lambda x: -abs(x - v.median()))
    first = (v[apart] - v.median()).abs().idxmax()
    shown = ", ".join(_fmt(x) for x in values[:3]) + (" ..." if len(values) > 3 else "")
    shown += f"; {'most extreme ' if len(apart) > 1 else ''}at {_row_label(df, first)}"
    parts = []
    if (v[apart] > v.median()).any():
        parts.append(f"{col} > {_plain(rest.max())}")
    if (v[apart] < v.median()).any():
        parts.append(f"{col} < {_plain(rest.min())}")
    condition = " or ".join(parts)
    # A column with missing values gives <NA> for them in "hours > 42" (after
    # to_numbers it's a nullable Int64), and the fix helpers refuse a
    # selection with <NA> in it. Hand over a condition that works first time:
    # a missing value is never one of the errors. Text columns get it too:
    # their placeholders and non-numbers become missing when converted.
    if text or df[col].isna().any():
        condition = f"({condition}).fillna(False)"
    verdict = "likely an error" if len(apart) == 1 else "likely errors"
    others = (f"; the other {_plural(n_rest, 'value')} far from typical look plausible"
              if n_rest else "")
    after = "after to_numbers, " if text else ""
    return (f"{col}: {verdict}: {_plural(len(apart), 'value')} ({shown}) far beyond every other "
            f"value{others} (suggest: {after}fix only the likely {'error' if len(apart) == 1 else 'errors'}, "
            f"selecting {'it' if len(apart) == 1 else 'them'} with the condition {condition}; ask whether to drop the rows or "
            f"set the value to missing{'; keep the plausible ones' if n_rest else ''})")

def _check_numbers(df: pd.DataFrame, col: str, v: pd.Series, out: list[str], issues: list[str],
                   text: bool = False) -> None:
    """Range, negatives and values far from typical for one number column.
    v: its non-missing values as numbers. text=True: the column is still
    text ('$1,200'), and v is what to_numbers() will make of it; the lines
    say so, because the conditions they give only work after converting."""
    # Interquartile range (IQR) rule: the middle half of the data lies
    # between the quartiles q1 and q3. Values more than 3 IQRs beyond
    # them are flagged. Unlike a standard-deviation rule, the outliers
    # being searched for can't drag the threshold towards themselves.
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 3 * iqr, q3 + 3 * iqr
    extreme = v[(v < low) | (v > high)]
    negatives = int((v < 0).sum())
    as_what = f"{col} (as numbers)" if text else col
    out.append(f"{'  ' if text else ''}{as_what}: min {_fmt(v.min())} | median {_fmt(v.median())} | "
               f"max {_fmt(v.max())} | negative {negatives} | zero {int((v == 0).sum())}")
    if negatives:
        issues.append(f"{col}: {_plural(negatives, 'negative value')}")
    if len(extreme):
        # Show the most extreme few, furthest from the median first.
        worst = (extreme - v.median()).abs().sort_values(ascending=False).index[:3]
        examples = ", ".join(f"{_fmt(v[i])} ({_row_label(df, i)})" for i in worst)
        # Say when the list is incomplete: a model given "most extreme:
        # A, B, C" for 4 values fixed those 3 by ID and missed the 4th.
        shown = (f"most extreme {len(worst)} of {len(extreme):,}"
                 if len(worst) < len(extreme) else "all of them")
        out.append(f"  far from typical (outside {_fmt(low)} to {_fmt(high)}): "
                   f"{len(extreme):,} values; {shown}: {examples}")
        issues.append(_outlier_issue(df, col, v, extreme, text=text))


def _issues_block(issues: list[str]) -> str:
    """The numbered checklist that ends every report. The model copies these
    lines into its summary, which is easier than collecting them from the
    report above (in testing it reliably dropped one or two)."""
    if not issues:
        return "Issues to report: none"
    lines = [f"Issues to report ({len(issues)}):"]
    lines += [f"  {i}. {issue}" for i, issue in enumerate(issues, start=1)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The helpers the model calls
# ---------------------------------------------------------------------------

def profile(df: pd.DataFrame, name: str | None = None) -> None:
    """Print a compact data-quality report for one table, ending with a
    numbered list of every issue found. The name defaults to the one load()
    gave the table, so profile(orders) is headed "orders", not "table"."""
    out, issues = _profile(df, name or df.attrs.get("table") or "table")
    print("\n".join(out + [_issues_block(issues)]))


def issues_in(df: pd.DataFrame, name: str = "table") -> list[str]:
    """The issue lines alone, for the cleaning report's "still in the data"."""
    return _profile(df, name)[1]


def _profile(df: pd.DataFrame, name: str) -> tuple[list[str], list[str]]:
    """profile()'s work: (report lines, issue lines)."""
    out = [f"=== {name}: {len(df):,} rows x {df.shape[1]} columns ==="]
    issues = []

    spaced = [c for c in df.columns if isinstance(c, str) and c != c.strip()]
    if spaced:
        out.append(f"! column names with stray spaces: {_examples(spaced, 10)}")
        issues.append(f"column names with stray spaces: {_examples(spaced, 10)} "
                      f"(df[{spaced[0]!r}] needs the spaces to work)")

    dups = int(df.duplicated().sum())
    out.append(f"Exact duplicate rows: {dups}")
    if dups:
        issues.append(f"{_plural(dups, 'exact duplicate row')} (fix: drop_duplicates)")

    # One line per column: the overview.
    out.append("Columns (dtype | missing | unique):")
    width = max(len(c) for c in df.columns)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        out.append(f"  {col:<{width}}  {str(s.dtype):<8} {missing:>5} missing "
                   f"{s.nunique():>7,} unique")
        if missing:
            # The default is to leave them: any fill value is a guess about
            # data nobody recorded, so it's the user's call, not the model's.
            issues.append(f"{col}: {_plural(missing, 'missing value')} (suggest: leave missing, "
                          f"unless the user gives a fill value; then fill_missing)")

    # The table's own key: its FIRST ID column. In expenses that's expense_id
    # (one row per claim); employee_id comes after it and links to another
    # table, so it SHOULD repeat (many claims per person).
    key = next((c for c in df.columns if _is_identifier(c)), None)

    for col in df.columns:
        s = df[col]

        if _is_identifier(col):
            repeated = int(s.dropna().duplicated().sum())
            kind = "every value unique" if repeated == 0 else \
                f"{_plural(repeated, 'row')} repeat a value seen earlier in this column"
            out.append(f"{col} [identifier]: {kind}")
            if col == key:
                issue = _repeated_key(df, col)
                if issue:
                    out.append(f"  ! {issue.split(' (suggest:')[0]}")
                    issues.append(issue)

        elif _is_number(s):
            v = s.dropna()
            if v.empty:
                out.append(f"{col}: all values missing")
                continue
            _check_numbers(df, col, v, out, issues)

        elif types.is_datetime64_any_dtype(s):
            out.append(f"{col}: dates")
            _check_date_range(df, col, s, out, issues)

        elif _is_text(s):
            v = s.dropna()
            placeholder = _is_placeholder(v)
            if placeholder.any():
                found = v[placeholder].value_counts()
                listed = ", ".join(f"{k!r} {n:,}" for k, n in found.items())
                n = int(placeholder.sum())
                issues.append(f"{col}: {_plural(n, 'placeholder value')} that probably "
                              f"{'means' if n == 1 else 'mean'} missing ({listed}); pandas counts "
                              f"{'it' if n == 1 else 'them'} as text, not NaN "
                              f"(fix: placeholders_to_missing)")
            v = v[~placeholder]   # every check below looks at real values only

            spaced = v[(v != v.str.strip()) | v.str.contains("  ", regex=False)]
            if len(spaced):
                issues.append(f"{col}: {_plural(len(spaced), 'value')} with stray spaces "
                              f"(e.g. {_examples(spaced)}) (fix: strip_spaces)")

            layout = _layout_of(v)
            numbers = _as_numbers(v)
            looks_like_dates = len(v) and (layout.notna().mean() >= 0.8 or
                                           ("date" in col.lower() and layout.notna().mean() >= 0.5))
            # Leading zeros ('02134', '007') mean codes such as ZIP codes or phone
            # numbers: text on purpose, and converting would destroy the zeros.
            looks_like_numbers = (len(v) and not looks_like_dates and
                                  numbers.notna().mean() >= 0.8 and
                                  not v.str.strip().str.match(r"0\d").any())

            counts = s.value_counts(dropna=False)
            if looks_like_dates:
                _check_text_dates(df, col, v, layout, out, issues)
            elif len(counts) <= MAX_LISTED_VALUES:
                listed = ", ".join(f"{MISSING if pd.isna(k) else repr(k)} {n:,}" for k, n in counts.items())
                out.append(f"{col}: {listed}")
            else:
                out.append(f"{col}: {s.nunique():,} distinct values, e.g. {_examples(v)}")

            # Numbers with a unit ('12.5 kg', '30 min'): one unit across the
            # column. No helper strips units on its own, since the unit is part
            # of the meaning (kg or lb?): the user approves a custom_fix.
            units = v.str.extract(r"^\s*-?[\d.,]+\s*([A-Za-z°%]+)\s*$")[0]
            if (not looks_like_dates and not looks_like_numbers and len(v)
                    and units.notna().mean() >= 0.8 and units.nunique() == 1):
                unit = units.dropna().iloc[0]
                line = (f"{col}: numbers stored with a unit: {len(units.dropna()):,} of {len(v):,} "
                        f"values like {_examples(v[units.notna()], 2)} (unit {unit!r})")
                out.append(f"  ! {line}")
                issues.append(line + f"; no helper removes units: propose a custom_fix that "
                                     f"removes {unit!r}, then to_numbers")

            if looks_like_numbers:
                # Show the formatted ones first ('$1,200'): they explain why pandas
                # read the column as text.
                text = v[numbers.notna() & v.str.contains(r"[^\d.\-]")]
                other = v[numbers.isna()]
                line = (f"{col}: numbers stored as text: {len(numbers.dropna()):,} of {len(v):,} values "
                        f"are numbers (e.g. {_examples(text if len(text) else v)})")
                if len(other):
                    line += f"; not numbers: {_plural(len(other), 'value')} (e.g. {_examples(other)})"
                out.append(f"  ! {line}")
                issues.append(line + " (fix: to_numbers)")
                # Check the values as the numbers they will become. Otherwise a
                # typo like '$75,000' among '$750' day rates stays hidden until
                # after to_numbers, when the fixes are already approved and the
                # model is about to save.
                as_numbers = numbers.dropna()
                if len(as_numbers):
                    _check_numbers(df, col, as_numbers, out, issues, text=True)

            if not looks_like_dates and not looks_like_numbers and len(counts) <= MAX_LISTED_VALUES:
                for short, full, n_short, n_full in _abbreviations(v):
                    line = (f"{col}: possible abbreviation: {short!r} {n_short:,} may mean "
                            f"{full!r} {n_full:,}")
                    out.append(f"  ! {line}")
                    issues.append(line + f"; ask the user (suggest: if they agree, a custom_fix "
                                         f"that replaces {short!r} with {full!r})")

            for spellings in _variant_groups(s, spaces_only_too=False):
                variants = " / ".join(f"{k!r} {n:,}" for k, n in spellings.items())
                # The count a merge changes, ready to copy: given "'Cloud Ops' 6 /
                # 'cloud ops' 1 / 'CLOUD OPS' 1", gemma proposed merging "3 rows"
                # (it added the wrong numbers). merge_variants keeps the most
                # common spelling, which _variant_groups lists first.
                changed = int(spellings.iloc[1:].sum())
                merge = f"merging changes {_plural(changed, 'value')} to {spellings.index[0]!r}"
                out.append(f"  ! spelling variants in {col}: {variants}; {merge}")
                issues.append(f"{col}: spelling variants {variants}; {merge} (fix: merge_variants)")

    return out, issues


def compare_keys(left: pd.DataFrame, right: pd.DataFrame, on: str,
                 left_name: str = "left", right_name: str = "right") -> None:
    """Print how two tables link through a shared ID column, before merging."""
    lk, rk = left[on], right[on]
    out = [f"=== {left_name} vs {right_name} on '{on}' ==="]
    issues = []

    # "One row per ID" on one side and "several" on the other means a merge
    # copies each employee's details onto every one of their expense rows.
    unique = {}
    for name, k in ((left_name, lk), (right_name, rk)):
        repeats = len(k.dropna()) - k.nunique()
        unique[name] = repeats == 0
        shape = "one row per ID" if repeats == 0 else "several rows per ID"
        missing = int(k.isna().sum())
        out.append(f"{name}: {_plural(k.nunique(), 'distinct ID')}, {shape}, {missing} missing")
        if missing:
            issues.append(f"{name}: {_plural(missing, 'row')} with no {on}")

    # IDs found on one side only: the rows that a merge would drop or leave
    # without a match. Both the number of IDs AND the number of rows matter.
    for name, k, other_name, other in ((left_name, lk, right_name, rk),
                                       (right_name, rk, left_name, lk)):
        only = k[k.notna() & ~k.isin(other)]
        ids = sorted(only.unique())
        if ids:
            # str(), not _fmt(): IDs are labels, so no thousands separators.
            shown = ", ".join(str(i) for i in ids[:10]) + (" ..." if len(ids) > 10 else "")
            line = (f"{_plural(len(ids), 'ID')} in {_plural(len(only), 'row')} "
                    f"(IDs: {shown})")
            # Which direction is a problem? When one table lists each ID once
            # (customers) and the other refers to them many times (orders),
            # a customer with no orders is normal, but an order whose
            # customer doesn't exist points at nothing. Only that direction
            # is an issue. Between two one-row-per-ID tables, both are.
            parent_only = unique[name] and not unique[other_name]
            if parent_only:
                out.append(f"In {name} but not {other_name}: {line} "
                           f"(normal: not every {name} row has {other_name} rows)")
            else:
                out.append(f"In {name} but not {other_name}: {line}")
                # A ready-made condition, as for outliers: without one, the
                # model reported these rows but proposed no fix at all.
                issues.append(f"{name}: {line} not in {other_name}; these rows point at nothing "
                              f"(suggest: ask whether to drop these rows; to drop them, drop_rows with "
                              f"the condition ~{name}[{on!r}].isin({other_name}[{on!r}]))")
        else:
            out.append(f"In {name} but not {other_name}: none")

    out.append(_issues_block(issues))
    print("\n".join(out))


def check(*tables: pd.DataFrame) -> None:
    """The whole check in one call: profile() every table, then compare_keys()
    for every pair of tables that share an ID column.

        check(orders, customers)

    Tables from load() carry their names, so none need passing. In testing, a
    model planned profile() and compare_keys() as separate steps, ran only the
    first, and never saw that one table had rows for an ID the other lacked.
    One call means no step can be skipped.

    check() with no tables checks every table loaded so far. In testing, a
    model loaded three files and passed only two, so a table is never left
    out silently: any loaded table not passed is named at the end."""
    import cleaning   # here, not at the top: cleaning.py imports this module
    loaded = {name: versions[-1] for name, versions in cleaning._HISTORY.items()}
    if not tables:
        if not loaded:
            raise ValueError("No tables loaded yet. Load each file with load(), then call check().")
        tables = tuple(loaded.values())
    names = [t.attrs.get("table") or f"table {i}" for i, t in enumerate(tables, start=1)]
    for table, name in zip(tables, names):
        profile(table, name)
        print()
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            shared = [c for c in tables[i].columns if _is_identifier(c) and c in tables[j].columns]
            if not shared:
                print(f"=== {names[i]} vs {names[j]}: no shared ID column, so not compared ===\n")
            for col in shared:
                compare_keys(tables[i], tables[j], col, names[i], names[j])
                print()
    left_out = [name for name in loaded if name not in names]
    if left_out:
        print(f"WARNING: loaded but not checked: {', '.join(left_out)}. "
              f"Call check() with every table, or check() with no arguments for all of them.")