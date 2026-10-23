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
    orders["region"] = merge_variants(orders["region"])
                                           fix spelling variants, report changes
"""

import pandas as pd
from pandas.api import types

MAX_LISTED_VALUES = 15   # text columns with at most this many values get full counts
MISSING = "NaN (missing)"  # how missing values are shown. Not "<missing>": the model
                           # mistook that for a real text value and searched for it.

# A fixed order of 8 distinct colours, checked for colour-blind separation.
# Series get them in this order, so the first series is always blue.
CHART_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"]


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


def _variant_groups(s: pd.Series) -> list[pd.Series]:
    """Groups of spellings that are the same after ignoring capitalisation and
    surrounding spaces, e.g. 'Meals' / 'meals ' / ' MEALS'. Each group is a
    value_counts() Series, most common spelling first. Only groups with more
    than one spelling are returned."""
    v = s.dropna()
    groups = []
    for _, group in v.groupby(v.str.strip().str.lower()):
        spellings = group.value_counts()
        if len(spellings) > 1:
            groups.append(spellings)
    return groups


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

def profile(df: pd.DataFrame, name: str = "table") -> None:
    """Print a compact data-quality report for one table, ending with a
    numbered list of every issue found."""
    out = [f"=== {name}: {len(df):,} rows x {df.shape[1]} columns ==="]
    issues = []

    dups = int(df.duplicated().sum())
    out.append(f"Exact duplicate rows: {dups}")
    if dups:
        issues.append(f"{_plural(dups, 'exact duplicate row')} (remove with df.drop_duplicates())")

    # One line per column: the overview.
    out.append("Columns (dtype | missing | unique):")
    width = max(len(c) for c in df.columns)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        out.append(f"  {col:<{width}}  {str(s.dtype):<8} {missing:>5} missing "
                   f"{s.nunique():>7,} unique")
        if missing:
            issues.append(f"{col}: {_plural(missing, 'missing value')}")

    for col in df.columns:
        s = df[col]

        if _is_identifier(col):
            # Whether repeats are fine depends on the table: in expenses,
            # employee_id SHOULD repeat (many claims per person); expense_id
            # should not. The model decides; we just report the count.
            repeated = int(s.dropna().duplicated().sum())
            kind = "every value unique" if repeated == 0 else \
                f"{_plural(repeated, 'row')} repeat a value seen earlier in this column"
            out.append(f"{col} [identifier]: {kind}")

        elif _is_number(s):
            v = s.dropna()
            if v.empty:
                out.append(f"{col}: all values missing")
                continue
            # Interquartile range (IQR) rule: the middle half of the data lies
            # between the quartiles q1 and q3. Values more than 3 IQRs beyond
            # them are flagged. Unlike a standard-deviation rule, the outliers
            # being searched for can't drag the threshold towards themselves.
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 3 * iqr, q3 + 3 * iqr
            extreme = v[(v < low) | (v > high)]
            negatives = int((v < 0).sum())
            out.append(f"{col}: min {_fmt(v.min())} | median {_fmt(v.median())} | "
                       f"max {_fmt(v.max())} | negative {negatives} | zero {int((v == 0).sum())}")
            if negatives:
                issues.append(f"{col}: {_plural(negatives, 'negative value')}")
            if len(extreme):
                # Show the most extreme few, furthest from the median first.
                worst = (extreme - v.median()).abs().sort_values(ascending=False).index[:3]
                examples = ", ".join(f"{_fmt(df.at[i, col])} ({_row_label(df, i)})" for i in worst)
                out.append(f"  far from typical (outside {_fmt(low)} to {_fmt(high)}): "
                           f"{len(extreme):,} values; most extreme: {examples}")
                # Only the single most extreme value here; the others are listed above.
                first = worst[0]
                issues.append(f"{col}: {_plural(len(extreme), 'value')} far from typical, most extreme "
                              f"{_fmt(df.at[first, col])} ({_row_label(df, first)}); say which look "
                              f"like errors and which look plausible")

        elif _is_text(s):
            counts = s.value_counts(dropna=False)
            if len(counts) <= MAX_LISTED_VALUES:
                listed = ", ".join(f"{MISSING if pd.isna(k) else k} {n:,}" for k, n in counts.items())
                out.append(f"{col}: {listed}")
            elif "date" in col.lower():
                dates = pd.to_datetime(s, errors="coerce")
                bad = int(dates.isna().sum() - s.isna().sum())
                out.append(f"{col}: dates {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, "
                           f"{bad} unparseable")
                if bad:
                    issues.append(f"{col}: {_plural(bad, 'value')} that are not valid dates")
            else:
                sample = ", ".join(repr(x) for x in s.dropna().unique()[:3])
                out.append(f"{col}: {s.nunique():,} distinct values, e.g. {sample}")

            for spellings in _variant_groups(s):
                variants = " / ".join(f"{k!r} {n:,}" for k, n in spellings.items())
                out.append(f"  ! spelling variants in {col}: {variants}")
                issues.append(f"{col}: spelling variants {variants} (fix: merge_variants)")

    out.append(_issues_block(issues))
    print("\n".join(out))


def compare_keys(left: pd.DataFrame, right: pd.DataFrame, on: str,
                 left_name: str = "left", right_name: str = "right") -> None:
    """Print how two tables link through a shared ID column, before merging."""
    lk, rk = left[on], right[on]
    out = [f"=== {left_name} vs {right_name} on '{on}' ==="]
    issues = []

    # "One row per ID" on one side and "several" on the other means a merge
    # copies each employee's details onto every one of their expense rows.
    for name, k in ((left_name, lk), (right_name, rk)):
        repeats = len(k.dropna()) - k.nunique()
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
            out.append(f"In {name} but not {other_name}: {line}")
            issues.append(f"{name}: {line} not in {other_name}; a merge would drop these rows")
        else:
            out.append(f"In {name} but not {other_name}: none")

    out.append(_issues_block(issues))
    print("\n".join(out))


def merge_variants(s: pd.Series) -> pd.Series:
    """Fix spelling variants in a text column and print exactly what changed.

    Each group of spellings that differ only in capitalisation or surrounding
    spaces is mapped to the spelling MOST rows already use. Nothing else is
    touched: 'HR' stays 'HR', because it has no variants. Returns a new
    Series, so use it as: df["col"] = merge_variants(df["col"])
    """
    mapping = {}
    changes = []
    for spellings in _variant_groups(s):
        target = spellings.index[0]   # value_counts() puts the most common first
        for spelling, n in spellings.iloc[1:].items():
            mapping[spelling] = target
            changes.append(f"  {spelling!r} -> {target!r}: {_plural(n, 'row')}")

    label = s.name if s.name is not None else "column"
    if changes:
        total = sum(int(s.isin([k]).sum()) for k in mapping)
        print(f"merge_variants({label}): changed {_plural(total, 'row')}\n" + "\n".join(changes))
    else:
        print(f"merge_variants({label}): no spelling variants, nothing changed")
    # .replace() with a dict swaps only the exact values listed in it.
    return s.replace(mapping)


# ---------------------------------------------------------------------------
# Chart style
# ---------------------------------------------------------------------------

def apply_chart_style() -> None:
    """House style for every chart, applied once when the kernel starts.

    rcParams is matplotlib's global settings dictionary: every chart drawn
    afterwards uses these defaults unless the code overrides them. Setting
    them here means the model gets readable charts without having to
    remember any styling, and every chart in a report looks consistent.
    """
    import matplotlib.pyplot as plt
    from cycler import cycler   # matplotlib's tool for "use these colours in turn"

    ink, muted, rule = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),        # 16:9-ish, fits a report page
        "figure.dpi": 110,                 # sharper than the default
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.prop_cycle": cycler(color=CHART_COLORS),
        "axes.spines.top": False,          # no box around the plot:
        "axes.spines.right": False,        # just the two axes that carry values
        "axes.edgecolor": "#c9c8c3",
        "axes.grid": True,                 # light grid lines help read values...
        "axes.axisbelow": True,            # ...but sit behind the data
        "grid.color": rule,
        "grid.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",      # titles read like headlines
        "axes.titlecolor": ink,
        "axes.labelsize": 11,
        "axes.labelcolor": muted,
        "xtick.color": muted,
        "ytick.color": muted,
        "text.color": ink,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 10.5,
    })