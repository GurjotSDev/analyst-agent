"""
The cleaning report (PDF) and the data dictionary (Markdown).

Both are written by code from the change log and the tables themselves,
never by the model, so they can't contain an invented number. save_clean()
rewrites them after every save, so they always match the files on disk.

    output/clean/cleaning_report.pdf   what was found, changed and left alone
    output/clean/data_dictionary.md    every column of every saved table

The PDF uses fpdf2's built-in fonts, which only cover Latin-1, so other
characters (arrows, some symbols) are replaced before writing.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

INK = (11, 11, 11)
MUTED = (82, 81, 78)
RULE = (201, 200, 195)
HEAD_FILL = (240, 239, 236)
MAX_IDS = 10         # removed-row IDs shown per change (the count is always exact)
MAX_CELL = 60        # characters per table cell before it's shortened
HEADING = FontFace(emphasis="BOLD", fill_color=HEAD_FILL)   # table header row style


def _latin(text) -> str:
    """Text the built-in PDF fonts can draw: common Unicode first, then
    anything else outside Latin-1 becomes '?'."""
    text = "" if text is None else str(text)
    for a, b in (("→", "->"), ("–", "-"), ("—", "-"), ("…", "..."), ("‘", "'"), ("’", "'"),
                 ("“", '"'), ("”", '"'), ("€", "EUR ")):
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def _short(value, limit: int = MAX_CELL) -> str:
    if isinstance(value, pd.Timestamp):   # 2025-06-27, not 2025-06-27 00:00:00
        text = value.strftime("%Y-%m-%d %H:%M:%S" if value != value.normalize() else "%Y-%m-%d")
    else:
        text = "(missing)" if value is None else repr(value) if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."


class _Report(FPDF):
    def footer(self) -> None:
        # fpdf2 calls this at the bottom of every page. {nb} becomes the page count.
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"Data cleaning report - page {self.page_no()} of {{nb}}", align="C")

    def heading(self, text: str, size: int = 13) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*INK)
        self.multi_cell(0, 7, _latin(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def para(self, text: str, size: int = 10, muted: bool = False) -> None:
        self.set_font("Helvetica", size=size)
        self.set_text_color(*(MUTED if muted else INK))
        # align="L": multi_cell justifies by default, stretching short lines.
        self.multi_cell(0, 5, _latin(text), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def grid(self, header: list[str], rows: list[list], widths: list[int] | None = None) -> None:
        """A table with a shaded header row. fpdf2's table() wraps long cells
        and repeats the header if the table runs onto a new page."""
        self.set_font("Helvetica", size=9)
        self.set_text_color(*INK)
        self.set_draw_color(*RULE)
        with self.table(col_widths=widths, headings_style=HEADING, line_height=5,
                        text_align="LEFT", first_row_as_headings=True) as table:
            for row in [header] + rows:
                cells = table.row()
                for value in row:
                    cells.cell(_latin(value))
        self.ln(2)


def _for_people(issue: str) -> str:
    """An issue line without the hints meant for the model ('(fix: ...)',
    '(suggest: ...)', '; say if ...', '; ask ...'):
    'salary: 7 missing values (suggest: leave missing...)' -> 'salary: 7 missing values'."""
    issue = re.sub(r"\s*\((fix|suggest): .*\)$", "", issue)
    return re.sub(r"; (say if|ask) .*$", "", issue)


def write_report(path: Path, tables: list[dict], log: list[dict], remaining: dict[str, list[str]]) -> None:
    """tables: one dict per loaded table (table, source, rows_before, cols_before,
    rows_after, cols_after, changes, saved). remaining: issues still found in
    each table's last version."""
    pdf = _Report(orientation="portrait", format="A4")
    pdf.set_margins(16, 16, 16)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()

    pdf.heading("Data cleaning report", size=18)
    pdf.para(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Every change below was approved and logged "
             f"as it was made; this report is written from that log. The original files were not "
             f"changed: cleaned copies are in output/clean/.", muted=True)

    pdf.heading("Summary")
    pdf.grid(["Table", "Source file", "Rows before", "Rows after", "Changes", "Saved"],
             [[t["table"], Path(t["source"]).name, f"{t['rows_before']:,}", f"{t['rows_after']:,}",
               str(t["changes"]), f"clean/{t['table']}.csv" if t["saved"] else "not saved"]
              for t in tables],
             widths=[22, 34, 18, 18, 14, 30])

    pdf.heading("Still in the data")
    pdf.para("Issues the checks still find in each table's final version: problems the user chose "
             "to leave, and values that are fine as they are (for example, normal negative amounts).",
             muted=True)
    for t in tables:
        issues = remaining.get(t["table"], [])
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, _latin(t["table"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if not issues:
            pdf.para("  None")
        for issue in issues:
            pdf.para(f"  - {_for_people(issue)}")
        pdf.ln(1)

    pdf.heading(f"Changes ({len(log)})")
    if not log:
        pdf.para("No changes were made.")
    for e in log:
        column = f" '{e['column']}'" if e["column"] else ""
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 6, _latin(f"{e['n']}. {e['table']}: {e['action']}{column}"),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.para(f"Reason: {e['reason']}")
        pdf.para(f"What changed: {e['summary']} (rows {e['rows_before']:,} -> {e['rows_after']:,})")
        for col, change in e.get("type_changes", {}).items():
            pdf.para(f"Type of {col}: {change}")
        if e["rows_removed"]:
            ids = ", ".join(e["removed"][:MAX_IDS])
            more = f", and {e['rows_removed'] - MAX_IDS:,} more (all listed in change_log.json)" \
                if e["rows_removed"] > MAX_IDS else ""
            pdf.para(f"Rows removed: {ids}{more}")
        if e["examples"]:
            pdf.ln(1)
            pdf.grid(["Row", "Column", "Before", "After"],
                     [[x["row"], x["column"], _short(x["before"]), _short(x["after"])] for x in e["examples"]],
                     widths=[40, 30, 52, 52])
        pdf.ln(2)

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


# ---------------------------------------------------------------------------
# The data dictionary
# ---------------------------------------------------------------------------

def _kind(s: pd.Series) -> str:
    """A column's type in plain words."""
    if pd.api.types.is_bool_dtype(s):
        return "true/false"
    if pd.api.types.is_integer_dtype(s):
        return "whole number"
    if pd.api.types.is_float_dtype(s):
        return "decimal number"
    if pd.api.types.is_datetime64_any_dtype(s):
        has_time = (s.dropna().dt.normalize() != s.dropna()).any()
        return "date and time" if has_time else "date"
    # Dates still stored as text matter to anyone querying the table: they
    # sort and compare as text, not as dates.
    from analyst_helpers import _layout_of
    values = s.dropna().astype(str)
    if len(values) and _layout_of(values).notna().mean() >= 0.8:
        return "text (holds dates)"
    return "text"


def _cell(text) -> str:
    """Markdown table cells can't contain '|' or line breaks."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _touches(e: dict, col: str) -> bool:
    """Did this log entry change cells in col? A helper names its column;
    a custom_fix has none, so its per-column cell counts decide."""
    return e["column"] == col or (not e["column"] and col in e.get("cells_by_column", {}))


def _change_to(e: dict, col: str) -> str:
    """The entry as a column's "Changes made" cell. A custom_fix may change
    several columns, so its count is the one for THIS column, and its reason
    is shown because 'custom_fix' alone says nothing."""
    if e["column"] == col:
        return f"{e['action']} ({e['summary']})"
    n = e["cells_by_column"][col]
    return f"{e['action']}: {e['reason']} ({n:,} cell{'' if n == 1 else 's'} changed)"


def _changes_rows(e: dict) -> bool:
    """Does a column-less entry belong under "Changes to whole rows"? Yes if it
    removed or added rows, or changed no cells (e.g. it added a column), so
    no column row would show it. A custom_fix that only edited cells is
    listed under those columns instead. Entries without per-column counts
    also stay here."""
    return bool(e.get("rows_removed") or e.get("rows_added") or not e.get("cells_by_column"))


def write_dictionary(path: Path, tables: dict[str, pd.DataFrame], sources: dict[str, str],
                     log: list[dict], is_identifier) -> None:
    """One section per saved table: each column's type, missing count,
    distinct values, examples, what was done to it, and links to other
    tables through shared ID columns. Descriptions are left for the data's
    owner: code can't know what a column means."""
    lines = ["# Data dictionary", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} from the cleaned files in output/clean/. "
             "The Description column is left blank for the data's owner to fill in.", ""]
    for name, df in tables.items():
        lines += [f"## {name}", "",
                  f"File: clean/{name}.csv (from {Path(sources[name]).name}), "
                  f"{len(df):,} rows x {df.shape[1]} columns.", "",
                  "| Column | Type | Missing | Distinct | Examples | Changes made | Links to | Description |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for col in df.columns:
            s = df[col]
            examples = ", ".join(_short(v, 20) for v in s.dropna().unique()[:3]) or "-"
            changes = "; ".join(_change_to(e, col) for e in log
                                if e["table"] == name and _touches(e, col)) or "-"
            links = []
            if is_identifier(col):
                unique = not s.dropna().duplicated().any()
                for other, odf in tables.items():
                    if other != name and col in odf.columns:
                        role = "one row per ID" if unique else "many rows per ID"
                        links.append(f"{other}.{col} ({role} here)")
            lines.append("| " + " | ".join(_cell(x) for x in [
                col, _kind(s), f"{int(s.isna().sum()):,}", f"{s.nunique():,}", examples, changes,
                ", ".join(links) or "-", ""]) + " |")
        whole = [e for e in log if e["table"] == name and not e["column"] and _changes_rows(e)]
        if whole:
            lines += ["", "Changes to whole rows: " + "; ".join(f"{e['action']}: {e['summary']} "
                                                              f"({e['reason']})" for e in whole) + "."]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")