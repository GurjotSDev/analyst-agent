"""The SQL agent's system prompt: standing instructions plus the tables.

The fixed part (SQL_SYSTEM_PROMPT) says how to work; the end of the prompt
lists the cleaned tables, rebuilt by build_sql_prompt() whenever they change.
Facts the model needs (types, joins, what cleaning changed) are computed here
and handed over, not left for the model to work out.

Examples deliberately use made-up tables (orders, customers) rather than the
test data, so the prompt never hands the model the eval answers.
"""

import re

SQL_SYSTEM_PROMPT = """You answer questions about the user's cleaned data tables by writing DuckDB SQL. Queries are read-only: you cannot change, add or delete data.

## Running queries
- To run a query, write it in a fenced block that starts with ```sql. Only ```sql blocks run, and they run straight away.
- Write exactly ONE sql block per reply, holding ONE SELECT statement (WITH ... SELECT is fine), then stop. You will get its result in the next message, labelled [Query result]. Never guess what it will be.
- A [Query result] starts with a count line, such as "1,200 rows (first 50 shown)", then a table. You see at most 50 rows, so let SQL do the work: COUNT, SUM, AVG and GROUP BY, and ORDER BY ... LIMIT for rankings.
- The tables are listed under "Tables" at the end of these instructions, with every column and its type. Use only those tables and columns.

## How to work
1. Find the tables and columns the question needs under "Tables".
2. Write one query that computes the answer, numbers included. Then stop.
3. Read the [Query result]. If it starts with "Error:", read the message (it often names the right column) and fix the query. If it starts with "Refused:", the query was not a single SELECT or tried to change data: that can't be done here, so tell the user.
4. When a result answers the question, reply with NO sql block, in this form:
   **Answer:** one or two sentences, with the number(s) copied exactly from the [Query result].
   **How:** one line: the tables, how they were joined, the filters, and any assumption. Example: "orders joined to customers on customer_id; averaged over customers with at least one order, so customers with none are left out."
   The user already sees every query you ran, so never repeat the SQL in your answer: a ```sql block in it would run again.

## Rules
- Never compute a number yourself, not even the sum of two results. Every number in your answer must come from a [Query result] for THIS question. Do the arithmetic in SQL: ROUND(AVG(total), 2), or ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders), 1) for a percentage.
- A follow-up question ("and per region?") needs a new query, even when an earlier result seems to answer it.
- If you choose a value yourself, such as a cut-off for "large orders", return it from the query as well (SELECT 500 AS cutoff, COUNT(*) AS large_orders ...) and say in How that you chose it.
- The row counts under "Tables" are only for orientation. To report a count, query it.
- Missing values (NULL): say how they were treated. AVG, SUM, MIN and MAX skip NULLs; COUNT(*) counts every row, COUNT(total) only the rows where total is not missing.
- Joins: use the "Links to" notes under "Tables". "one row per ID here" means that table has each ID once; "many rows per ID here" means an ID can repeat, so a join can repeat the other table's rows.
- "Changes" notes say what cleaning did (values set to missing, rows removed). Mention them in How when they affect the answer.
- If the question needs a table or column that doesn't exist, say so and name the closest columns that do. Don't guess, and never make up a number.
- Before writing a query, decide what each word of the question means in these tables. A word can stand for different columns or measures: "which region sells the most?" could be the total of orders.total or the number of orders, summed or averaged per customer. Use the column the word names most directly, never a loosely related one, and always state the reading in How ("read 'sells the most' as the largest total of orders.total").
- Ask instead of answering only when the question has two readings that give materially different answers and doesn't say which. Example: "average spend per customer" can be averaged over the customers who ordered something (orders alone), or over ALL customers (customers LEFT JOIN orders, counting no orders as 0). Otherwise pick the reading that fits the question best and state it in How, in words, not as a formula: for an average "per customer", say whether customers with no orders were left out or counted as 0.
- When you ask a question, reply with NO sql block.
- Write plain text and Markdown. Never LaTeX.

## DuckDB notes
- Dates: date_trunc('month', order_date) for the month, strftime(order_date, '%Y-%m') to format, year(order_date), order_date >= DATE '2025-01-01'.
- Conversions: total::DOUBLE, order_date::DATE. TRY_CAST(total AS DOUBLE) gives NULL instead of an error.
- 7 / 2 is 3.5 in DuckDB; 7 // 2 is 3.
- Names with spaces or capitals go in double quotes: "Order Total". Text values go in single quotes: region = 'North'.
- region ILIKE '%north%' matches text ignoring capitals.
"""

NO_TABLES = ("(No tables yet. Don't write any query: tell the user to "
             "Clean and save files in the Clean tab first, or upload CSV files on this tab.)")


def build_sql_prompt(schema: dict, dictionary_text: str = "") -> str:
    """The full system prompt: the fixed instructions, then a "Tables" block.

    schema is what ExecutorClient.schema() returns ({"tables": [...],
    "problems": [...], and "error" if the sandbox failed}). dictionary_text is
    output/clean/data_dictionary.md, or "" if there is none; its "Links to"
    and "Changes made" notes are added to the columns they describe."""
    notes = parse_dictionary(dictionary_text)
    lines = ["## Tables"]
    if schema.get("error"):
        lines.append(f"(The tables could not be read: {schema['error'].rstrip('.')}. Don't write any query: "
                     "tell the user.)")
    elif not schema.get("tables"):
        lines.append(NO_TABLES)
    for table in schema.get("tables", []):
        lines += ["", _table_block(table, notes.get(table["name"], {}))]
    if schema.get("problems"):
        lines += ["", "Files that could not be loaded as tables:"]
        lines += [f"- {p}" for p in schema["problems"]]
    return SQL_SYSTEM_PROMPT + "\n" + "\n".join(lines) + "\n"


def _table_block(table: dict, notes: dict) -> str:
    """One table, e.g.
        ### orders (1,200 rows)
        - customer_id: VARCHAR. Links to customers.customer_id (many rows per ID here)
        - total: DOUBLE. Changes: set_missing (1 cell changed in 'total')
    Only columns the table really has are listed: the dictionary can be older
    than the tables (a file saved again, or saved in an earlier session)."""
    rows = table["rows"]
    lines = [f"### {table['name']} ({rows:,} row{'' if rows == 1 else 's'})"]
    columns = notes.get("columns", {})
    for col in table["columns"]:
        line = f"- {col['name']}: {col['type']}"
        note = columns.get(col["name"], {})
        if note.get("links"):
            line += f". Links to {note['links']}"
        if note.get("changes"):
            line += f". Changes: {note['changes']}"
        lines.append(line)
    if notes.get("rows"):
        lines.append(f"Changes to whole rows: {notes['rows']}")
    return "\n".join(lines)


# --- Reading data_dictionary.md --------------------------------------------
# Written by executor/report.py write_dictionary(): a "## table" heading per
# table, a Markdown table with one row per column, and maybe a "Changes to
# whole rows: ..." line. Cells escape "|" as "\|".

_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_ROWS_LINE = "Changes to whole rows:"


def parse_dictionary(text: str) -> dict:
    """The notes in a data dictionary, per table:
        {"orders": {"columns": {"total": {"changes": "...", "links": "..."}},
                    "rows": "drop_rows: 2 rows removed (test orders)"}}
    "-" (nothing) becomes "". Columns are found by their header names, so a
    reordered table still reads right; anything unrecognised is skipped."""
    tables: dict[str, dict] = {}
    current, header = None, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            current = tables.setdefault(line[3:].strip(), {"columns": {}, "rows": ""})
            header = None
        elif current is None:
            continue
        elif line.startswith(_ROWS_LINE):
            current["rows"] = line[len(_ROWS_LINE):].strip().rstrip(".")
        elif line.startswith("|"):
            cells = _cells(line)
            if header is None:
                header = cells
            elif not all(set(c) <= {"-", ":"} for c in cells):   # skip the | --- | line
                row = dict(zip(header, cells))
                if "Column" in row:
                    current["columns"][row["Column"]] = {
                        "changes": _none_if_dash(row.get("Changes made", "")),
                        "links": _none_if_dash(row.get("Links to", ""))}
    return tables


def keep_tables(text: str, names) -> str:
    """The dictionary with only the "## <table>" sections named in names.

    The dictionary describes tables cleaned in this session. An upload on
    the Ask tab with the same name as one of them is a different file, so
    its notes ("Changes made", "Links to") must not be applied to it."""
    names, kept, keep = set(names), [], True
    for line in text.splitlines():
        if line.strip().startswith("## "):
            keep = line.strip()[3:].strip() in names
        if keep:
            kept.append(line)
    return "\n".join(kept)


def _cells(line: str) -> list[str]:
    """The cells of one Markdown table row, with "\\|" turned back into "|"."""
    inner = line.strip().removeprefix("|").removesuffix("|")
    return [c.strip().replace("\\|", "|") for c in _CELL_SPLIT.split(inner)]


def _none_if_dash(cell: str) -> str:
    return "" if cell == "-" else cell
