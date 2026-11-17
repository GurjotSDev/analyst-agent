"""
Scoring: does a reply state a fact correctly?

A "fact" is a number that must appear on the same line as one of its
keywords, e.g. "salary" and 6 (missing values). Same line, because models
report results in table rows and bullet points ("| Missing salaries | 6 |"),
and because a number anywhere in the reply would match by coincidence: "6"
appears all over a data-quality summary, but "6" next to "salary" is the
specific claim we're checking.
"""

import re
from dataclasses import dataclass

# A number as models write them: 12974, 12,974, $12,974.15, 1.25M, 306k.
#   (?<![\w.])      not glued to a word or decimal point before it
#   \$?             optional dollar sign
#   (\d{1,3}(?:,\d{3})+|\d+)   digits, with or without thousands separators
#   (?:\.(\d+))?    optional decimals
#   ([kKmM]|\s?million|\s?thousand)?   optional suffix: 306k, 1.25M, 1.25 million...
#   (?![A-Za-z\d])  ...and nothing alphabetic right after, so the "k" in
#                   "5 key" or the "m" in "12 meals" isn't read as a suffix
NUMBER = re.compile(r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
                    r"([kKmM]|\s?million|\s?thousand)?(?![A-Za-z\d])")
SCALE = {"k": 1_000, "t": 1_000, "m": 1_000_000}   # keyed by the suffix's first letter


def numbers_in(text: str) -> list[float]:
    """Every number in a piece of text, as floats."""
    values = []
    for whole, decimals, suffix in NUMBER.findall(text):
        value = float(whole.replace(",", "") + (f".{decimals}" if decimals else ""))
        if suffix:
            value *= SCALE[suffix.strip()[0].lower()]
        values.append(value)
    return values


@dataclass
class Fact:
    name: str                 # shown in the results, e.g. "6 missing salaries"
    keywords: list[str]       # at least one must be on the line (case-insensitive)
    value: float              # the correct number: counts and values from the report are exact

    def found_in(self, text: str) -> bool:
        # Within 0.5, so "999,999.99" written as "999,999" or "1.25M" for
        # 1,250,000 still match, but a wrong count (4 instead of 3) never does.
        for line in text.splitlines():
            lowered = line.lower()
            if not any(k.lower() in lowered for k in self.keywords):
                continue
            if any(abs(n - self.value) <= 0.5 for n in numbers_in(line)):
                return True
        return False


@dataclass
class Says:
    """A check with no number: the reply must contain this pattern somewhere,
    e.g. that the bot ASKS about day/month order, or says there's nothing to
    fix. pattern is a regular expression, matched case-insensitively."""
    name: str
    pattern: str

    def found_in(self, text: str) -> bool:
        return re.search(self.pattern, text, re.IGNORECASE) is not None


@dataclass
class NeverProposes:
    """A check that a numbered proposal line does NOT match the pattern, e.g.
    that the bot never proposes dropping the 53 plausible expenses. Only lines
    starting with a number ("3. Drop ...") count: those are proposed fixes, so
    a sentence such as "I would keep the 53 large expenses" can't fail it.

    Why a separate check: the probe (Expect) only sees what happened to the
    data after the user approved fixes. A bad PROPOSAL the user happens not to
    approve leaves the data fine, so it would score 100% without this."""
    name: str
    pattern: str

    def found_in(self, text: str) -> bool:   # "passes", named like the others
        proposals = [line for line in text.splitlines() if re.match(r"\s*(\*\*)?\d+[.)]", line)]
        return not any(re.search(self.pattern, line, re.IGNORECASE) for line in proposals)


# ---------------------------------------------------------------------------
# Checks for the SQL eval (sql_scenario.py)
# ---------------------------------------------------------------------------

@dataclass
class Number:
    """The reply contains this value somewhere, within tol. For an answer
    that has no fixed keyword to anchor a Fact to, e.g. "the average spend
    per employee is $1,842.17". tol=0.5 (like Fact) accepts the value at any
    sensible rounding but not a different number; 0 means exactly."""
    name: str
    expected: float
    tol: float = 0.5

    def found_in(self, text: str) -> bool:
        return any(abs(n - self.expected) <= self.tol for n in numbers_in(text))


@dataclass
class InOrder:
    """Each item appears in the reply (case-insensitive), first mentions in
    this order: a top 3 must have the right 3 in the right order."""
    name: str
    items: list[str]

    def found_in(self, text: str) -> bool:
        lowered = text.lower()
        positions = [lowered.find(item.lower()) for item in self.items]
        return -1 not in positions and positions == sorted(positions)


@dataclass
class AnyOf:
    """Passes if any of its checks passes: an answer that may be right in
    more than one way, e.g. it asks OR it states the reading it chose."""
    name: str
    checks: list

    def found_in(self, text: str) -> bool:
        return any(c.found_in(text) for c in self.checks)


@dataclass
class AllOf:
    """Passes if all its checks pass. Used inside AnyOf to keep a number
    together with the reading it belongs to: "34 employees' average" must
    come with the 34-employee value, not the 40-employee one."""
    name: str
    checks: list

    def found_in(self, text: str) -> bool:
        return all(c.found_in(text) for c in self.checks)


@dataclass
class SqlRun:
    """What one SQL question turn did, for checks that need more than the
    final reply's text."""
    question: str
    final_reply: str
    queries: list[dict]         # {"sql": ..., "status": "ok" | "error" | "refused" | "timeout"}
    results: list[str]          # what the model was shown for each OK query


@dataclass
class SelectOnly:
    """Query hygiene: no query was refused, i.e. every query the model wrote
    was one read-only SELECT. (Refused is the engine's verdict on writes,
    two statements, file access and settings; see executor/sql.py.)"""
    name: str = "every query was a SELECT"

    def check(self, run: SqlRun) -> bool:
        return all(q["status"] != "refused" for q in run.queries)


@dataclass
class QueryRan:
    """Query hygiene: at least one query ran without an error. Not used where
    the right reply needs no query (a column that doesn't exist, a question
    asked back)."""
    name: str = "a query ran OK"

    def check(self, run: SqlRun) -> bool:
        return any(q["status"] == "ok" for q in run.queries)


@dataclass
class NoInventedNumber:
    """Every number in the final reply comes from a query result of this
    turn (or the question, a year, a "top 3"): the same test as the agent's
    own number check (agent.loop.unsourced_numbers), applied to the reply
    the user finally got. The guard sends a reply back once; this checks
    what happened after that."""
    name: str = "no invented number"

    def check(self, run: SqlRun) -> bool:
        from agent.loop import unsourced_numbers   # here, so the cleaning eval doesn't need it
        return not unsourced_numbers(run.final_reply, run.question, run.results)


def score_facts(facts: list, text: str, run: SqlRun | None = None) -> dict[str, bool]:
    """{fact name: passed?} for every check. Text checks (Fact, Says, Number,
    ...) read the final reply; run checks (SelectOnly, QueryRan,
    NoInventedNumber) need the whole turn, given as run."""
    return {fact.name: fact.check(run) if hasattr(fact, "check") else fact.found_in(text)
            for fact in facts}