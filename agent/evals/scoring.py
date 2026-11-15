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


def score_facts(facts: list, text: str) -> dict[str, bool]:
    """{fact name: passed?} for every Fact, Says or NeverProposes."""
    return {fact.name: fact.found_in(text) for fact in facts}