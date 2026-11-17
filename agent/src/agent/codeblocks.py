"""
Code-block mode: how code travels between the model and the agent.

The model writes code in fenced blocks inside its normal reply (```python for
the cleaning agent, ```sql for the question agent), and results go back as a
user message. This module is the ONLY place that knows that format, so a
tool-calling version would be a sibling module with the same functions.
"""

import re

# Which fence labels count as which language. ```py is accepted for python
# because models write it; ```SQL because SQL is often written in capitals.
_LANG_LABELS = {"python": "python|py", "sql": "sql|SQL"}


def block_pattern(lang: str) -> re.Pattern:
    """A fenced block in one language, up to the first closing fence. A block
    with no closing fence is ignored on purpose: the reply was probably cut
    off, and running half a program (or half a query) is worse than nothing."""
    return re.compile(rf"```(?:{_LANG_LABELS[lang]})[ \t]*\n(.*?)```", re.DOTALL)


# Built once; cli.py and session.py use CODE_BLOCK to hide code in the display.
CODE_BLOCK = block_pattern("python")
SQL_BLOCK = block_pattern("sql")
_PATTERNS = {"python": CODE_BLOCK, "sql": SQL_BLOCK}


def extract(reply: str, lang: str = "python") -> list[str]:
    """All non-empty blocks of one language in a model reply, in order."""
    return [code.strip() for code in _PATTERNS[lang].findall(reply) if code.strip()]


def trim_after_first(reply: str, lang: str = "python") -> tuple[str, str]:
    """Split a reply at the end of its first block: (kept, cut).

    Anything after the first block was written before it ran, so it can only
    be a guess (more code, or invented results). Cutting it keeps the history
    to real results only. A reply without a block is kept whole."""
    match = _PATTERNS[lang].search(reply)
    if not match:
        return reply, ""
    return reply[:match.end()], reply[match.end():].strip()


def extract_code(reply: str) -> list[str]:
    """All python code blocks in a model reply, in order."""
    return extract(reply, "python")


def trim_after_first_block(reply: str) -> tuple[str, str]:
    """trim_after_first() for python blocks."""
    return trim_after_first(reply, "python")


def assistant_message(reply: str) -> dict:
    """The model's reply, as stored in the history."""
    return {"role": "assistant", "content": reply}


def result_message(text: str, label: str = "Execution result") -> dict:
    """A result, as stored in the history. In this mode it's a user message,
    labelled so it can't be mistaken for the user: [Execution result] for
    python, [Query result] for SQL."""
    return {"role": "user", "content": f"[{label}]\n{text}"}
