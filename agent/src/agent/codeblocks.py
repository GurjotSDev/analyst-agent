"""
Code-block mode: how code travels between the model and the agent.

The model writes code in ```python fenced blocks inside its normal reply, and
execution results go back as a user message. This module is the ONLY place that
knows that format. Supporting tool-calling mode later (see stage 1) would mean
writing a sibling module with the same three functions.
"""

import re

# ```python (or ```py), optional spaces, newline, then the code, lazily (.*?)
# up to the first closing ```. re.DOTALL lets "." match newlines too.
# A block with no closing fence is ignored on purpose: it usually means the
# reply was cut off, and running half a program is worse than running nothing.
CODE_BLOCK = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> list[str]:
    """All python code blocks in a model reply, in order."""
    return [code.strip() for code in CODE_BLOCK.findall(reply) if code.strip()]


def assistant_message(reply: str) -> dict:
    """The model's reply, as stored in the history."""
    return {"role": "assistant", "content": reply}


def result_message(text: str) -> dict:
    """An execution result, as stored in the history. In this mode it's a
    user message, clearly labelled so the model can't mistake it for you."""
    return {"role": "user", "content": f"[Execution result]\n{text}"}