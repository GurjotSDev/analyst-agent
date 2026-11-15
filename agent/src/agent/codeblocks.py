"""
Code-block mode: how code travels between the model and the agent.

The model writes code in ```python fenced blocks inside its normal reply, and
execution results go back as a user message. This module is the ONLY place that
knows that format, so a tool-calling version would be a sibling module with
the same functions.
"""

import re

# A ```python (or ```py) block up to the first closing fence. A block with no
# closing fence is ignored on purpose: the reply was probably cut off, and
# running half a program is worse than running nothing.
CODE_BLOCK = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> list[str]:
    """All python code blocks in a model reply, in order."""
    return [code.strip() for code in CODE_BLOCK.findall(reply) if code.strip()]


def trim_after_first_block(reply: str) -> tuple[str, str]:
    """Split a reply at the end of its first code block: (kept, cut).

    Anything after the first block was written before any code ran, so it
    can only be a guess (more code, or invented results). Cutting it keeps
    the history to real results only. A reply without a code block is kept
    whole."""
    match = CODE_BLOCK.search(reply)
    if not match:
        return reply, ""
    return reply[:match.end()], reply[match.end():].strip()


def assistant_message(reply: str) -> dict:
    """The model's reply, as stored in the history."""
    return {"role": "assistant", "content": reply}


def result_message(text: str) -> dict:
    """An execution result, as stored in the history. In this mode it's a
    user message, labelled so it can't be mistaken for the user."""
    return {"role": "user", "content": f"[Execution result]\n{text}"}