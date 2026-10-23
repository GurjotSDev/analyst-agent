"""
The agent loop: model writes code -> sandbox runs it -> model reads the result
-> ... until the model answers without code.

The loop reports what happens as a stream of EVENTS (small objects like
"the model said this", "code ran, here's the result"). It doesn't print or
draw anything itself. The terminal (cli.py) and later the Gradio UI just
display events, so neither needs to know how the loop works.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import codeblocks
from .executor import ExecutorClient, format_for_model

# Ways matplotlib/pandas code can set each chart part: a function call like
# ax.set_title(...) / plt.title(...) / fig.suptitle(...), or a keyword like
# df.plot(title=...) / ax.set(xlabel=...). \b = a word boundary, so "title"
# doesn't match inside e.g. "subtitle_text". \s* allows spaces before ( or =.
CHART_PARTS = {
    "a title": re.compile(r"\b(set_title|suptitle|title)\s*[(=]"),
    "an x-axis label": re.compile(r"\b(set_xlabel|xlabel)\s*[(=]"),
    "a y-axis label": re.compile(r"\b(set_ylabel|ylabel)\s*[(=]"),
}


# Repeated-error detection. In testing, the model spent ten steps trying
# variations of a fix for an error it had misdiagnosed, never stopping to
# re-read it. Counting each distinct error within one turn catches that.
NUDGE_AT = 3   # on the 3rd occurrence: tell the model to stop and diagnose
STOP_AT = 5    # on the 5th: end the turn and hand back to the user


def error_signature(result: dict) -> str | None:
    """The line that identifies an error, e.g. "KeyError: 'employee_id'".
    Python puts the exception type and message on the last line of a
    traceback, so the last non-empty line is the error's fingerprint.
    None for results that aren't the code's own errors (sandbox failures,
    restarts and timeouts have their own handling)."""
    if result.get("status") != "error" or result.get("executor_failure") or \
            result.get("kernel_restarted") or not result.get("error"):
        return None
    lines = [line.strip() for line in result["error"].splitlines() if line.strip()]
    return lines[-1] if lines else None


def missing_chart_parts(code: str) -> list[str]:
    """Which of title / x label / y label the code never sets.

    A text check, not a guarantee: it reads the code, not the image. It can
    be fooled (e.g. a title set inside a helper function), but it catches the
    common case of a model simply forgetting, which is what we need.
    """
    return [part for part, pattern in CHART_PARTS.items() if not pattern.search(code)]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
# @dataclass writes __init__ (and a readable print form) for us from the
# field list, so each event is a one-line definition.

@dataclass
class ModelReply:
    text: str                       # the model's full reply, code included


@dataclass
class NeedsApproval:
    code: str                       # waiting for the user: run this or not?


@dataclass
class CodeResult:
    code: str
    result: dict                    # raw executor response (for display)
    model_text: str                 # what the model was told


@dataclass
class Notice:
    text: str                       # something the user should know


@dataclass
class TurnDone:
    reason: str                     # "answered", "declined", "step_limit" or "stuck"


Event = ModelReply | NeedsApproval | CodeResult | Notice | TurnDone


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    llm: object                     # anything with chat(messages) -> str
    executor: ExecutorClient
    system_prompt: str
    require_approval: bool = True
    max_steps: int = 15
    max_result_chars: int = 5000
    history: list[dict] = field(default_factory=list)
    pending_code: str | None = None  # code waiting for approval, if any
    steps: int = 0                   # executions since the user's last message
    # How often each error has happened since the user's last message,
    # e.g. {"KeyError: 'employee_id'": 2}, and the error that made us stop.
    error_counts: dict[str, int] = field(default_factory=dict)
    stuck_on: str | None = None

    def __post_init__(self) -> None:
        # Runs right after the dataclass __init__. The conversation always
        # starts with the system prompt.
        if not self.history:
            self.history = [{"role": "system", "content": self.system_prompt}]

    # --- Public API: what the CLI / UI calls ---

    def send(self, user_text: str) -> Iterator[Event]:
        """The user says something. Runs until the model answers, asks a
        question, needs approval, or hits the step limit."""
        if self.pending_code is not None:
            raise RuntimeError("Approve or decline the pending code first.")
        self.history.append({"role": "user", "content": user_text})
        self.steps = 0
        self.error_counts = {}  # a new message is a fresh start
        self.stuck_on = None
        yield from self._run()

    def resume(self, approved: bool, feedback: str = "") -> Iterator[Event]:
        """Answer a NeedsApproval event.

        approved=True runs the code and carries on. A decline with feedback
        passes the feedback to the model so it can try again; a plain decline
        ends the turn.
        """
        code, self.pending_code = self.pending_code, None
        if code is None:
            raise RuntimeError("Nothing is waiting for approval.")
        if approved:
            yield from self._execute(code)
            yield from self._run()
        elif feedback:
            self.history.append({"role": "user", "content":
                f"I did not run that code. My feedback: {feedback}"})
            yield from self._run()
        else:
            self.history.append({"role": "user", "content":
                "I did not run that code. Wait for my next instruction."})
            yield TurnDone("declined")

    def new_analysis(self) -> None:
        """Fresh kernel and an empty conversation."""
        self.executor.reset()
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.pending_code = None
        self.steps = 0
        self.error_counts = {}
        self.stuck_on = None

    # --- Internals ---

    def _run(self) -> Iterator[Event]:
        while True:
            if self.steps >= self.max_steps:
                yield from self._wrap_up()
                return
            if self.stuck_on:
                # Same error STOP_AT times: more attempts are unlikely to help.
                # Stop and hand back to the user, and record it in the history
                # so the model knows why the turn ended.
                self.history.append({"role": "user", "content":
                    f"[Stopped] The same error happened {STOP_AT} times: {self.stuck_on}. "
                    "Wait for my next instruction."})
                yield Notice(f"Stopped: the same error happened {STOP_AT} times "
                             f"({self.stuck_on}). Give the model a hint, or try /new.")
                yield TurnDone("stuck")
                return

            reply = self.llm.chat(self.history)
            self.history.append(codeblocks.assistant_message(reply))
            yield ModelReply(reply)

            blocks = codeblocks.extract_code(reply)
            if not blocks:
                yield TurnDone("answered")  # an answer or a question: user's turn
                return
            if len(blocks) > 1:
                yield Notice(f"The model wrote {len(blocks)} code blocks; only the first runs.")

            code = blocks[0]
            if self.require_approval:
                self.pending_code = code
                yield NeedsApproval(code)
                return  # paused; the caller continues with resume()
            yield from self._execute(code, extra_blocks=len(blocks) - 1)

    def _execute(self, code: str, extra_blocks: int = 0) -> Iterator[Event]:
        self.steps += 1
        result = self.executor.execute(code)
        text = format_for_model(result, self.max_result_chars)
        if extra_blocks:
            text += ("\n\nNote: your reply had more than one python block. Only the first "
                     "was run. Write exactly one block per reply.")

        # A chart came back: did the code give it a title and axis labels?
        # If not, say exactly what's missing, so the model's next step is a
        # redraw. Telling it right away works better than a rule it may skip.
        missing = missing_chart_parts(code) if result.get("charts") else []
        if missing:
            text += (f"\n\nNote: the chart is missing {' and '.join(missing)}. Redraw it "
                     "with ax.set_title(...) stating the finding, and ax.set_xlabel(...) / "
                     "ax.set_ylabel(...) with units.")

        # The same error again? Count it; on the NUDGE_AT-th time, tell the
        # model to stop changing things and find the cause first.
        signature = error_signature(result)
        repeats = 0
        if signature:
            repeats = self.error_counts.get(signature, 0) + 1
            self.error_counts[signature] = repeats
            if repeats == NUDGE_AT:
                text += (f"\n\nNote: this same error has now happened {repeats} times: "
                         f"{signature}. Stop trying variations. First explain, in one or two "
                         "sentences, what is actually causing it: check every DataFrame, column "
                         "and variable used on the failing line, including column subsets like "
                         "df[['a', 'b']], which drop every column not listed. Then fix that cause.")
            if repeats >= STOP_AT:
                self.stuck_on = signature

        self.history.append(codeblocks.result_message(text))
        yield CodeResult(code, result, text)
        if missing:
            yield Notice(f"Chart is missing {', '.join(missing)}; asked the model to redraw it.")
        if repeats == NUDGE_AT:
            yield Notice(f"Same error {repeats} times ({signature}); asked the model to diagnose it.")
        if result.get("executor_failure"):
            yield Notice(f"Sandbox problem: {result['error']}")

    def _wrap_up(self) -> Iterator[Event]:
        """Step limit reached: ask for a summary instead of more code."""
        yield Notice(f"Step limit ({self.max_steps}) reached.")
        self.history.append({"role": "user", "content":
            "You have reached the step limit for this message. Do not write any more code. "
            "Summarize what you found so far and what you would check next."})
        reply = self.llm.chat(self.history)
        self.history.append(codeblocks.assistant_message(reply))
        yield ModelReply(reply)
        yield TurnDone("step_limit")