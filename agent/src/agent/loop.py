"""
The agent loop: model writes code -> sandbox runs it -> model reads the result
-> ... until the model answers without code.

The loop reports what happens as a stream of EVENTS (small objects like
"the model said this", "code ran, here's the result"). It doesn't print or
draw anything itself. The terminal (cli.py) and the web UI (session.py) just
display events, so neither needs to know how the loop works.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import codeblocks
from .executor import ExecutorClient, format_for_model

# Repeated-error detection: a model can keep trying variations of a fix for an
# error it misdiagnosed. Counting each distinct error within one turn catches that.
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


# A fix helper prints "[3] orders: drop_rows: 2 rows removed" for every change
# it logs. Those lines, before an error, are fixes that already happened.
LOGGED = re.compile(r"^\[(\d+)\] ", re.MULTILINE)

# A final reply saying the files are saved, used to catch a claim made without
# save_clean() having run. Narrow on purpose: "I'll save them once you approve"
# doesn't match; "have been saved" and "are available in output/clean" do.
SAVE_CLAIM = re.compile(r"\b(?:have|has) been saved\b|\b(?:were|was|are|is) (?:now )?saved\b|"
                        r"\bI(?: have|'ve)? saved\b|\b(?:are|is) (?:now )?available in\b[^\n]*output/clean",
                        re.IGNORECASE)
UNSAVED_NOTE = ("[Check] Your reply says the files were saved, but save_clean() has not run since the "
                "last change, so nothing was written. If the user asked to save, run save_clean() for "
                "each changed table now, in one block. Otherwise, correct your reply.")


def applied_note(result: dict) -> str:
    """After an error part-way through a block: which fixes already ran.

    Re-running a whole block after an error repeats fixes that already ran,
    and a repeated fix can fail on its own earlier result (e.g. a condition
    meeting a value it set to missing). Told what already ran, the model can
    continue from the failing line instead. Empty when nothing was logged."""
    if result.get("status") != "error" or result.get("kernel_restarted"):
        return ""
    numbers = LOGGED.findall(result.get("stdout") or "")
    if not numbers:
        return ""
    done = f"[{numbers[0]}]" if len(numbers) == 1 else f"[{numbers[0]}] to [{numbers[-1]}]"
    return (f"Note: before the error, {len(numbers)} fix{'' if len(numbers) == 1 else 'es'} "
            f"ran and {'was' if len(numbers) == 1 else 'were'} logged ({done} above). They are "
            "already applied: don't run them again. Your next block starts at the line that "
            "failed, and keeps every approved line after it.")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

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
    cut_note: str = ""               # told to the model with the next result
    kernel: int | None = None        # number of the kernel this conversation uses
    ran_code: bool = False           # has any code run in this kernel for us yet?
    unsaved: bool = False            # a logged change since the last save_clean()?
    save_checked: bool = False       # the false-"saved" check already ran this turn

    def __post_init__(self) -> None:
        # The conversation always starts with the system prompt.
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
        self.cut_note = ""
        self.save_checked = False
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
        if not approved:
            self.cut_note = ""   # it was about code that will never run
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
        self.kernel = self.executor.reset()
        self.ran_code = False
        self.unsaved = False
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.pending_code = None
        self.steps = 0
        self.error_counts = {}
        self.stuck_on = None
        self.cut_note = ""

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
            blocks = codeblocks.extract_code(reply)
            # Keep the reply only up to the end of its first code block: the
            # rest (more blocks, invented results, a premature summary) was
            # written before any code ran. Neither the history nor the user
            # sees it, and the model is told why with the real result.
            reply, cut = codeblocks.trim_after_first_block(reply)
            if len(blocks) > 1:
                self.cut_note = (f"Note: your reply had {len(blocks)} python blocks. Only the first was "
                                 "run, and everything after it was removed. Write exactly one block per "
                                 "reply, then stop.")
            elif cut:
                self.cut_note = ("Note: the text you wrote after your code block was removed, because no "
                                 "code had run yet. Never write results or conclusions yourself; stop "
                                 "after the block and wait for the [Execution result].")
            self.history.append(codeblocks.assistant_message(reply))
            yield ModelReply(reply)

            if not blocks:
                # Claims the files are saved, but changes since the last save
                # were never written: tell the model, once per turn.
                if self.unsaved and not self.save_checked and SAVE_CLAIM.search(reply):
                    self.save_checked = True
                    self.history.append({"role": "user", "content": UNSAVED_NOTE})
                    yield Notice("The model said the files were saved, but save_clean() hadn't run "
                                 "since the last change; it was asked to save them or correct its reply.")
                    continue
                yield TurnDone("answered")  # an answer or a question: user's turn
                return
            if len(blocks) > 1:
                yield Notice(f"The model wrote {len(blocks)} code blocks; only the first runs, "
                             "and the rest of its reply was removed.")

            code = blocks[0]
            if self.require_approval:
                self.pending_code = code
                yield NeedsApproval(code)
                return  # paused; the caller continues with resume()
            yield from self._execute(code)

    def _execute(self, code: str) -> Iterator[Event]:
        self.steps += 1
        # Another client reset the shared kernel since our last step. The code
        # would fail on tables that no longer exist, so don't run it: tell the
        # model why. If none of our code has run yet, nothing was lost, so just
        # adopt the new kernel.
        other_reset = False
        if self.kernel is not None:
            now = self.executor.kernel()
            if now is not None and now != self.kernel:
                if self.ran_code:
                    other_reset = True
                else:
                    self.kernel = now
        if other_reset:
            result = {"status": "error", "stdout": "", "stderr": "", "truncated": False,
                      "kernel_restarted": True, "kernel": now,
                      "error": "Your code was NOT run: another program reset the sandbox's kernel "
                               "since your last step. Every table, variable and the change log are "
                               "gone. Load the files again with load(), then redo the fixes the user "
                               "approved."}
        else:
            result = self.executor.execute(code)
            self.ran_code = self.ran_code or not result.get("executor_failure")
        if result.get("kernel"):
            self.kernel = result["kernel"]   # adopt the new kernel, ours from here on
        self._track_saves(result.get("stdout") or "")
        text = format_for_model(result, self.max_result_chars)
        note = applied_note(result)
        if note:
            text += "\n\n" + note
        if self.cut_note:
            text += "\n\n" + self.cut_note
            self.cut_note = ""

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
        if repeats == NUDGE_AT:
            yield Notice(f"Same error {repeats} times ({signature}); asked the model to diagnose it.")
        if result.get("executor_failure"):
            yield Notice(f"Sandbox problem: {result['error']}")
        if other_reset:
            yield Notice("Another program (the tests, the CLI or the eval) reset the sandbox, so every "
                         "table was lost. The code was not run; the model was told to load the files again.")

    def _track_saves(self, stdout: str) -> None:
        """Is there a logged change that no save_clean() has written yet?
        A [n] line is a change; a save_clean: line writes everything logged
        before it. So whichever comes LAST in the printout decides."""
        last_change = max((m.start() for m in LOGGED.finditer(stdout)), default=-1)
        last_save = stdout.rfind("save_clean:")
        if last_change > last_save:
            self.unsaved = True
        elif last_save > last_change:
            self.unsaved = False

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