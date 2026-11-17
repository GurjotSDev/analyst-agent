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
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import ClassVar

from . import codeblocks
from .executor import ExecutorClient, format_for_model, format_sql_for_model

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
    # Hooks a subclass sets (see SqlAgent): which fenced blocks run, how the
    # result is labelled in the history, and the hint given with the
    # repeated-error nudge. Class-level, because they describe the kind of
    # agent, not one conversation.
    block_lang: ClassVar[str] = "python"
    result_label: ClassVar[str] = "Execution result"
    nudge_hint: ClassVar[str] = (
        "Stop trying variations. First explain, in one or two sentences, what is actually causing "
        "it: check every DataFrame, column and variable used on the failing line, including column "
        "subsets like df[['a', 'b']], which drop every column not listed. Then fix that cause.")

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
        self._clear_conversation()

    def _clear_conversation(self) -> None:
        """An empty conversation: only the system prompt, nothing pending."""
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
            blocks = codeblocks.extract(reply, self.block_lang)
            # Keep the reply only up to the end of its first code block: the
            # rest (more blocks, invented results, a premature summary) was
            # written before any code ran. Neither the history nor the user
            # sees it, and the model is told why with the real result.
            reply, cut = codeblocks.trim_after_first(reply, self.block_lang)
            if len(blocks) > 1:
                self.cut_note = (f"Note: your reply had {len(blocks)} {self.block_lang} blocks. Only the first was "
                                 "run, and everything after it was removed. Write exactly one block per "
                                 "reply, then stop.")
            elif cut:
                self.cut_note = ("Note: the text you wrote after your code block was removed, because no "
                                 "code had run yet. Never write results or conclusions yourself; stop "
                                 f"after the block and wait for the [{self.result_label}].")
            self.history.append(codeblocks.assistant_message(reply))
            yield ModelReply(reply)

            if not blocks:
                # A final reply. _check_final() can send it back (at most once
                # per turn, each check keeps its own flag), with a note to the
                # model and a notice to the user.
                problem = self._check_final(reply)
                if problem:
                    note, notice = problem
                    self.history.append({"role": "user", "content": note})
                    yield Notice(notice)
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

    # --- Hooks a subclass can replace ---

    def run(self, code: str) -> tuple[dict, str]:
        """Run one block. Returns (raw result, for display; text, for the
        model). Cleaning: the python kernel, with the kernel-generation
        check, save tracking and the "already ran" note."""
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
                      "kernel_restarted": True, "kernel": now, "not_run": True,
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
        return result, text

    def _error_signature(self, result: dict) -> str | None:
        """The fingerprint of an error, for the repeated-error count."""
        return error_signature(result)

    def _check_final(self, reply: str) -> tuple[str, str] | None:
        """Is something wrong with a final (code-free) reply? Returns (note
        for the model, notice for the user), or None to end the turn.

        Cleaning: the reply says the files are saved, but changes since the
        last save were never written. Once per turn."""
        if self.unsaved and not self.save_checked and SAVE_CLAIM.search(reply):
            self.save_checked = True
            return UNSAVED_NOTE, ("The model said the files were saved, but save_clean() hadn't run "
                                  "since the last change; it was asked to save them or correct its reply.")
        return None

    def _execute(self, code: str) -> Iterator[Event]:
        """Run one block through run(), then do what every agent shares: the
        step count, the cut note, the repeated-error count, the history."""
        self.steps += 1
        result, text = self.run(code)
        if self.cut_note:
            text += "\n\n" + self.cut_note
            self.cut_note = ""

        # The same error again? Count it; on the NUDGE_AT-th time, tell the
        # model to stop changing things and find the cause first.
        signature = self._error_signature(result)
        repeats = 0
        if signature:
            repeats = self.error_counts.get(signature, 0) + 1
            self.error_counts[signature] = repeats
            if repeats == NUDGE_AT:
                text += (f"\n\nNote: this same error has now happened {repeats} times: "
                         f"{signature}. {self.nudge_hint}")
            if repeats >= STOP_AT:
                self.stuck_on = signature

        self.history.append(codeblocks.result_message(text, self.result_label))
        yield CodeResult(code, result, text)
        if repeats == NUDGE_AT:
            yield Notice(f"Same error {repeats} times ({signature}); asked the model to diagnose it.")
        if result.get("executor_failure"):
            yield Notice(f"Sandbox problem: {result['error']}")
        if result.get("not_run"):
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


# ---------------------------------------------------------------------------
# The number check (SqlAgent): every number in an answer must come from a query
# ---------------------------------------------------------------------------

# A number as people write it: optional $ and minus, thousands commas,
# decimals, and a scale word or a %. Not part of a word or a longer number
# (E001, Q3, 3rd, v1.2 are skipped).
_NUMBER = re.compile(
    r"(?<![\w.,])(?P<text>-?\$?-?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"(?:(?P<pct>%)|\s?(?P<word>thousand|million|billion)\b|(?P<short>[kKM]|bn)(?!\w))?)(?![\w%])")
_SCALE = {"k": 3, "K": 3, "thousand": 3, "M": 6, "million": 6, "bn": 9, "billion": 9}

# Numbers inside a query result, as str() writes them (exponents included).
_RESULT_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")   # the commas in 1,000 rows

# Not checked: code, fenced or inline (`LIMIT 10` is SQL, not a claim), and
# list markers.
_FENCED_CODE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)

# "top 3", "the first 5": a small whole number after a ranking word counts
# things, it isn't a measured value.
_RANKING = re.compile(r"\b(?:top|bottom|first|last|highest|lowest|largest|smallest|biggest|best|"
                      r"worst)\s+$", re.IGNORECASE)
SMALL_COUNT = 10

# "treated as 0", "counting none as $0": a zero after "as" says how missing
# values were counted (the prompt's own example How line does this). It's a
# convention, not a measured value, so it needs no query.
_AS_ZERO = re.compile(r"\bas\s+\$?\s*$", re.IGNORECASE)

UNSOURCED_NOTE = ("[Check] Your reply states {numbers}, which no query returned. "
                  "Run a query for it or correct the reply.")


@dataclass
class Stated:
    """A number written in a reply: its text, its value, and the step it was
    written at (1 for "1,520", 0.1 for "1,520.4", 100000 for "1.2 million")."""
    text: str
    value: Decimal
    quantum: Decimal
    percent: bool
    start: int        # where it starts in the text numbers_in() scanned


def _prose(text: str) -> str:
    """The text with code and list markers blanked out."""
    return _LIST_MARKER.sub(" ", _INLINE_CODE.sub(" ", _FENCED_CODE.sub(" ", text)))


def numbers_in(text: str) -> list[Stated]:
    """Every number written in a piece of prose (see _prose()), with the
    precision it's written at."""
    found = []
    for m in _NUMBER.finditer(text):
        decimals = len(m["dec"]) - 1 if m["dec"] else 0
        scale = _SCALE.get(m["word"] or m["short"] or "", 0)
        digits = m["int"].replace(",", "") + (m["dec"] or "")
        found.append(Stated(text=m["text"].strip(), value=Decimal(digits).scaleb(scale),
                            quantum=Decimal(1).scaleb(scale - decimals), percent=bool(m["pct"]),
                            start=m.start()))
    return found


def result_values(texts: list[str]) -> set[Decimal]:
    """Every number in some query results, sign dropped ("fell by 5" can
    describe -5)."""
    values = set()
    for text in texts:
        for token in _RESULT_NUMBER.findall(_THOUSANDS.sub("", text)):
            values.add(Decimal(token))
    return values


def _shown_as(value: Decimal, quantum: Decimal) -> set[Decimal]:
    """value written at a given step, the ways a person or model might round
    it: half up, half to even (Python's own rounding), or simply cut off."""
    steps = value / quantum
    return {steps.to_integral_value(rounding=mode) * quantum
            for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN, ROUND_DOWN)}


def matches(stated: Stated, values: set[Decimal]) -> bool:
    """Is this number one of the values, at the precision it's written at?
    "1,520.4" matches 1520.40 and 1520.43; "1,520" matches 1520.4. A % can
    also be a fraction the query returned (12.5% from 0.125)."""
    target = stated.value
    for value in values:
        candidates = [value, value * 100] if stated.percent else [value]
        if any(target in _shown_as(c, stated.quantum) for c in candidates):
            return True
    return False


def unsourced_numbers(reply: str, question: str, results: list[str]) -> list[str]:
    """The numbers in a final reply that no query result backs up.

    Exempt: numbers the user wrote in the question, years (1900-2100,
    written as plain whole numbers), small counts after a ranking word
    ("top 3"), and a zero after "as" ("no expenses counted as 0")."""
    values = result_values(results)
    asked = {abs(n.value) for n in numbers_in(_prose(question))}
    prose = _prose(reply)
    missing = []
    for stated in numbers_in(prose):
        stated.value = abs(stated.value)
        whole = stated.quantum == 1
        if stated.value in asked:
            continue
        if whole and "," not in stated.text and 1900 <= stated.value <= 2100:
            continue
        if whole and stated.value <= SMALL_COUNT and _RANKING.search(prose[:stated.start]):
            continue
        if stated.value == 0 and _AS_ZERO.search(prose[:stated.start]):
            continue
        if not matches(stated, values):
            missing.append(stated.text)
    return list(dict.fromkeys(missing))   # each once, in order


# ---------------------------------------------------------------------------
# The question agent
# ---------------------------------------------------------------------------

def sql_error_signature(result: dict) -> str | None:
    """The fingerprint of a SQL error: its FIRST line. DuckDB ends a message
    with the query and a ^^^^ marker, so the last line (python's choice) would
    make every error look the same. A refused write counts too: a model
    that keeps trying to change data should be stopped the same way."""
    if result.get("status") not in ("error", "refused") or result.get("executor_failure") \
            or not result.get("error"):
        return None
    lines = [line.strip() for line in result["error"].splitlines() if line.strip()]
    return lines[0] if lines else None


@dataclass
class SqlAgent(Agent):
    """Answers questions with read-only SQL. Same loop as the cleaning agent
    (one block per reply, step limit, repeated-error nudge and stop); only
    the block language, how a block runs, and the final-reply check differ.
    Queries can't change anything, so they run without asking."""

    block_lang: ClassVar[str] = "sql"
    result_label: ClassVar[str] = "Query result"
    nudge_hint: ClassVar[str] = (
        "Stop trying variations. First explain, in one or two sentences, what is actually causing "
        "it: check each table and column name against the schema, and the type of every value you "
        "compare or combine. Then fix that cause.")

    require_approval: bool = False
    question: str = ""                                       # the user's latest message
    turn_results: list[str] = field(default_factory=list)   # successful results this turn
    numbers_checked: bool = False    # the number check already sent a reply back this turn
    # Sandbox paths of the tables this session may query, sent with every
    # query (AskSession sets it). Kept by new_analysis: the tables don't change.
    files: list[str] = field(default_factory=list)

    def send(self, user_text: str) -> Iterator[Event]:
        self.question = user_text
        self.turn_results = []
        self.numbers_checked = False
        yield from super().send(user_text)

    def new_analysis(self) -> None:
        """An empty conversation. Unlike the cleaning agent, the kernel is NOT
        reset: /sql doesn't use it, and the cleaning tab may be using it."""
        self._clear_conversation()
        self.question = ""
        self.turn_results = []
        self.numbers_checked = False

    def run(self, query: str) -> tuple[dict, str]:
        result = self.executor.sql(query, self.files)
        text = format_sql_for_model(result)
        if result.get("status") == "ok":
            # Only real rows back a number: an error message quotes the
            # query, and its numbers were written by the model.
            self.turn_results.append(text)
        return result, text

    def _error_signature(self, result: dict) -> str | None:
        return sql_error_signature(result)

    def _check_final(self, reply: str) -> tuple[str, str] | None:
        """The number check: a number in the answer that no query returned
        this turn sends the reply back once, so the model checks it."""
        if self.numbers_checked:
            return None
        missing = unsourced_numbers(reply, self.question, self.turn_results)
        if not missing:
            return None
        self.numbers_checked = True
        shown = missing[:5]
        named = shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]
        return (UNSOURCED_NOTE.format(numbers=named),
                f"The reply stated {named}, which no query returned; the model was asked to "
                "check it.")

