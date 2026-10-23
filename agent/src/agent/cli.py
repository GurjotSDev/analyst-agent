"""
Terminal interface for the agent (stage 4). The Gradio UI replaces this in stage 6.

Run from the agent/ folder:
    uv run analyst

Commands while chatting:
    /new        start a new analysis (fresh kernel, empty conversation)
    /auto       toggle approval mode (ask before running each code block)
    /quit       exit
"""

import textwrap

from . import config
from .codeblocks import CODE_BLOCK
from .executor import ExecutorClient, ExecutorNotSafe
from .llm import ModelNotAvailable, OllamaLLM
from .loop import Agent, CodeResult, ModelReply, NeedsApproval, Notice, TurnDone
from .prompts import build_system_prompt

RULE = "-" * 70


def show_code(code: str) -> None:
    # Number the lines so it's easy to refer to them in feedback ("line 4 is wrong").
    for n, line in enumerate(code.splitlines(), start=1):
        print(f"  {n:3d} | {line}")


def show_result(event: CodeResult, show_code_first: bool) -> None:
    if show_code_first:  # in auto mode nobody saw the code yet
        print("\nRunning:")
        show_code(event.code)
    r = event.result
    status = {"ok": "ran OK", "error": "ERROR", "timeout": "TIMED OUT"}.get(r["status"], r["status"])
    print(f"\n[{status}]")
    if r.get("stdout"):
        # Same limit as what the model sees, so you never see less than it does.
        limit = config.MAX_RESULT_CHARS
        print(textwrap.indent(r["stdout"][:limit].rstrip(), "  "))
        if len(r["stdout"]) > limit:
            print("  ... (more output, cut for display)")
    if r.get("error"):
        # The last lines of a traceback carry the actual error.
        print(textwrap.indent("\n".join(r["error"].strip().splitlines()[-6:]), "  "))
    for chart in r.get("charts", []):
        print(f"  chart saved: {config.OUTPUT_DIR / 'charts' / (chart['id'] + '.png')}")
    if r.get("kernel_restarted"):
        print("  (the kernel was restarted; all variables were lost)")


def ask_approval(code: str) -> tuple[bool, str]:
    print("\nCode to run:")
    show_code(code)
    answer = input("Run it? [y] yes / [n] no / or type feedback for the model: ").strip()
    if answer.lower() in ("y", "yes"):
        return True, ""
    if answer.lower() in ("n", "no", ""):
        return False, ""
    return False, answer  # anything else is feedback


def drive(agent: Agent, events) -> None:
    """Display events; when approval is needed, ask and keep going."""
    while events is not None:
        next_events = None
        for event in events:
            match event:  # Python's pattern matching: pick a branch by event type
                case ModelReply(text=text):
                    # Code blocks are shown separately, numbered, so replace
                    # them here with a placeholder instead of printing twice.
                    prose = CODE_BLOCK.sub("[code below]", text).strip()
                    print(f"\n{RULE}\nModel:\n{prose}\n{RULE}")
                case NeedsApproval(code=code):
                    approved, feedback = ask_approval(code)
                    next_events = agent.resume(approved, feedback)
                case CodeResult():
                    show_result(event, show_code_first=not agent.require_approval)
                case Notice(text=text):
                    print(f"\n(note) {text}")
                case TurnDone(reason="step_limit"):
                    print("\n(Reached the step limit for this message.)")
        events = next_events


def main() -> None:
    try:
        executor = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
        executor.check_sandboxed()
        llm = OllamaLLM(config.MODEL, config.OLLAMA_HOST, config.THINK, config.NUM_CTX)
        llm.check_available()
    except (ExecutorNotSafe, ModelNotAvailable) as e:
        print(f"Can't start: {e}")
        return

    files = sorted(p.name for p in config.DATA_DIR.glob("*.csv"))
    agent = Agent(
        llm=llm,
        executor=executor,
        system_prompt=build_system_prompt(files),
        max_steps=config.MAX_STEPS,
        max_result_chars=config.MAX_RESULT_CHARS,
    )

    print(f"Analyst ready. Model: {config.MODEL}. Files: {', '.join(files) or 'none'}")
    print("Approval mode is ON. Commands: /new  /auto  /quit\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+Z/Ctrl+D or Ctrl+C
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            agent.new_analysis()
            print("(Started a new analysis.)")
            continue
        if text == "/auto":
            agent.require_approval = not agent.require_approval
            print(f"(Approval mode {'ON' if agent.require_approval else 'OFF'}.)")
            continue
        try:
            drive(agent, agent.send(text))
        except Exception as e:
            # Anything unexpected (e.g. Ollama stopped mid-answer): report it
            # and return to the prompt instead of ending the program. The
            # kernel, and everything loaded in it, survives.
            print(f"\n(error) {type(e).__name__}: {e}")
            print("(Your session is still open. Fix the problem, then continue or /new.)")
            agent.pending_code = None  # never leave the agent stuck waiting for approval


if __name__ == "__main__":
    main()