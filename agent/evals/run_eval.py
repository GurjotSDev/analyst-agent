"""
Run the evaluation: replay each scenario with one or more models, several
times each, and score every answer against the known-correct results.

Needs the sandbox running (docker compose up -d), Ollama with the models
pulled, and the eval datasets (uv run python evals/make_datasets.py, once).
Run from the agent/ folder:

    uv run python evals/run_eval.py                              # every scenario, 3 runs
    uv run python evals/run_eval.py --scenarios banking --runs 5
    uv run python evals/run_eval.py --models gemma4:12b qwen3.5:9b --runs 3
    uv run python evals/run_eval.py --steps 1                    # first message only (quick)

Each run is saved in full (transcript + scores) to output/evals/, so you can
read exactly what happened in any run that failed.
"""

import argparse   # reads command-line options like --runs 5
import json
import time
from datetime import datetime

from agent import config
from agent.executor import ExecutorClient
from agent.llm import OllamaLLM
from agent.loop import Agent, CodeResult, ModelReply, Notice, TurnDone
from agent.prompts import build_system_prompt
from scenario import BY_NAME, SCENARIOS, Scenario, probe_code, score_probe
from scoring import score_facts


def run_step(agent: Agent, step, scenario: Scenario, executor: ExecutorClient) -> dict:
    """Send one message, collect what happened, and score it."""
    start = time.perf_counter()
    replies, notices = [], []
    executions = errors = 0
    peak_tokens = 0
    end = "error"
    try:
        for event in agent.send(step.message):
            match event:
                case ModelReply(text=text):
                    replies.append(text)
                    # Largest request in this message: how full the context got.
                    usage = getattr(agent.llm, "last_usage", {})   # fake models have none
                    peak_tokens = max(peak_tokens, usage.get("prompt_tokens", 0))
                case CodeResult(result=r):
                    executions += 1
                    errors += r["status"] != "ok"   # True counts as 1, False as 0
                case Notice(text=text):
                    notices.append(text)
                case TurnDone(reason=reason):
                    end = reason
        crashed = None
    except Exception as e:   # one broken run must not stop the whole evaluation
        crashed = f"{type(e).__name__}: {e}"

    final_reply = replies[-1] if replies else ""
    checks = score_facts(step.facts, final_reply)
    probe_errors = {}
    if step.probe:
        probe = executor.execute(probe_code(scenario))
        found, probe_errors = score_probe(probe.get("stdout", ""), scenario)
        checks |= found

    return {
        "step": step.name,
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "probe_errors": probe_errors,   # why an expectation couldn't be evaluated
        "seconds": round(time.perf_counter() - start, 1),
        "executions": executions,
        "errors": errors,
        "end": end,
        "crashed": crashed,
        "notices": notices,
        "peak_prompt_tokens": peak_tokens,
        "final_reply": final_reply,
    }


def run_once(llm, executor: ExecutorClient, scenario: Scenario, n_steps: int | None = None,
             run_name: str = "test") -> list[dict]:
    """One conversation about one scenario's files, from a fresh start."""
    agent = Agent(llm=llm, executor=executor, system_prompt=build_system_prompt(scenario.files),
                  require_approval=False,   # unattended: no one to approve each step
                  max_steps=config.MAX_STEPS, max_result_chars=config.MAX_RESULT_CHARS)
    agent.new_analysis()   # fresh kernel and conversation for every run
    # Each run saves into its own folder, output/evals/<run_name>/clean/, so
    # eval runs never overwrite your real output/clean/ or each other.
    # The folder is emptied first, so files from an earlier run with the same
    # name (the harness tests always use "test") can never count for this one.
    executor.execute(
        "import cleaning, shutil\n"
        f"cleaning.OUTPUT_DIR = cleaning.OUTPUT_DIR / 'evals' / {run_name!r}\n"
        "shutil.rmtree(cleaning.OUTPUT_DIR, ignore_errors=True)")
    results = []
    for step in scenario.steps[:n_steps]:
        results.append(run_step(agent, step, scenario, executor))
    results.append({"transcript": agent.history})
    return results


def print_summary(all_results: dict[str, dict[str, list[list[dict]]]], scenarios: list[Scenario]) -> None:
    """One row per model, one column per scenario: checks passed / checks run."""
    width = 12 * len(scenarios) + 49
    print("\n" + "=" * width)
    print(f"{'model':<18}" + "".join(f"{s.name:>12}" for s in scenarios) +
          f"{'overall':>9}{'min/run':>9}{'peak tok':>10}{'crashes':>9}")
    for model, by_scenario in all_results.items():
        cells, passed_all, total_all, seconds, n_runs, peak, crashes = [], 0, 0, 0.0, 0, 0, 0
        for s in scenarios:
            runs = by_scenario[s.name]
            steps = [step for run in runs for step in run[:-1]]   # [-1] is the transcript
            passed, total = sum(x["passed"] for x in steps), sum(x["total"] for x in steps)
            cells.append(f"{passed}/{total}")
            passed_all, total_all = passed_all + passed, total_all + total
            seconds += sum(x["seconds"] for x in steps)
            n_runs += len(runs)
            peak = max([peak] + [x["peak_prompt_tokens"] for x in steps])
            crashes += sum(any(x["crashed"] for x in run[:-1]) for run in runs)
        overall = f"{100 * passed_all / max(total_all, 1):.0f}%"
        print(f"{model:<18}" + "".join(f"{c:>12}" for c in cells) +
              f"{overall:>9}{seconds / max(n_runs, 1) / 60:>9.1f}{peak:>10,}{crashes:>9}")
    print("=" * width)
    print("Each cell: checks passed / checks run, summed over all runs of that scenario.")
    print("min/run: average minutes per scenario run.")
    print(f"peak tok: largest request sent to the model (context window: {config.NUM_CTX:,}). "
          "Near the window, Ollama silently drops the oldest messages.")
    for s in scenarios:
        print(f"  {s.name:<10} {s.tests}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on the cleanup scenarios.")
    parser.add_argument("--models", nargs="+", default=[config.MODEL])
    parser.add_argument("--scenarios", nargs="+", default=[s.name for s in SCENARIOS],
                        choices=[s.name for s in SCENARIOS])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=None,
                        help="only run the first N messages of each scenario (for a quick check)")
    args = parser.parse_args()
    scenarios = [BY_NAME[name] for name in args.scenarios]

    missing = [f for s in scenarios for f in s.files if not (config.DATA_DIR / f).exists()]
    if missing:
        print(f"Missing data files: {', '.join(missing)}\n"
              "Generate the eval datasets first: uv run python evals/make_datasets.py")
        return

    executor = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
    executor.check_sandboxed()
    out_dir = config.OUTPUT_DIR / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    all_results = {}
    for model in args.models:
        llm = OllamaLLM(model, config.OLLAMA_HOST, config.THINK, config.NUM_CTX)
        llm.check_available()
        all_results[model] = {}
        for scenario in scenarios:
            runs = []
            for i in range(1, args.runs + 1):
                print(f"\n{model}  {scenario.name}  run {i}/{args.runs}")
                name = f"{stamp}_{model.replace(':', '-')}_{scenario.name}_run{i}"
                results = run_once(llm, executor, scenario, args.steps, run_name=name)
                for r in results[:-1]:
                    flag = "PASS" if r["passed"] == r["total"] else "    "
                    print(f"  {flag} {r['step']:<9} {r['passed']:>2}/{r['total']:<3}"
                          f"{r['seconds']:>6.1f}s  {r['executions']} runs, {r['errors']} errors  "
                          f"{r['peak_prompt_tokens']:>6,} tok"
                          + (f"  CRASHED: {r['crashed']}" if r["crashed"] else ""))
                    for check_name, ok in r["checks"].items():
                        if not ok:
                            reason = r["probe_errors"].get(check_name)
                            print(f"         missed: {check_name}" + (f"  ({reason})" if reason else ""))
                runs.append(results)
                # One JSON file per run: easy to open and read on its own.
                (out_dir / f"{name}.json").write_text(json.dumps(results, indent=2, default=str),
                                                      encoding="utf-8")
            all_results[model][scenario.name] = runs

    print_summary(all_results, scenarios)
    print(f"Full transcripts: {out_dir}")


if __name__ == "__main__":
    main()