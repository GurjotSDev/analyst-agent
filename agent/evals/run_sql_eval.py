"""
Run the SQL eval: ask the Ask-questions agent (SqlAgent) each scenario's
question with the real model, several times, and score every final answer
against answers computed with pandas (sql_scenario.py).

Needs the sandbox running (docker compose up -d), Ollama with the model
pulled, and the tables (uv run python evals/make_datasets.py, once).
Run from the agent/ folder:

    uv run python evals/run_sql_eval.py                         # every scenario, 3 runs
    uv run python evals/run_sql_eval.py --scenarios top3 --runs 5
    uv run python evals/run_sql_eval.py --models gemma4:12b qwen3.5:9b

Each run is saved in full (queries, transcript, scores) to output/evals/,
named ..._sql-<scenario>_runN.json, next to the cleaning eval's runs.
"""

import argparse
import json
import time
from datetime import datetime

from agent import config
from agent.executor import ExecutorClient
from agent.llm import OllamaLLM
from agent.loop import CodeResult, ModelReply, Notice, SqlAgent, TurnDone
from agent.sql_prompts import build_sql_prompt
from run_eval import print_summary
from scoring import SqlRun, score_facts
from sql_scenario import FILES, SQL_BY_NAME, SQL_SCENARIOS, SqlScenario, compute_answers


def make_agent(llm, executor: ExecutorClient, schema: dict) -> SqlAgent:
    """A fresh SqlAgent on the eval tables, set up the way make_ask_session()
    sets up the Ask tab's (same prompt builder, step limit and file list).
    No data dictionary: these tables were never cleaned, so there are no
    "Changes" notes, and the shared employee_id column is the only join."""
    return SqlAgent(llm=llm, executor=executor, system_prompt=build_sql_prompt(schema),
                    files=list(FILES), max_steps=config.MAX_STEPS,
                    max_result_chars=config.MAX_RESULT_CHARS)


def run_question(agent: SqlAgent, scenario: SqlScenario, answers: dict) -> list[dict]:
    """Ask one question, collect what happened, and score it. Returned like
    the cleaning eval's run_once(): [step result, {"transcript": ...}], so
    run_eval.print_summary() can total it."""
    start = time.perf_counter()
    replies, notices, queries, results = [], [], [], []
    peak_tokens, end = 0, "error"
    try:
        for event in agent.send(scenario.question):
            match event:
                case ModelReply(text=text):
                    replies.append(text)
                    usage = getattr(agent.llm, "last_usage", {})   # fake models have none
                    peak_tokens = max(peak_tokens, usage.get("prompt_tokens", 0))
                case CodeResult(code=sql, result=r, model_text=shown):
                    queries.append({"sql": sql, "status": r.get("status"), "error": r.get("error")})
                    if r.get("status") == "ok":
                        results.append(shown)
                case Notice(text=text):
                    notices.append(text)
                case TurnDone(reason=reason):
                    end = reason
        crashed = None
    except Exception as e:   # one broken run must not stop the whole evaluation
        crashed = f"{type(e).__name__}: {e}"

    final_reply = replies[-1] if replies else ""
    run = SqlRun(question=scenario.question, final_reply=final_reply, queries=queries, results=results)
    checks = score_facts(scenario.checks(answers), final_reply, run)
    return [{
        "step": "question",
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "probe_errors": {},
        "seconds": round(time.perf_counter() - start, 1),
        "executions": len(queries),
        "errors": sum(q["status"] != "ok" for q in queries),
        "end": end,
        "crashed": crashed,
        "notices": notices,
        "peak_prompt_tokens": peak_tokens,
        "queries": queries,
        "final_reply": final_reply,
    }, {"transcript": agent.history}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on the SQL questions.")
    parser.add_argument("--models", nargs="+", default=[config.MODEL])
    parser.add_argument("--scenarios", nargs="+", default=[s.name for s in SQL_SCENARIOS],
                        choices=[s.name for s in SQL_SCENARIOS])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    scenarios = [SQL_BY_NAME[name] for name in args.scenarios]

    folder = config.OUTPUT_DIR / "clean" / "sql_eval"
    missing = [f for f in FILES if not (folder / f.rsplit("/", 1)[1]).exists()]
    if missing:
        print(f"Missing tables: {', '.join(missing)}\n"
              "Generate them first: uv run python evals/make_datasets.py")
        return

    executor = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
    executor.check_sandboxed()
    schema = executor.schema(FILES)
    if schema.get("error") or schema.get("problems") or len(schema.get("tables", [])) != len(FILES):
        print(f"The tables could not be loaded: {schema}")
        return
    answers = compute_answers(executor, scenarios)
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
                print(f"\n{model}  sql-{scenario.name}  run {i}/{args.runs}", flush=True)
                results = run_question(make_agent(llm, executor, schema), scenario, answers[scenario.name])
                r = results[0]
                flag = "PASS" if r["passed"] == r["total"] else "    "
                print(f"  {flag} {r['passed']:>2}/{r['total']:<3}{r['seconds']:>6.1f}s  "
                      f"{r['executions']} queries, {r['errors']} errors  {r['peak_prompt_tokens']:>6,} tok"
                      + (f"  CRASHED: {r['crashed']}" if r["crashed"] else ""), flush=True)
                for check_name, ok in r["checks"].items():
                    if not ok:
                        print(f"         missed: {check_name}", flush=True)
                runs.append(results)
                name = f"{stamp}_{model.replace(':', '-')}_sql-{scenario.name}_run{i}"
                (out_dir / f"{name}.json").write_text(
                    json.dumps({"question": scenario.question, "answers": answers[scenario.name],
                                "results": results}, indent=2, default=str), encoding="utf-8")
            all_results[model][scenario.name] = runs

    print_summary(all_results, scenarios)
    print(f"Full transcripts: {out_dir}")


if __name__ == "__main__":
    main()
