import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .comparison import compare_experiments
from .diagnostics import diagnose
from .evaluation import evaluate
from .evaluation_summary import summarize_evaluation
from .models import RunLimits
from .retrieval_evaluation import evaluate_retrieval
from .runtime import DEFAULT_REQUEST, run_research
from .store import ExperimentStore
from .verification import verify_report


def main() -> int:
    load_dotenv(Path.cwd() / ".env")
    parser = argparse.ArgumentParser(description="Reproducible research with Google ADK")
    parser.add_argument(
        "--workspace", type=Path, default=Path(os.getenv("QUANT_WORKSPACE", "workspace"))
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run scripted ADK and real tools without API keys")
    demo.add_argument("--architecture", choices=["multi", "single"], default="multi")
    live = commands.add_parser("run", help="Run Gemini; requires local credentials")
    live.add_argument("prompt", nargs="?", default=DEFAULT_REQUEST)
    live.add_argument("--architecture", choices=["multi", "single"], default="multi")
    commands.add_parser("history", help="Read recent experiment records from SQL")
    commands.add_parser(
        "doctor", help="Check local setup without network calls or credential disclosure"
    )
    evaluation = commands.add_parser("evaluate", help="Run the live development evaluation set")
    evaluation.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="Acknowledge that this makes billable model calls",
    )
    evaluation.add_argument("--cases", type=Path, default=Path("evals/cases.json"))
    evaluation.add_argument("--trials", type=int, default=1, help="Repetitions per case, 1 to 20")
    evaluation.add_argument("--seed", type=int, default=42, help="Execution-order shuffle seed")
    for options_parser in (demo, live, evaluation):
        options_parser.add_argument("--max-llm-calls", type=int, default=16)
        options_parser.add_argument("--timeout-seconds", type=float, default=180)
        options_parser.add_argument(
            "--no-review-repair",
            action="store_true",
            help="Disable the correction attempt for an ablation",
        )
    retrieval = commands.add_parser(
        "evaluate-retrieval", help="Score retrieval without model calls"
    )
    retrieval.add_argument("--cases", type=Path, default=Path("evals/retrieval_cases.json"))
    retrieval.add_argument("--k", type=int, default=3)
    retrieval.add_argument("--output", type=Path)
    verify = commands.add_parser(
        "verify", help="Verify a report and replay its synthetic experiment offline"
    )
    verify.add_argument("report", type=Path)
    compare = commands.add_parser(
        "compare", help="Compare saved experiments with matching datasets and windows"
    )
    compare.add_argument("left_id")
    compare.add_argument("right_id")
    summarize = commands.add_parser(
        "summarize", help="Summarize live evaluations without hiding failures"
    )
    summarize.add_argument("evaluation", type=Path)
    summarize.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        limits = (
            RunLimits(
                max_llm_calls=args.max_llm_calls,
                timeout_seconds=args.timeout_seconds,
                review_repair=not args.no_review_repair,
            )
            if args.command in {"demo", "run", "evaluate"}
            else None
        )
        if args.command == "summarize":
            result = summarize_evaluation(json.loads(args.evaluation.read_text()))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "compare":
            result = compare_experiments(
                ExperimentStore(args.workspace, os.getenv("DATABASE_URL")),
                args.left_id,
                args.right_id,
            )
        elif args.command == "verify":
            result = verify_report(args.report)
        elif args.command == "evaluate-retrieval":
            result = evaluate_retrieval(json.loads(args.cases.read_text()), k=args.k)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "doctor":
            result = diagnose(args.workspace)
        elif args.command == "history":
            result = ExperimentStore(args.workspace, os.getenv("DATABASE_URL")).recent()
        elif args.command == "evaluate":
            if not (
                os.getenv("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
            ):
                parser.error("Set GOOGLE_API_KEY in .env before running live evaluations.")
            result = {
                "evaluation_file": str(
                    asyncio.run(
                        evaluate(
                            args.cases,
                            args.workspace,
                            trials=args.trials,
                            seed=args.seed,
                            limits=limits,
                        )
                    )
                )
            }
        else:
            result = asyncio.run(
                run_research(
                    DEFAULT_REQUEST if args.command == "demo" else args.prompt,
                    args.workspace,
                    args.architecture,
                    "scripted" if args.command == "demo" else "live",
                    database_url=os.getenv("DATABASE_URL"),
                    limits=limits,
                )
            )
        print(json.dumps(result, indent=2))
        return 0
    except ValueError as error:
        parser.exit(2, f"Configuration or validation error: {error}\n")
    except Exception as error:
        # Provider errors may contain URLs or credentials; write safe failure metadata.
        parser.exit(
            1,
            f"Run failed ({type(error).__name__}). Inspect workspace/runs/*/run.json "
            "and trace.jsonl. Check provider credentials and model access.\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
