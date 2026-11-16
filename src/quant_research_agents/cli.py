import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .diagnostics import diagnose
from .evaluation import evaluate
from .retrieval_evaluation import evaluate_retrieval
from .runtime import DEFAULT_REQUEST, run_research
from .store import ExperimentStore


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
    retrieval = commands.add_parser(
        "evaluate-retrieval", help="Score retrieval without model calls"
    )
    retrieval.add_argument("--cases", type=Path, default=Path("evals/retrieval_cases.json"))
    retrieval.add_argument("--k", type=int, default=3)
    retrieval.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "evaluate-retrieval":
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
                        evaluate(args.cases, args.workspace, trials=args.trials, seed=args.seed)
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
