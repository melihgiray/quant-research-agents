import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .evaluation import evaluate
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
    args = parser.parse_args()
    try:
        if args.command == "history":
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
