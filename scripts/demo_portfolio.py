"""Run the complete offline walkthrough; no credentials or provider calls."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from quant_research_agents.comparison import compare_experiments
from quant_research_agents.experiment import run_experiment
from quant_research_agents.models import ExperimentConfig
from quant_research_agents.retrieval_evaluation import evaluate_retrieval
from quant_research_agents.runtime import DEFAULT_REQUEST, run_research
from quant_research_agents.store import ExperimentStore
from quant_research_agents.verification import verify_report


async def walkthrough(workspace: Path, cases: Path) -> Path:
    destination = workspace / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    results = {}
    for architecture in ("single", "multi"):
        run = await run_research(
            DEFAULT_REQUEST, destination / architecture, architecture=architecture, mode="scripted"
        )
        results[architecture] = {"run": run, "verification": verify_report(Path(run["json"]))}
    retrieval = evaluate_retrieval(json.loads(cases.read_text()))
    (destination / "retrieval.json").write_text(json.dumps(retrieval, indent=2, allow_nan=False))
    store = ExperimentStore(destination / "multi")
    higher_cost = run_experiment(ExperimentConfig(cost_bps=20))
    store.save(higher_cost)
    comparison = compare_experiments(
        store, results["multi"]["run"]["experiment_id"], higher_cost["experiment_id"]
    )
    manifest = {
        "mode": "scripted-portfolio-walkthrough",
        "live_model_calls": 0,
        "architectures": results,
        "retrieval_file": str((destination / "retrieval.json").resolve()),
        "cost_comparison": comparison,
        "comparison_origin": "The 20-bps experiment is a direct Python fixture, not a model-planned action.",
        "limits": "Synthetic data and scripted model responses exercise integration. This is not a live agent quality benchmark or evidence of investment performance.",
    }
    path = destination / "walkthrough.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspace/portfolio-demo"))
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evals/retrieval_cases.json",
    )
    arguments = parser.parse_args()
    path = asyncio.run(walkthrough(arguments.workspace, arguments.cases))
    print(json.dumps({"walkthrough": str(path.resolve()), "live_model_calls": 0}, indent=2))
