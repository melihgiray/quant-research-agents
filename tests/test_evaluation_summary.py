import pytest

from quant_research_agents.evaluation_summary import summarize_evaluation


def test_summary_counts_failures_without_zero_filling_usage():
    result = summarize_evaluation(
        {
            "mode": "live",
            "status": "interrupted",
            "scheduled_runs": 8,
            "results": [
                {
                    "architecture": "multi",
                    "automated_pass": True,
                    "run": {
                        "mode": "live",
                        "status": "completed",
                        "elapsed_seconds": 2,
                        "usage_observed": True,
                        "total_tokens": 100,
                        "review_repair_attempts": 1,
                    },
                },
                {
                    "architecture": "multi",
                    "automated_pass": False,
                    "run": {
                        "mode": "live",
                        "status": "failed",
                        "elapsed_seconds": 10,
                        "usage_observed": False,
                        "total_tokens": None,
                        "review_repair_attempts": 0,
                    },
                },
                {"architecture": "multi", "automated_pass": False, "run": None},
            ],
        }
    )
    group = result["architectures"]["multi"]
    assert group["automated_pass_rate"] == pytest.approx(1 / 3)
    assert group["median_seconds"] == 6
    assert group["p95_seconds"] == 10
    assert group["observed_total_tokens"] == 100
    assert group["usage_coverage"] == pytest.approx(1 / 3)
    assert group["observed_cost_usd"] is None
    assert group["repair_attempt_rate"] == 0.5
    assert result["evaluation_status"] == "interrupted"


def test_summary_rejects_scripted_results_in_live_envelope():
    with pytest.raises(ValueError, match="scripted"):
        summarize_evaluation(
            {"mode": "live", "results": [{"architecture": "single", "run": {"mode": "scripted"}}]}
        )


def test_empty_evaluation_has_no_invented_scores():
    assert summarize_evaluation({"mode": "live", "results": []})["architectures"] == {}
