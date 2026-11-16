import pytest

from quant_research_agents.retrieval import EvidenceIndex
from quant_research_agents.retrieval_evaluation import evaluate_retrieval


def test_retrieval_metrics_have_correct_denominators():
    index = EvidenceIndex(
        [
            {"id": key, "title": word, "text": word, "source_url": "https://example.com"}
            for key, word in (("a", "alpha"), ("b", "beta"))
        ]
    )
    result = evaluate_retrieval(
        [
            {"id": "one", "query": "alpha", "relevant_ids": ["a", "b"]},
            {"id": "two", "query": "void", "relevant_ids": ["b"]},
            {"id": "three", "query": "unknown", "relevant_ids": []},
        ],
        index=index,
        k=1,
    )
    assert result["hit_rate_at_k"] == 0.5
    assert result["mean_recall_at_k"] == 0.25
    assert result["mrr_at_k"] == 0.5
    assert result["abstention_accuracy"] == 1


def test_missing_metric_group_stays_null_and_invalid_labels_fail():
    case = {"id": "one", "query": "momentum", "relevant_ids": ["momentum"]}
    assert evaluate_retrieval([case])["abstention_accuracy"] is None
    with pytest.raises(ValueError, match="corpus"):
        evaluate_retrieval([{**case, "relevant_ids": ["invented"]}])
    with pytest.raises(ValueError, match="unique"):
        evaluate_retrieval([case, case])
