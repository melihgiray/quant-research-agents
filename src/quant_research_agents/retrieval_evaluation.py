"""Transparent retrieval metrics on a visible development set; no model calls."""

from statistics import mean

from pydantic import Field

from .evidence import content_digest
from .models import StrictModel
from .retrieval import EvidenceIndex


class RetrievalCase(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=1000)
    relevant_ids: list[str]


def evaluate_retrieval(cases: list[dict], index: EvidenceIndex | None = None, k: int = 3) -> dict:
    index = index if index is not None else EvidenceIndex()
    cases = [RetrievalCase.model_validate(case).model_dump() for case in cases]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("retrieval cases must be nonempty with unique IDs")
    if not 1 <= k <= 5:
        raise ValueError("k must be between 1 and 5")
    known = {d["id"] for d in index.documents}
    results = []
    for case in cases:
        expected = set(case["relevant_ids"])
        if not expected <= known:
            raise ValueError("relevance labels must refer to the evaluated corpus")
        ids = [hit["id"] for hit in index.search(case["query"], limit=k)]
        matches = expected.intersection(ids)
        rank = next((i for i, doc_id in enumerate(ids, start=1) if doc_id in expected), None)
        results.append(
            {
                **case,
                "retrieved_ids": ids,
                "hit": bool(matches) if expected else None,
                "recall": len(matches) / len(expected) if expected else None,
                "reciprocal_rank": (1 / rank if rank else 0.0) if expected else None,
                "correct_abstention": not ids if not expected else None,
            }
        )
    answerable = [r for r in results if r["relevant_ids"]]
    unanswerable = [r for r in results if not r["relevant_ids"]]
    return {
        "mode": "retrieval-only",
        "k": k,
        "corpus_sha256": index.corpus_sha256,
        "cases_sha256": content_digest(cases),
        "cases": len(cases),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "hit_rate_at_k": mean(r["hit"] for r in answerable) if answerable else None,
        "mean_recall_at_k": mean(r["recall"] for r in answerable) if answerable else None,
        "mrr_at_k": mean(r["reciprocal_rank"] for r in answerable) if answerable else None,
        "abstention_accuracy": mean(r["correct_abstention"] for r in unanswerable)
        if unanswerable
        else None,
        "results": results,
        "limits": "Visible development cases over a small corpus. These metrics do not measure generation quality or out-of-domain retrieval quality.",
    }
