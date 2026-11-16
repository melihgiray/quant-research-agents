import pytest

from quant_research_agents.retrieval import EvidenceIndex


def note(**updates):
    return {
        "id": "one",
        "title": "Momentum",
        "text": "Momentum method",
        "source_url": "https://example.com/method",
        **updates,
    }


def test_corpus_fingerprint_is_order_independent_and_content_sensitive():
    a, b = note(), note(id="two", text="Volatility method")
    assert EvidenceIndex([a, b]).corpus_sha256 == EvidenceIndex([b, a]).corpus_sha256
    assert EvidenceIndex([a]).corpus_sha256 != EvidenceIndex([note(text="Changed")]).corpus_sha256


def test_returned_evidence_cannot_mutate_index():
    index = EvidenceIndex([note()])
    hit = index.search("momentum")[0]
    assert len(hit["content_sha256"]) == 64
    hit["text"] = "overwritten"
    index.documents[0]["id"] = "changed"
    assert index.search("momentum")[0]["text"] == "Momentum method"
    assert index.search("momentum")[0]["id"] == "one"


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///secret",
        "https://name:password@example.com",
        "local://docs/../secret",
    ],
)
def test_corpus_rejects_unsafe_source_references(url):
    with pytest.raises(ValueError):
        EvidenceIndex([note(source_url=url)])


def test_duplicate_document_ids_rejected_and_empty_index_abstains():
    with pytest.raises(ValueError, match="unique"):
        EvidenceIndex([note(), note()])
    assert EvidenceIndex([]).search("momentum") == []
