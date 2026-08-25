import pytest

from research_extension.reranking import rerank
from research_extension.types import Document, RankedDocument


class ReverseScorer:
    def score(self, query, documents):
        return list(range(len(documents)))

    def metadata(self):
        return {"model_id": "fake"}


def test_reranking_preserves_candidate_set_and_tail():
    candidates = [
        RankedDocument("a", 3),
        RankedDocument("b", 2),
        RankedDocument("c", 1),
    ]
    documents = {
        doc_id: Document(doc_id, text=doc_id) for doc_id in ("a", "b", "c")
    }
    output = rerank("query", candidates, documents, ReverseScorer(), depth=2)
    assert [item.doc_id for item in output] == ["b", "a", "c"]
    assert {item.doc_id for item in output} == {"a", "b", "c"}


def test_reranker_cannot_introduce_unseen_documents():
    candidates = [RankedDocument("missing", 1)]
    with pytest.raises(KeyError):
        rerank("query", candidates, {}, ReverseScorer(), depth=1)


def test_duplicate_candidates_are_rejected():
    candidates = [RankedDocument("a", 2), RankedDocument("a", 1)]
    with pytest.raises(ValueError, match="duplicate"):
        rerank("query", candidates, {"a": Document("a")}, ReverseScorer())
