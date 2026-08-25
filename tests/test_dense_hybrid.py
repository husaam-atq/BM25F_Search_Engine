import numpy as np
import pytest

from research_extension.dense import DenseIndex
from research_extension.hybrid import reciprocal_rank_fusion
from research_extension.types import RankedDocument


def test_dense_ranking_direction_and_round_trip(tmp_path):
    index = DenseIndex(
        ["positive", "orthogonal", "negative"],
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
        {"model_id": "fake"},
    )
    results = index.search_embedding(np.asarray([1.0, 0.0]), top_k=3)
    assert [item.doc_id for item in results] == [
        "positive",
        "orthogonal",
        "negative",
    ]
    path = tmp_path / "dense.npz"
    index.save(path)
    restored = DenseIndex.load(path)
    assert restored.doc_ids == index.doc_ids
    assert np.allclose(restored.embeddings, index.embeddings)


def test_zero_embeddings_are_rejected():
    with pytest.raises(ValueError, match="Zero-length"):
        DenseIndex(["bad"], np.zeros((1, 2), dtype=np.float32))


def test_rrf_scores_and_duplicate_handling():
    lexical = [
        RankedDocument("a", 10),
        RankedDocument("a", 9),
        RankedDocument("b", 8),
    ]
    dense = [RankedDocument("b", 1), RankedDocument("c", 0.5)]
    fused = reciprocal_rank_fusion([lexical, dense], rrf_k=60)
    scores = {item.doc_id: item.score for item in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 63 + 1 / 61)
    assert len(scores) == 3
    assert fused[0].doc_id == "b"
