from dataclasses import replace

import pytest

from research_extension.config import LexicalConfig
from research_extension.lexical import LexicalIndex
from research_extension.types import Document


def _scores(results):
    return {item.doc_id: item.score for item in results}


def test_bm25_term_frequency_is_monotonic():
    index = LexicalIndex.build(
        [
            Document("once", text="signal filler filler"),
            Document("twice", text="signal signal filler"),
        ]
    )
    results = index.search("signal", variant="bm25")
    assert results[0].doc_id == "twice"
    assert results[0].score > results[1].score


def test_bm25f_title_weighting(synthetic_index):
    config = replace(LexicalConfig(), title_weight=5.0)
    results = synthetic_index.search("quantum catalyst", "bm25f", config)
    assert results[0].doc_id == "d1"


def test_phrase_and_proximity_bonuses(synthetic_index):
    config = replace(
        LexicalConfig(),
        phrase_bonus=2.0,
        proximity_bonus=1.0,
        proximity_window=4,
    )
    bm25f = _scores(synthetic_index.search("red fox", "bm25f", config))
    dependent = _scores(
        synthetic_index.search("red fox", "bm25f_phrase_proximity", config)
    )
    assert dependent["d3"] - bm25f["d3"] > dependent["d4"] - bm25f["d4"]
    assert dependent["d3"] > dependent["d4"]


def test_repeated_query_terms_are_not_discarded(synthetic_index):
    single = _scores(synthetic_index.search("quantum catalyst", "bm25f"))
    repeated = _scores(
        synthetic_index.search("quantum quantum catalyst", "bm25f")
    )
    assert repeated["d1"] > single["d1"]
    assert repeated["d2"] > single["d2"]


def test_missing_fields_and_empty_query_are_safe(synthetic_index):
    assert synthetic_index.search("", "bm25f") == []
    results = synthetic_index.search("reaction", "bm25f")
    assert {item.doc_id for item in results} == {"d1", "d2"}


def test_unknown_variant_rejected(synthetic_index):
    with pytest.raises(ValueError):
        synthetic_index.search("red", "unsupported")
