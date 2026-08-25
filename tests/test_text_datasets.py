import json

import pytest

from research_extension.datasets import load_corpus, load_qrels, load_queries
from research_extension.text import tokenise, tokenise_with_positions


def test_tokenisation_is_deterministic_and_unicode_aware():
    text = "Café-based BM25F costs 12.5 ms."
    expected = ["café", "based", "bm25f", "costs", "12.5", "ms"]
    assert tokenise(text) == expected
    assert tokenise(text) == expected
    assert tokenise_with_positions("A B A") == [("a", 0), ("b", 1), ("a", 2)]


def test_beir_field_extraction_and_qrels(synthetic_dataset):
    documents = load_corpus(synthetic_dataset)
    queries = load_queries(synthetic_dataset)
    qrels = load_qrels(synthetic_dataset, "test")
    assert documents["d1"].title == "Quantum Catalyst"
    assert documents["d1"].text == "reaction improves yield"
    assert documents["d5"].title == documents["d5"].text == ""
    assert queries == {"q1": "quantum catalyst", "q2": "red fox"}
    assert qrels["q1"] == {"d1": 2, "d2": 1}


def test_duplicate_document_ids_are_rejected(tmp_path):
    rows = [
        {"_id": "same", "title": "first", "text": ""},
        {"_id": "same", "title": "second", "text": ""},
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate document ID"):
        load_corpus(tmp_path)
