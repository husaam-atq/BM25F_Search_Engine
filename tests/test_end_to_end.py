from research_extension.datasets import load_corpus, load_qrels, load_queries
from research_extension.evaluation import evaluate_run
from research_extension.hybrid import fuse_runs
from research_extension.lexical import LexicalIndex, retrieve_all


def test_synthetic_end_to_end_without_private_data_or_models(synthetic_dataset):
    documents = load_corpus(synthetic_dataset)
    queries = load_queries(synthetic_dataset)
    qrels = load_qrels(synthetic_dataset, "test")
    index = LexicalIndex.build(documents)
    bm25 = retrieve_all(index, queries, "bm25", top_k=100)
    bm25f = retrieve_all(index, queries, "bm25f", top_k=100)
    hybrid = fuse_runs([bm25, bm25f], top_k=100)
    aggregate, rows = evaluate_run(qrels, hybrid)
    assert aggregate["queries"] == 2
    assert aggregate["MAP"] > 0
    assert set(rows) == set(queries)
