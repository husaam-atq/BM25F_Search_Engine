import pytest

from research_extension.evaluation import (
    average_precision,
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from research_extension.significance import paired_comparison
from research_extension.types import RankedDocument


def test_metric_reference_case():
    ranking = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}
    assert average_precision(ranking, relevant) == pytest.approx((1 + 2 / 3) / 2)
    assert precision_at_k(ranking, relevant, 10) == pytest.approx(0.2)
    assert recall_at_k(ranking, relevant, 2) == pytest.approx(0.5)
    assert ndcg_at_k(ranking, {"d1": 2, "d3": 1}, 3) > 0.9


def test_run_evaluation_deduplicates_and_handles_missing_queries():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    run = {
        "q1": [
            RankedDocument("d1", 2),
            RankedDocument("d1", 1),
        ]
    }
    aggregate, rows = evaluate_run(qrels, run)
    assert aggregate["queries"] == 2
    assert aggregate["MAP"] == pytest.approx(0.5)
    assert rows["q1"]["Recall@100"] == 1.0
    assert rows["q2"]["Recall@100"] == 0.0


def test_paired_analysis_is_seeded_and_reports_wins():
    baseline = {f"q{i}": {"AP": 0.1 * i} for i in range(1, 6)}
    candidate = {f"q{i}": {"AP": 0.1 * i + 0.05} for i in range(1, 6)}
    first = paired_comparison(
        baseline, candidate, bootstrap_samples=500, permutation_samples=1000
    )
    second = paired_comparison(
        baseline, candidate, bootstrap_samples=500, permutation_samples=1000
    )
    assert first == second
    assert first["wins"] == 5
    assert first["mean_difference"] == pytest.approx(0.05)
