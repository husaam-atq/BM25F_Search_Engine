"""Established IR metrics with query-level output and candidate recall."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable

from .types import Qrels, Run, unique_doc_ids

DEFAULT_DEPTHS = (10, 50, 100, 1000)


def _relevant(judgments: dict[str, int]) -> set[str]:
    return {doc_id for doc_id, grade in judgments.items() if grade > 0}


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranking[:k]) / k


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranking[:k]) / len(relevant)


def average_precision(ranking: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def r_precision(ranking: list[str], relevant: set[str]) -> float:
    return precision_at_k(ranking, relevant, len(relevant)) if relevant else 0.0


def ndcg_at_k(
    ranking: list[str], judgments: dict[str, int], k: int
) -> float:
    if k <= 0:
        return 0.0
    gains = [
        (2 ** max(0, judgments.get(doc_id, 0)) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranking[:k], start=1)
    ]
    ideal_grades = sorted(
        (grade for grade in judgments.values() if grade > 0), reverse=True
    )[:k]
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return sum(gains) / ideal if ideal else 0.0


def evaluate_query(
    ranking: list[str],
    judgments: dict[str, int],
    depths: Iterable[int] = DEFAULT_DEPTHS,
) -> dict[str, float]:
    ranking = list(dict.fromkeys(ranking))
    relevant = _relevant(judgments)
    output = {
        "AP": average_precision(ranking, relevant),
        "P@10": precision_at_k(ranking, relevant, 10),
        "nDCG@10": ndcg_at_k(ranking, judgments, 10),
        "MRR": reciprocal_rank(ranking, relevant),
        "R-Precision": r_precision(ranking, relevant),
    }
    for depth in depths:
        output[f"Recall@{depth}"] = recall_at_k(ranking, relevant, depth)
    return output


def evaluate_run(
    qrels: Qrels,
    run: Run,
    depths: Iterable[int] = DEFAULT_DEPTHS,
) -> tuple[dict[str, float | int], dict[str, dict[str, float]]]:
    per_query: dict[str, dict[str, float]] = {}
    for query_id, judgments in qrels.items():
        ranking = unique_doc_ids(run.get(query_id, []))
        per_query[query_id] = evaluate_query(ranking, judgments, depths)
    metric_names = sorted({name for row in per_query.values() for name in row})
    aggregate: dict[str, float | int] = {"queries": len(per_query)}
    for metric in metric_names:
        values = [row[metric] for row in per_query.values()]
        aggregate["MAP" if metric == "AP" else metric] = (
            fmean(values) if values else 0.0
        )
    return aggregate, per_query

