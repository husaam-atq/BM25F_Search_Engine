"""Rank-based lexical+dense fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .types import RankedDocument, Run


def reciprocal_rank_fusion(
    rankings: Iterable[list[RankedDocument]],
    rrf_k: int = 60,
    top_k: int = 1000,
) -> list[RankedDocument]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if top_k <= 0:
        return []
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            scores[item.doc_id] += 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        RankedDocument(doc_id, float(score))
        for doc_id, score in ordered[:top_k]
    ]


def fuse_runs(
    runs: list[Run],
    rrf_k: int = 60,
    top_k: int = 1000,
) -> Run:
    if not runs:
        return {}
    query_ids = set(runs[0])
    if any(set(run) != query_ids for run in runs[1:]):
        raise ValueError("All runs must contain the same query IDs")
    return {
        query_id: reciprocal_rank_fusion(
            [run[query_id] for run in runs],
            rrf_k=rrf_k,
            top_k=top_k,
        )
        for query_id in sorted(query_ids)
    }

