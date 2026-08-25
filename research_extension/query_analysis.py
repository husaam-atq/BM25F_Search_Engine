"""Predeclared, objectively derived query slices and win/loss summaries."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from .lexical import LexicalIndex
from .text import tokenise
from .types import Documents, Run, unique_doc_ids


def ranking_jaccard(first: list[str], second: list[str], depth: int = 10) -> float:
    left = set(first[:depth])
    right = set(second[:depth])
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def derive_query_labels(
    query: str,
    index: LexicalIndex,
    lexical_ranking: list[str],
    dense_ranking: list[str],
    bm25f_ranking: list[str] | None = None,
    phrase_ranking: list[str] | None = None,
) -> list[str]:
    terms = tokenise(query)
    labels: list[str] = []
    if len(terms) <= 5:
        labels.append("short_<=5_tokens")
    elif len(terms) >= 10:
        labels.append("long_>=10_tokens")
    else:
        labels.append("medium_6-9_tokens")

    raw_words = query.split()
    entity_like = sum(
        word[:1].isupper() or (len(word) > 1 and word.isupper())
        for word in raw_words[1:]
    )
    if raw_words and entity_like / len(raw_words) >= 0.20:
        labels.append("named_entity_heavy")

    known_ratios = [
        index.document_frequency(term) / max(index.document_count, 1)
        for term in terms
    ]
    if known_ratios and min(known_ratios) <= 0.01:
        labels.append("rare_or_oov_term")

    if ranking_jaccard(lexical_ranking, dense_ranking, depth=10) <= 0.20:
        labels.append("strong_lexical_dense_disagreement")

    if (
        bm25f_ranking is not None
        and phrase_ranking is not None
        and bm25f_ranking[:10] != phrase_ranking[:10]
    ):
        labels.append("phrase_proximity_sensitive")
    return labels


def analyse_queries(
    queries: dict[str, str],
    index: LexicalIndex,
    runs: dict[str, Run],
    per_query: dict[str, dict[str, dict[str, float]]],
    baseline_name: str = "bm25f_phrase_proximity",
    candidate_name: str = "hybrid_rrf",
) -> dict[str, object]:
    required = {"bm25f", "bm25f_phrase_proximity", "dense", baseline_name, candidate_name}
    missing = required - set(runs)
    if missing:
        raise ValueError(f"Missing runs for query analysis: {sorted(missing)}")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    examples: list[dict[str, object]] = []

    for query_id, query in queries.items():
        lexical = unique_doc_ids(runs[baseline_name][query_id])
        dense = unique_doc_ids(runs["dense"][query_id])
        labels = derive_query_labels(
            query,
            index,
            lexical,
            dense,
            bm25f_ranking=unique_doc_ids(runs["bm25f"][query_id]),
            phrase_ranking=unique_doc_ids(runs["bm25f_phrase_proximity"][query_id]),
        )
        delta = (
            per_query[candidate_name][query_id]["AP"]
            - per_query[baseline_name][query_id]["AP"]
        )
        row = {
            "query_id": query_id,
            "query": query,
            "labels": labels,
            "ap_delta": delta,
            "lexical_dense_top10_jaccard": ranking_jaccard(lexical, dense, 10),
        }
        examples.append(row)
        for label in labels:
            grouped[label].append(row)

    categories: dict[str, object] = {}
    for label, rows in sorted(grouped.items()):
        deltas = [float(row["ap_delta"]) for row in rows]
        categories[label] = {
            "queries": len(rows),
            "mean_ap_delta": fmean(deltas),
            "wins": sum(delta > 1e-12 for delta in deltas),
            "losses": sum(delta < -1e-12 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        }

    ranked_examples = sorted(
        examples,
        key=lambda row: (-abs(float(row["ap_delta"])), str(row["query_id"])),
    )
    return {
        "category_protocol": {
            "short": "<=5 lexical tokens",
            "long": ">=10 lexical tokens",
            "named_entity_heavy": ">=20% of whitespace tokens after the first are capitalised/acronyms",
            "rare_or_oov": "minimum query-term document-frequency ratio <=1%",
            "ranking_disagreement": "top-10 set Jaccard <=0.20",
            "phrase_sensitive": "BM25F and BM25F+phrase/proximity top-10 order differs",
        },
        "categories": categories,
        "representative_largest_absolute_deltas": ranked_examples[:20],
    }

