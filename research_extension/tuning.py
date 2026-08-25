"""Small, predeclared lexical grid evaluated only on the primary train split."""

from __future__ import annotations

from dataclasses import asdict, replace

from .config import LexicalConfig
from .evaluation import evaluate_run
from .lexical import LexicalIndex, retrieve_all
from .types import Qrels, Queries, RankedDocument


def select_lexical_config(
    index: LexicalIndex,
    queries: Queries,
    qrels: Qrels,
    base: LexicalConfig | None = None,
    top_k: int = 1000,
) -> tuple[LexicalConfig, dict[str, object]]:
    """Tune title weight, then dependence bonuses, using train nDCG@10."""
    base = base or LexicalConfig()
    title_rows: list[dict[str, object]] = []
    for title_weight in (1.0, 2.0, 3.0, 5.0):
        candidate = replace(base, title_weight=title_weight)
        run = retrieve_all(index, queries, "bm25f", candidate, top_k)
        aggregate, _ = evaluate_run(qrels, run)
        title_rows.append(
            {
                "title_weight": title_weight,
                "nDCG@10": aggregate["nDCG@10"],
                "MAP": aggregate["MAP"],
            }
        )
    best_title = sorted(
        title_rows,
        key=lambda row: (-float(row["nDCG@10"]), float(row["title_weight"])),
    )[0]
    selected = replace(base, title_weight=float(best_title["title_weight"]))

    dependence_grid = (
        (0.0, 0.0),
        (0.10, 0.00),
        (0.10, 0.05),
        (0.25, 0.05),
        (0.25, 0.10),
        (0.50, 0.10),
        (0.50, 0.20),
    )
    component_cache = {
        query_id: index.score_components(query, selected)
        for query_id, query in queries.items()
    }
    dependence_rows: list[dict[str, object]] = []
    for phrase_bonus, proximity_bonus in dependence_grid:
        candidate = replace(
            selected,
            phrase_bonus=phrase_bonus,
            proximity_bonus=proximity_bonus,
        )
        run = {}
        for query_id, components in component_cache.items():
            scored = [
                RankedDocument(
                    doc_id,
                    lexical_score
                    + phrase_bonus * phrase_feature
                    + proximity_bonus * proximity_feature,
                )
                for doc_id, lexical_score, phrase_feature, proximity_feature
                in components
            ]
            scored.sort(key=lambda item: (-item.score, item.doc_id))
            run[query_id] = scored[:top_k]
        aggregate, _ = evaluate_run(qrels, run)
        dependence_rows.append(
            {
                "phrase_bonus": phrase_bonus,
                "proximity_bonus": proximity_bonus,
                "nDCG@10": aggregate["nDCG@10"],
                "MAP": aggregate["MAP"],
            }
        )
    # (0, 0) is retained as the BM25F control, not as a selectable
    # phrase+proximity implementation. The extension explicitly carries one
    # active dependence setting into the untouched test splits.
    active_dependence_rows = [
        row
        for row in dependence_rows
        if float(row["phrase_bonus"]) > 0
        and float(row["proximity_bonus"]) > 0
    ]
    best_dependence = sorted(
        active_dependence_rows,
        key=lambda row: (
            -float(row["nDCG@10"]),
            float(row["phrase_bonus"]) + float(row["proximity_bonus"]),
            float(row["phrase_bonus"]),
        ),
    )[0]
    selected = replace(
        selected,
        phrase_bonus=float(best_dependence["phrase_bonus"]),
        proximity_bonus=float(best_dependence["proximity_bonus"]),
    )
    return selected, {
        "selection_split": "primary benchmark train",
        "selection_metric": "nDCG@10",
        "dependence_constraint": (
            "both phrase and proximity coefficients must be positive; "
            "(0,0) is the BM25F ablation control"
        ),
        "tie_break": "lower title weight / smaller total active dependence bonus",
        "title_weight_grid": title_rows,
        "dependence_grid": dependence_rows,
        "selected": asdict(selected),
    }
