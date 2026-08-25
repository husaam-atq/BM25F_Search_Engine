"""Generate the committed extension figures from recorded experiment artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SYSTEMS = [
    "bm25",
    "bm25f",
    "bm25f_phrase_proximity",
    "dense",
    "hybrid_rrf",
    "hybrid_rerank",
]
LABELS = {
    "bm25": "BM25",
    "bm25f": "BM25F",
    "bm25f_phrase_proximity": "BM25F + P/P",
    "dense": "E5 dense",
    "hybrid_rrf": "RRF hybrid",
    "lexical_rerank": "Lexical + CE",
    "dense_rerank": "Dense + CE",
    "hybrid_rerank": "Hybrid + CE",
}
COLORS = {
    "bm25": "#5B8FF9",
    "bm25f": "#61DDAA",
    "bm25f_phrase_proximity": "#65789B",
    "dense": "#F6BD16",
    "hybrid_rrf": "#E8684A",
    "hybrid_rerank": "#9270CA",
}


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    artifact_dir: str | Path = "artifacts",
    output_dir: str | Path = "reports/figures",
) -> dict[str, object]:
    _style()
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        name: _load(artifact_dir / name / "results.json")
        for name in ("scifact", "nfcorpus")
    }
    per_query = {
        name: _load(artifact_dir / name / "per_query.json")
        for name in datasets
    }

    # Retrieval quality comparison.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, (dataset_name, payload) in zip(axes, datasets.items()):
        values = [payload["aggregates"][system]["nDCG@10"] for system in SYSTEMS]
        axis.bar(
            range(len(SYSTEMS)),
            values,
            color=[COLORS[system] for system in SYSTEMS],
        )
        axis.set_xticks(range(len(SYSTEMS)), [LABELS[s] for s in SYSTEMS], rotation=28, ha="right")
        axis.set_ylabel("nDCG@10")
        axis.set_title(dataset_name.capitalize())
        axis.set_ylim(0, max(values) * 1.18)
    _save(fig, output_dir / "retrieval_quality_comparison.png")

    # Candidate recall at each depth.
    depths = (10, 50, 100, 1000)
    candidate_systems = ("bm25f_phrase_proximity", "dense", "hybrid_rrf")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for axis, (dataset_name, payload) in zip(axes, datasets.items()):
        for system in candidate_systems:
            axis.plot(
                depths,
                [
                    payload["aggregates"][system][f"Recall@{depth}"]
                    for depth in depths
                ],
                marker="o",
                label=LABELS[system],
            )
        axis.set_xscale("log")
        axis.set_xticks(depths, [str(depth) for depth in depths])
        axis.set_xlabel("Candidate depth")
        axis.set_ylabel("Recall")
        axis.set_title(dataset_name.capitalize())
        axis.grid(alpha=0.25)
    axes[1].legend()
    _save(fig, output_dir / "candidate_recall_by_depth.png")

    # Query-level hybrid minus lexical AP distributions.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for axis, dataset_name in zip(axes, datasets):
        lexical = per_query[dataset_name]["bm25f_phrase_proximity"]
        hybrid = per_query[dataset_name]["hybrid_rrf"]
        deltas = np.asarray(
            [hybrid[q]["AP"] - lexical[q]["AP"] for q in sorted(lexical)]
        )
        axis.hist(deltas, bins=31, color="#E8684A", alpha=0.85)
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(float(deltas.mean()), color="#5B8FF9", linestyle="--", label="mean")
        axis.set_xlabel("Per-query ΔAP (hybrid − lexical)")
        axis.set_ylabel("Queries")
        axis.set_title(dataset_name.capitalize())
        axis.legend()
    _save(fig, output_dir / "query_level_improvement_distribution.png")

    # Lexical versus dense wins/losses.
    fig, axis = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(datasets))
    width = 0.24
    for offset, outcome, color in (
        (-width, "wins", "#61DDAA"),
        (0, "losses", "#E8684A"),
        (width, "ties", "#65789B"),
    ):
        counts = []
        for dataset_name in datasets:
            lexical = per_query[dataset_name]["bm25f_phrase_proximity"]
            dense = per_query[dataset_name]["dense"]
            deltas = [dense[q]["AP"] - lexical[q]["AP"] for q in lexical]
            if outcome == "wins":
                counts.append(sum(delta > 1e-12 for delta in deltas))
            elif outcome == "losses":
                counts.append(sum(delta < -1e-12 for delta in deltas))
            else:
                counts.append(sum(abs(delta) <= 1e-12 for delta in deltas))
        axis.bar(x + offset, counts, width, label=outcome.capitalize(), color=color)
    axis.set_xticks(x, [name.capitalize() for name in datasets])
    axis.set_ylabel("Queries (dense relative to lexical)")
    axis.legend()
    _save(fig, output_dir / "lexical_vs_dense_wins_losses.png")

    # Quality-latency frontier (mean online latency; CPU only).
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, (dataset_name, payload) in zip(axes, datasets.items()):
        efficiency = payload["efficiency"]
        for system in SYSTEMS:
            if system == "hybrid_rrf":
                latency = efficiency[system]["estimated_end_to_end_mean_latency_ms"]
            elif system == "hybrid_rerank":
                latency = (
                    efficiency["hybrid_rrf"]["estimated_end_to_end_mean_latency_ms"]
                    + efficiency[system]["mean_latency_ms"]
                )
            else:
                latency = efficiency[system]["mean_latency_ms"]
            quality = payload["aggregates"][system]["nDCG@10"]
            axis.scatter(latency, quality, color=COLORS[system], s=42)
            axis.annotate(LABELS[system], (latency, quality), xytext=(4, 3), textcoords="offset points", fontsize=7)
        axis.set_xscale("log")
        axis.set_xlabel("Mean query latency (ms, CPU, log scale)")
        axis.set_ylabel("nDCG@10")
        axis.set_title(dataset_name.capitalize())
        axis.grid(alpha=0.22)
    _save(fig, output_dir / "quality_latency_frontier.png")

    # Controlled ladder / ablation in MAP.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, (dataset_name, payload) in zip(axes, datasets.items()):
        values = [payload["aggregates"][system]["MAP"] for system in SYSTEMS]
        axis.plot(
            range(1, len(SYSTEMS) + 1),
            values,
            marker="o",
            color="#5B8FF9",
        )
        axis.set_xticks(
            range(1, len(SYSTEMS) + 1),
            [LABELS[s] for s in SYSTEMS],
            rotation=28,
            ha="right",
        )
        axis.set_ylabel("MAP")
        axis.set_title(dataset_name.capitalize())
        axis.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "controlled_ablation_ladder.png")

    summary = {
        dataset_name: {
            "aggregates": payload["aggregates"],
            "significance_vs_bm25f_phrase_proximity": payload[
                "significance_vs_bm25f_phrase_proximity"
            ],
            "paired_stage_comparisons": payload["paired_stage_comparisons"],
            "efficiency": payload["efficiency"],
            "query_analysis": payload["query_analysis"],
        }
        for dataset_name, payload in datasets.items()
    }
    (output_dir / "generated_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
