"""End-to-end frozen public-benchmark experiment runner."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .config import FROZEN_CONFIG, ExperimentConfig, LexicalConfig
from .datasets import dataset_manifest, download_dataset, load_dataset
from .dense import DenseIndex, SentenceTransformerEncoder
from .efficiency import (
    environment_manifest,
    file_size_bytes,
    measure_queries,
)
from .evaluation import evaluate_run
from .hybrid import reciprocal_rank_fusion
from .io import save_run, save_trec_run
from .lexical import LexicalIndex
from .query_analysis import analyse_queries
from .reranking import CrossEncoderScorer, rerank
from .significance import paired_comparison
from .tuning import select_lexical_config
from .types import Documents, RankedDocument, Run


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_lexical(
    documents: Documents, path: Path
) -> tuple[LexicalIndex, dict[str, object]]:
    start = time.perf_counter()
    rss_before = _rss()
    index = LexicalIndex.build(documents)
    build_seconds = time.perf_counter() - start
    rss_after = _rss()
    index.save(path)
    return index, {
        "build_time_s": build_seconds,
        "index_size_bytes": file_size_bytes(path),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": _difference(rss_after, rss_before),
        "documents": index.document_count,
        "terms": len(index.postings),
    }


def _rss() -> int | None:
    from .efficiency import current_rss_bytes

    return current_rss_bytes()


def _difference(after: int | None, before: int | None) -> int | None:
    return after - before if after is not None and before is not None else None


def _build_dense(
    documents: Documents,
    encoder: SentenceTransformerEncoder,
    path: Path,
) -> tuple[DenseIndex, dict[str, object]]:
    start = time.perf_counter()
    rss_before = _rss()
    index = DenseIndex.build(documents, encoder)
    embedding_seconds = time.perf_counter() - start
    rss_after = _rss()
    index.save(path)
    return index, {
        "document_embedding_time_s": embedding_seconds,
        "documents_per_second": (
            len(documents) / embedding_seconds if embedding_seconds else 0.0
        ),
        "dense_index_size_bytes": file_size_bytes(path),
        "embedding_dimension": index.embeddings.shape[1],
        "embedding_dtype": str(index.embeddings.dtype),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": _difference(rss_after, rss_before),
    }


def _measure_transform(
    query_ids: list[str],
    transform,
) -> tuple[Run, dict[str, float | int | None]]:
    probe = {query_id: query_id for query_id in query_ids}
    outputs, timing = measure_queries(
        probe, lambda query_id: transform(query_id), warmup=False
    )
    return outputs, timing


def run_dataset(
    dataset_name: str,
    lexical_config: LexicalConfig,
    experiment_config: ExperimentConfig = FROZEN_CONFIG,
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
    device: str | None = None,
) -> dict[str, object]:
    """Evaluate one test split without adapting its configuration."""
    documents, queries, qrels = load_dataset(
        dataset_name,
        experiment_config.test_split,
        data_dir=data_dir,
        download=True,
    )
    output_dir = Path(artifact_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    lexical_path = output_dir / "lexical_index.pkl"
    lexical_index, lexical_build = _build_lexical(documents, lexical_path)

    runs: dict[str, Run] = {}
    efficiency: dict[str, object] = {
        "lexical_index": lexical_build,
        "environment": environment_manifest(),
    }
    for name, variant in (
        ("bm25", "bm25"),
        ("bm25f", "bm25f"),
        ("bm25f_phrase_proximity", "bm25f_phrase_proximity"),
    ):
        run, timing = measure_queries(
            queries,
            lambda query, variant=variant: lexical_index.search(
                query,
                variant=variant,
                config=lexical_config,
                top_k=experiment_config.retrieval_depth,
            ),
        )
        runs[name] = run
        efficiency[name] = timing

    model_load_start = time.perf_counter()
    encoder = SentenceTransformerEncoder(
        experiment_config.dense_model,
        revision=experiment_config.dense_revision,
        device=device,
        batch_size=experiment_config.dense_batch_size,
        max_length=experiment_config.dense_max_length,
    )
    encoder_load_time = time.perf_counter() - model_load_start
    dense_path = output_dir / "dense_index.npz"
    dense_index, dense_build = _build_dense(documents, encoder, dense_path)
    dense_build["model_load_time_s"] = encoder_load_time
    dense_build["model"] = encoder.metadata()
    efficiency["dense_index"] = dense_build

    dense_run, dense_timing = measure_queries(
        queries,
        lambda query: dense_index.search(
            query, encoder, top_k=experiment_config.retrieval_depth
        ),
    )
    runs["dense"] = dense_run
    efficiency["dense"] = dense_timing

    query_ids = list(queries)
    hybrid_run, hybrid_timing = _measure_transform(
        query_ids,
        lambda query_id: reciprocal_rank_fusion(
            [
                runs["bm25f_phrase_proximity"][query_id],
                runs["dense"][query_id],
            ],
            rrf_k=experiment_config.rrf_k,
            top_k=experiment_config.retrieval_depth,
        ),
    )
    runs["hybrid_rrf"] = hybrid_run
    hybrid_timing["component_query_latency_note"] = (
        "fusion-only; end-to-end hybrid adds lexical and dense first-stage latency"
    )
    hybrid_timing["estimated_end_to_end_mean_latency_ms"] = (
        float(efficiency["bm25f_phrase_proximity"]["mean_latency_ms"])
        + float(efficiency["dense"]["mean_latency_ms"])
        + float(hybrid_timing["mean_latency_ms"])
    )
    efficiency["hybrid_rrf"] = hybrid_timing

    reranker_load_start = time.perf_counter()
    reranker_rss_before = _rss()
    scorer = CrossEncoderScorer(
        experiment_config.reranker_model,
        revision=experiment_config.reranker_revision,
        device=device,
        batch_size=experiment_config.rerank_batch_size,
        max_length=experiment_config.reranker_max_length,
    )
    reranker_rss_after = _rss()
    efficiency["reranker_model"] = {
        "model_load_time_s": time.perf_counter() - reranker_load_start,
        "model": scorer.metadata(),
        "fixed_candidate_depth": experiment_config.rerank_depth,
        "rss_before_bytes": reranker_rss_before,
        "rss_after_bytes": reranker_rss_after,
        "rss_delta_bytes": _difference(
            reranker_rss_after, reranker_rss_before
        ),
    }
    for source, target in (
        ("bm25f_phrase_proximity", "lexical_rerank"),
        ("dense", "dense_rerank"),
        ("hybrid_rrf", "hybrid_rerank"),
    ):
        reranked, timing = _measure_transform(
            query_ids,
            lambda query_id, source=source: rerank(
                queries[query_id],
                runs[source][query_id],
                documents,
                scorer,
                depth=experiment_config.rerank_depth,
            ),
        )
        runs[target] = reranked
        timing["candidate_source"] = source
        timing["fixed_candidate_depth"] = experiment_config.rerank_depth
        efficiency[target] = timing

    aggregates: dict[str, dict[str, float | int]] = {}
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for system, run in runs.items():
        aggregate, query_rows = evaluate_run(qrels, run)
        aggregates[system] = aggregate
        per_query[system] = query_rows
        save_run(run, output_dir / "runs" / f"{system}.json")
        save_trec_run(
            run,
            output_dir / "runs" / f"{system}.run.trec",
            tag=f"ext26_{system}",
        )

    comparisons: dict[str, object] = {}
    baseline = "bm25f_phrase_proximity"
    for system in ("dense", "hybrid_rrf", "lexical_rerank", "dense_rerank", "hybrid_rerank"):
        comparisons[system] = {
            metric: paired_comparison(
                per_query[baseline],
                per_query[system],
                metric=metric,
                seed=experiment_config.random_seed,
            )
            for metric in ("AP", "nDCG@10")
        }
    stage_pairs = (
        ("bm25_to_bm25f", "bm25", "bm25f"),
        (
            "bm25f_to_phrase_proximity",
            "bm25f",
            "bm25f_phrase_proximity",
        ),
        (
            "lexical_to_dense",
            "bm25f_phrase_proximity",
            "dense",
        ),
        (
            "lexical_to_hybrid",
            "bm25f_phrase_proximity",
            "hybrid_rrf",
        ),
        (
            "lexical_to_lexical_rerank",
            "bm25f_phrase_proximity",
            "lexical_rerank",
        ),
        ("dense_to_dense_rerank", "dense", "dense_rerank"),
        ("hybrid_to_hybrid_rerank", "hybrid_rrf", "hybrid_rerank"),
    )
    stage_comparisons = {
        label: {
            metric: paired_comparison(
                per_query[source],
                per_query[target],
                metric=metric,
                seed=experiment_config.random_seed,
            )
            for metric in ("AP", "nDCG@10")
        }
        for label, source, target in stage_pairs
    }

    query_analysis = analyse_queries(
        queries,
        lexical_index,
        runs,
        per_query,
        baseline_name=baseline,
        candidate_name="hybrid_rrf",
    )
    payload = {
        "extension_label": experiment_config.label,
        "dataset": dataset_name,
        "split": experiment_config.test_split,
        "documents": len(documents),
        "queries": len(queries),
        "qrels_judgments": sum(len(values) for values in qrels.values()),
        "lexical_config": asdict(lexical_config),
        "experiment_config": experiment_config.to_dict(),
        "aggregates": aggregates,
        "significance_vs_bm25f_phrase_proximity": comparisons,
        "paired_stage_comparisons": stage_comparisons,
        "efficiency": efficiency,
        "query_analysis": query_analysis,
    }
    _write_json(output_dir / "results.json", payload)
    _write_json(output_dir / "per_query.json", per_query)
    return payload


def run_all(
    experiment_config: ExperimentConfig = FROZEN_CONFIG,
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
    device: str | None = None,
) -> dict[str, object]:
    """Download, tune on primary train, freeze, then evaluate both test sets."""
    for name in (
        experiment_config.primary_dataset,
        experiment_config.generalisation_dataset,
    ):
        download_dataset(name, data_dir)

    train_documents, train_queries, train_qrels = load_dataset(
        experiment_config.primary_dataset,
        experiment_config.train_split,
        data_dir=data_dir,
        download=False,
    )
    train_index = LexicalIndex.build(train_documents)
    lexical_config, tuning_log = select_lexical_config(
        train_index,
        train_queries,
        train_qrels,
        base=experiment_config.lexical,
        top_k=experiment_config.retrieval_depth,
    )
    protocol = {
        "extension_label": experiment_config.label,
        "test_qrels_access_rule": (
            "configuration selected on primary train split before test qrels are loaded"
        ),
        "experiment_config": experiment_config.to_dict(),
        "selected_lexical_config": asdict(lexical_config),
        "tuning_log": tuning_log,
        "dataset_manifests": {
            name: dataset_manifest(name, data_dir)
            for name in (
                experiment_config.primary_dataset,
                experiment_config.generalisation_dataset,
            )
        },
    }
    artifact_dir = Path(artifact_dir)
    _write_json(artifact_dir / "protocol_frozen_before_test.json", protocol)

    results = {
        name: run_dataset(
            name,
            lexical_config,
            experiment_config=experiment_config,
            data_dir=data_dir,
            artifact_dir=artifact_dir,
            device=device,
        )
        for name in (
            experiment_config.primary_dataset,
            experiment_config.generalisation_dataset,
        )
    }
    headline = {"protocol": protocol, "results": results}
    _write_json(artifact_dir / "headline_results.json", headline)
    return headline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("download", "full"),
        help="Download public data or reproduce the complete frozen experiment",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.command == "download":
        for name in (
            FROZEN_CONFIG.primary_dataset,
            FROZEN_CONFIG.generalisation_dataset,
        ):
            path = download_dataset(name, args.data_dir)
            print(f"{name}: {path}")
    else:
        run_all(
            data_dir=args.data_dir,
            artifact_dir=args.artifact_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
