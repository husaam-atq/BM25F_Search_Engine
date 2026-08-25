"""Fixed-depth cross-encoder reranking with candidate-set safeguards."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .types import Document, Documents, RankedDocument, Run


class PairScorer(Protocol):
    def score(
        self, query: str, documents: Sequence[Document]
    ) -> Sequence[float]: ...

    def metadata(self) -> dict[str, object]: ...


class CrossEncoderScorer:
    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 256,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model_id
        self.requested_revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = CrossEncoder(
            model_id,
            revision=revision,
            device=device,
            max_length=max_length,
            trust_remote_code=False,
        )
        config = getattr(self.model.model, "config", None)
        self.resolved_revision = getattr(config, "_commit_hash", None)
        self.device = str(self.model.device)

    def score(
        self, query: str, documents: Sequence[Document]
    ) -> Sequence[float]:
        pairs = [(query, document.formatted()) for document in documents]
        return self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).reshape(-1)

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "input_format": "(query, title: {title}\\nbody: {text})",
            "maximum_length_tokens": self.max_length,
            "device": self.device,
        }


def rerank(
    query: str,
    candidates: list[RankedDocument],
    documents: Documents,
    scorer: PairScorer,
    depth: int = 50,
) -> list[RankedDocument]:
    candidate_ids = [item.doc_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate ranking contains duplicate document IDs")
    if depth <= 0 or not candidates:
        return list(candidates)
    head = candidates[:depth]
    missing = [item.doc_id for item in head if item.doc_id not in documents]
    if missing:
        raise KeyError(f"Candidate documents missing from corpus: {missing[:3]}")
    scores = np.asarray(
        scorer.score(query, [documents[item.doc_id] for item in head]),
        dtype=np.float64,
    ).reshape(-1)
    if len(scores) != len(head):
        raise ValueError("Reranker returned a different number of scores")
    rescored = [
        RankedDocument(item.doc_id, float(score))
        for item, score in zip(head, scores)
    ]
    rescored.sort(key=lambda item: (-item.score, item.doc_id))
    output = rescored + list(candidates[depth:])
    if set(candidate_ids) != {item.doc_id for item in output}:
        raise AssertionError("Reranking changed the candidate set")
    return output


def rerank_run(
    queries: dict[str, str],
    run: Run,
    documents: Documents,
    scorer: PairScorer,
    depth: int = 50,
) -> Run:
    if set(queries) != set(run):
        raise ValueError("Queries and run IDs must match")
    return {
        query_id: rerank(
            queries[query_id],
            run[query_id],
            documents,
            scorer,
            depth=depth,
        )
        for query_id in queries
    }

