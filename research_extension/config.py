"""Frozen protocol values for the 2026 independent extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LexicalConfig:
    k1: float = 1.2
    b: float = 0.75
    b_title: float = 0.75
    b_body: float = 0.75
    title_weight: float = 2.0
    body_weight: float = 1.0
    phrase_bonus: float = 0.50
    proximity_bonus: float = 0.10
    proximity_window: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    label: str = "Post-Course Independent Retrieval Research Extension — 2026"
    random_seed: int = 2026
    primary_dataset: str = "scifact"
    generalisation_dataset: str = "nfcorpus"
    train_split: str = "train"
    test_split: str = "test"
    dense_model: str = "intfloat/e5-small-v2"
    dense_revision: str | None = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
    embedding_dimension: int = 384
    dense_max_length: int = 512
    dense_batch_size: int = 128
    similarity: str = "cosine (L2-normalised dot product)"
    dense_index: str = "exact NumPy matrix search"
    rrf_k: int = 60
    retrieval_depth: int = 1000
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L2-v2"
    reranker_revision: str | None = "1b5cd67b15209f24824c50370e0397743aa9b787"
    rerank_depth: int = 50
    rerank_batch_size: int = 256
    reranker_max_length: int = 512
    lexical: LexicalConfig = LexicalConfig()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FROZEN_CONFIG = ExperimentConfig()
