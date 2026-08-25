"""Pretrained dense retrieval with exact, inspectable NumPy search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .types import Document, Documents, RankedDocument


class RetrievalEncoder(Protocol):
    model_id: str
    dimension: int

    def encode_documents(self, documents: Sequence[Document]) -> np.ndarray: ...

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray: ...

    def metadata(self) -> dict[str, object]: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-length embedding encountered")
    return matrix / norms


class SentenceTransformerEncoder:
    """Lazy adapter around the E5 sentence-transformers checkpoint."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 128,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.requested_revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = SentenceTransformer(
            model_id,
            revision=revision,
            device=device,
            trust_remote_code=False,
        )
        self.model.max_seq_length = max_length
        dimension_method = getattr(self.model, "get_embedding_dimension", None)
        if dimension_method is None:
            dimension_method = self.model.get_sentence_embedding_dimension
        self.dimension = int(dimension_method())
        module = self.model._first_module()
        config = getattr(getattr(module, "auto_model", None), "config", None)
        self.resolved_revision = getattr(config, "_commit_hash", None)
        self.device = str(self.model.device)

    def encode_documents(self, documents: Sequence[Document]) -> np.ndarray:
        texts = [f"passage: {document.formatted()}" for document in documents]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        texts = [f"query: {query}" for query in queries]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "embedding_dimension": self.dimension,
            "document_format": "passage: title: {title}\\nbody: {text}",
            "query_format": "query: {query}",
            "maximum_length_tokens": self.max_length,
            "pooling": "model-defined average pooling",
            "normalisation": "L2",
            "similarity": "dot product (equivalent to cosine after L2 normalisation)",
            "device": self.device,
        }


class DenseIndex:
    """An exact dense index deliberately kept small and transparent."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        doc_ids: list[str],
        embeddings: np.ndarray,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Dense-index document IDs must be unique")
        embeddings = _normalise(embeddings)
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("Dense-index IDs and embeddings do not align")
        self.doc_ids = list(doc_ids)
        self.embeddings = embeddings
        self.metadata = dict(metadata or {})
        self._doc_id_array = np.asarray(self.doc_ids, dtype=str)

    @classmethod
    def build(
        cls,
        documents: Documents | Sequence[Document],
        encoder: RetrievalEncoder,
    ) -> "DenseIndex":
        ordered = (
            list(documents.values())
            if isinstance(documents, dict)
            else list(documents)
        )
        embeddings = encoder.encode_documents(ordered)
        return cls(
            [document.doc_id for document in ordered],
            embeddings,
            metadata=encoder.metadata(),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray([self.SCHEMA_VERSION], dtype=np.int64),
            doc_ids=np.asarray(self.doc_ids, dtype=str),
            embeddings=self.embeddings,
            metadata=np.asarray([json.dumps(self.metadata, sort_keys=True)]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        with np.load(Path(path), allow_pickle=False) as payload:
            schema = int(payload["schema_version"][0])
            if schema != cls.SCHEMA_VERSION:
                raise ValueError("Unsupported dense-index schema")
            return cls(
                payload["doc_ids"].astype(str).tolist(),
                payload["embeddings"],
                json.loads(str(payload["metadata"][0])),
            )

    def search_embedding(
        self, query_embedding: np.ndarray, top_k: int = 1000
    ) -> list[RankedDocument]:
        if top_k <= 0:
            return []
        query = _normalise(np.asarray(query_embedding).reshape(1, -1))[0]
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query and document embedding dimensions differ")
        scores = self.embeddings @ query
        depth = min(top_k, len(self.doc_ids))
        if depth == len(self.doc_ids):
            candidates = np.arange(len(self.doc_ids))
        else:
            candidates = np.argpartition(-scores, depth - 1)[:depth]
        order = sorted(
            candidates.tolist(),
            key=lambda index: (-float(scores[index]), self.doc_ids[index]),
        )
        return [
            RankedDocument(self.doc_ids[index], float(scores[index]))
            for index in order
        ]

    def search(
        self,
        query: str,
        encoder: RetrievalEncoder,
        top_k: int = 1000,
    ) -> list[RankedDocument]:
        return self.search_embedding(encoder.encode_queries([query])[0], top_k)

    def search_many(
        self,
        queries: dict[str, str],
        encoder: RetrievalEncoder,
        top_k: int = 1000,
    ) -> dict[str, list[RankedDocument]]:
        query_ids = list(queries)
        embeddings = encoder.encode_queries([queries[query_id] for query_id in query_ids])
        return {
            query_id: self.search_embedding(embedding, top_k=top_k)
            for query_id, embedding in zip(query_ids, embeddings)
        }
