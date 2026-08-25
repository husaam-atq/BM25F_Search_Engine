"""A compact positional BM25/BM25F engine for the independent extension."""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Literal

from .config import LexicalConfig
from .text import tokenise
from .types import Document, Documents, RankedDocument

LexicalVariant = Literal["bm25", "bm25f", "bm25f_phrase_proximity"]


@dataclass(frozen=True)
class Posting:
    doc_index: int
    title_positions: tuple[int, ...]
    body_positions: tuple[int, ...]

    @property
    def title_tf(self) -> int:
        return len(self.title_positions)

    @property
    def body_tf(self) -> int:
        return len(self.body_positions)


class LexicalIndex:
    """In-memory positional index with a simple, versioned pickle format."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        doc_ids: list[str],
        title_lengths: list[int],
        body_lengths: list[int],
        postings: dict[str, tuple[Posting, ...]],
    ) -> None:
        if not (len(doc_ids) == len(title_lengths) == len(body_lengths)):
            raise ValueError("Document IDs and field-length arrays must align")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Document IDs must be unique")
        self.doc_ids = doc_ids
        self.title_lengths = title_lengths
        self.body_lengths = body_lengths
        self.postings = postings
        self.average_title_length = (
            sum(title_lengths) / len(title_lengths) if title_lengths else 0.0
        )
        self.average_body_length = (
            sum(body_lengths) / len(body_lengths) if body_lengths else 0.0
        )

    @property
    def document_count(self) -> int:
        return len(self.doc_ids)

    @classmethod
    def build(cls, documents: Documents | Iterable[Document]) -> "LexicalIndex":
        ordered = (
            list(documents.values())
            if isinstance(documents, dict)
            else list(documents)
        )
        doc_ids: list[str] = []
        title_lengths: list[int] = []
        body_lengths: list[int] = []
        mutable: dict[str, dict[int, list[list[int]]]] = defaultdict(dict)

        for doc_index, document in enumerate(ordered):
            if document.doc_id in doc_ids:
                raise ValueError(f"Duplicate document ID {document.doc_id!r}")
            doc_ids.append(document.doc_id)
            title_terms = tokenise(document.title)
            body_terms = tokenise(document.text)
            title_lengths.append(len(title_terms))
            body_lengths.append(len(body_terms))

            for field, terms in enumerate((title_terms, body_terms)):
                for position, term in enumerate(terms):
                    fields = mutable[term].setdefault(doc_index, [[], []])
                    fields[field].append(position)

        postings = {
            term: tuple(
                Posting(doc_index, tuple(fields[0]), tuple(fields[1]))
                for doc_index, fields in sorted(by_doc.items())
            )
            for term, by_doc in mutable.items()
        }
        return cls(doc_ids, title_lengths, body_lengths, postings)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "doc_ids": self.doc_ids,
            "title_lengths": self.title_lengths,
            "body_lengths": self.body_lengths,
            "postings": self.postings,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "LexicalIndex":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported lexical-index schema")
        return cls(
            payload["doc_ids"],
            payload["title_lengths"],
            payload["body_lengths"],
            payload["postings"],
        )

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def _idf(self, term: str) -> float:
        df = self.document_frequency(term)
        if df <= 0 or self.document_count <= 0:
            return 0.0
        return math.log(
            1.0 + (self.document_count - df + 0.5) / (df + 0.5)
        )

    @staticmethod
    def _normalised_tf(tf: int, length: int, average: float, b: float) -> float:
        if tf <= 0:
            return 0.0
        denominator = 1.0 - b + b * length / average if average > 0 else 1.0
        return tf / denominator

    @staticmethod
    def _saturate(tf: float, k1: float) -> float:
        return (tf * (k1 + 1.0)) / (tf + k1) if tf > 0 else 0.0

    def _term_score(
        self,
        term: str,
        posting: Posting,
        variant: LexicalVariant,
        config: LexicalConfig,
    ) -> float:
        doc_index = posting.doc_index
        if variant == "bm25":
            tf = posting.title_tf + posting.body_tf
            length = self.title_lengths[doc_index] + self.body_lengths[doc_index]
            average = self.average_title_length + self.average_body_length
            normalised = self._normalised_tf(tf, length, average, config.b)
        else:
            title_tf = self._normalised_tf(
                posting.title_tf,
                self.title_lengths[doc_index],
                self.average_title_length,
                config.b_title,
            )
            body_tf = self._normalised_tf(
                posting.body_tf,
                self.body_lengths[doc_index],
                self.average_body_length,
                config.b_body,
            )
            normalised = (
                config.title_weight * title_tf + config.body_weight * body_tf
            )
        return self._idf(term) * self._saturate(normalised, config.k1)

    @staticmethod
    def _ordered_adjacent(first: tuple[int, ...], second: tuple[int, ...]) -> bool:
        second_set = set(second)
        return any(position + 1 in second_set for position in first)

    @staticmethod
    def _minimum_gap(
        first: tuple[int, ...],
        second: tuple[int, ...],
        same_term: bool,
    ) -> int | None:
        if not first or not second:
            return None
        if same_term:
            return min(
                (right - left for left, right in zip(first, first[1:])),
                default=None,
            )
        i = j = 0
        best: int | None = None
        while i < len(first) and j < len(second):
            gap = abs(first[i] - second[j])
            best = gap if best is None else min(best, gap)
            if first[i] < second[j]:
                i += 1
            else:
                j += 1
        return best

    def _dependence_features(
        self,
        query_terms: list[str],
        posting_map: dict[str, Posting],
    ) -> tuple[float, float]:
        phrase_feature = 0.0
        proximity_feature = 0.0
        for first_term, second_term in zip(query_terms, query_terms[1:]):
            first = posting_map.get(first_term)
            second = posting_map.get(second_term)
            if first is None or second is None:
                continue
            title_match = self._ordered_adjacent(
                first.title_positions, second.title_positions
            )
            body_match = self._ordered_adjacent(
                first.body_positions, second.body_positions
            )
            if title_match or body_match:
                phrase_feature += 1.0

        for left_index, right_index in combinations(range(len(query_terms)), 2):
            first_term = query_terms[left_index]
            second_term = query_terms[right_index]
            first = posting_map.get(first_term)
            second = posting_map.get(second_term)
            if first is None or second is None:
                continue
            same_term = first_term == second_term
            gaps = [
                self._minimum_gap(
                    first.title_positions, second.title_positions, same_term
                ),
                self._minimum_gap(
                    first.body_positions, second.body_positions, same_term
                ),
            ]
            valid = [gap for gap in gaps if gap is not None]
            if not valid:
                continue
            gap = min(valid)
            if 1 <= gap <= 8:
                closeness = (
                    8 - gap + 1
                ) / 8
                proximity_feature += closeness
        return phrase_feature, proximity_feature

    def _dependence_bonus(
        self,
        query_terms: list[str],
        posting_map: dict[str, Posting],
        config: LexicalConfig,
    ) -> float:
        phrase_feature, proximity_feature = self._dependence_features(
            query_terms, posting_map
        )
        # The frozen protocol uses an eight-token window. Keeping the feature
        # extraction fixed makes train-grid reuse exact and transparent.
        if config.proximity_window != 8:
            proximity_feature = self._proximity_feature(
                query_terms, posting_map, config.proximity_window
            )
        return (
            config.phrase_bonus * phrase_feature
            + config.proximity_bonus * proximity_feature
        )

    def _proximity_feature(
        self,
        query_terms: list[str],
        posting_map: dict[str, Posting],
        window: int,
    ) -> float:
        feature = 0.0
        for left_index, right_index in combinations(range(len(query_terms)), 2):
            first_term = query_terms[left_index]
            second_term = query_terms[right_index]
            first = posting_map.get(first_term)
            second = posting_map.get(second_term)
            if first is None or second is None:
                continue
            same_term = first_term == second_term
            gaps = [
                self._minimum_gap(
                    first.title_positions, second.title_positions, same_term
                ),
                self._minimum_gap(
                    first.body_positions, second.body_positions, same_term
                ),
            ]
            valid = [gap for gap in gaps if gap is not None]
            if valid and 1 <= (gap := min(valid)) <= window:
                feature += (window - gap + 1) / window
        return feature

    def score_components(
        self,
        query: str,
        config: LexicalConfig | None = None,
    ) -> list[tuple[str, float, float, float]]:
        """Return BM25F, unit-phrase, and unit-proximity features per candidate."""
        config = config or LexicalConfig()
        query_terms = tokenise(query)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        candidates: dict[int, dict[str, Posting]] = {}
        for term in query_counts:
            for posting in self.postings.get(term, ()):
                candidates.setdefault(posting.doc_index, {})[term] = posting
        output: list[tuple[str, float, float, float]] = []
        for doc_index, posting_map in candidates.items():
            lexical_score = sum(
                count * self._term_score(term, posting, "bm25f", config)
                for term, count in query_counts.items()
                if (posting := posting_map.get(term)) is not None
            )
            phrase_feature, proximity_feature = self._dependence_features(
                query_terms, posting_map
            )
            if config.proximity_window != 8:
                proximity_feature = self._proximity_feature(
                    query_terms, posting_map, config.proximity_window
                )
            output.append(
                (
                    self.doc_ids[doc_index],
                    lexical_score,
                    phrase_feature,
                    proximity_feature,
                )
            )
        return output

    def search(
        self,
        query: str,
        variant: LexicalVariant = "bm25f",
        config: LexicalConfig | None = None,
        top_k: int = 1000,
    ) -> list[RankedDocument]:
        if variant not in {"bm25", "bm25f", "bm25f_phrase_proximity"}:
            raise ValueError(f"Unknown lexical variant {variant!r}")
        if top_k <= 0:
            return []
        config = config or LexicalConfig()
        query_terms = tokenise(query)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        candidates: dict[int, dict[str, Posting]] = {}
        for term in query_counts:
            for posting in self.postings.get(term, ()):
                candidates.setdefault(posting.doc_index, {})[term] = posting

        scored: list[tuple[float, str]] = []
        for doc_index, posting_map in candidates.items():
            score = sum(
                count * self._term_score(term, posting, variant, config)
                for term, count in query_counts.items()
                if (posting := posting_map.get(term)) is not None
            )
            if variant == "bm25f_phrase_proximity":
                score += self._dependence_bonus(query_terms, posting_map, config)
            if score > 0:
                scored.append((score, self.doc_ids[doc_index]))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RankedDocument(doc_id=doc_id, score=float(score))
            for score, doc_id in scored[:top_k]
        ]


def retrieve_all(
    index: LexicalIndex,
    queries: dict[str, str],
    variant: LexicalVariant,
    config: LexicalConfig | None = None,
    top_k: int = 1000,
) -> dict[str, list[RankedDocument]]:
    return {
        query_id: index.search(query, variant=variant, config=config, top_k=top_k)
        for query_id, query in queries.items()
    }
