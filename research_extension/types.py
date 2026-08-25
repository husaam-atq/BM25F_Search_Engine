"""Small shared data types for the independent extension."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str = ""
    text: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def formatted(self) -> str:
        """Use an explicit title/body boundary for neural models."""
        if self.title and self.text:
            return f"title: {self.title}\nbody: {self.text}"
        return self.title or self.text


@dataclass(frozen=True)
class RankedDocument:
    doc_id: str
    score: float


Run = dict[str, list[RankedDocument]]
Qrels = dict[str, dict[str, int]]
Queries = dict[str, str]
Documents = dict[str, Document]


def unique_doc_ids(ranking: Sequence[RankedDocument]) -> list[str]:
    """Return ranking IDs once, preserving first occurrence."""
    seen: set[str] = set()
    output: list[str] = []
    for item in ranking:
        if item.doc_id not in seen:
            seen.add(item.doc_id)
            output.append(item.doc_id)
    return output

