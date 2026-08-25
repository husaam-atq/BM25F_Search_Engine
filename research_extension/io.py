"""Stable JSON/TREC run persistence."""

from __future__ import annotations

import json
from pathlib import Path

from .types import RankedDocument, Run


def save_run(run: Run, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        query_id: [
            {"doc_id": item.doc_id, "score": item.score}
            for item in ranking
        ]
        for query_id, ranking in sorted(run.items())
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_run(path: str | Path) -> Run:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(query_id): [
            RankedDocument(str(item["doc_id"]), float(item["score"]))
            for item in ranking
        ]
        for query_id, ranking in payload.items()
    }


def save_trec_run(run: Run, path: str | Path, tag: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for query_id, ranking in sorted(run.items()):
        for rank, item in enumerate(ranking, start=1):
            lines.append(
                f"{query_id} Q0 {item.doc_id} {rank} {item.score:.10f} {tag}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

