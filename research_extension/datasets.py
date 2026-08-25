"""Download and load small public BEIR-format benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import ssl
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .types import Document, Documents, Qrels, Queries


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    md5: str
    source: str
    available_splits: tuple[str, ...]


_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATASETS: dict[str, DatasetSpec] = {
    "scifact": DatasetSpec(
        name="scifact",
        url=f"{_BASE_URL}/scifact.zip",
        md5="5f7d1de60b170fc8027bb7898e2efca1",
        source="BEIR release of SciFact",
        available_splits=("train", "test"),
    ),
    "nfcorpus": DatasetSpec(
        name="nfcorpus",
        url=f"{_BASE_URL}/nfcorpus.zip",
        md5="a89dba18a62ef92f7d323ec890a0d38d",
        source="BEIR release of NFCorpus",
        available_splits=("train", "dev", "test"),
    ),
}


def _digest(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def download_dataset(name: str, data_dir: str | Path = "data") -> Path:
    """Download one declared archive, verify the official MD5, and extract it."""
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset {name!r}; choose from {sorted(DATASETS)}")
    spec = DATASETS[name]
    data_dir = Path(data_dir)
    target = data_dir / name
    expected = target / "corpus.jsonl"
    if expected.exists():
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"
    partial = archive.with_suffix(".zip.part")
    if not archive.exists():
        try:
            import certifi

            tls_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            tls_context = ssl.create_default_context()
        with urllib.request.urlopen(spec.url, context=tls_context) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(archive)

    actual_md5 = _digest(archive)
    if actual_md5 != spec.md5:
        raise ValueError(
            f"{archive} failed MD5 verification: expected {spec.md5}, got {actual_md5}"
        )
    if target.exists():
        raise FileExistsError(
            f"{target} exists but is incomplete; inspect it before retrying"
        )
    _safe_extract(archive, data_dir)
    if not expected.exists():
        raise FileNotFoundError(f"Archive did not create {expected}")
    return target


def load_corpus(dataset_dir: str | Path) -> Documents:
    documents: Documents = {}
    path = Path(dataset_dir) / "corpus.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            doc_id = str(row["_id"])
            if doc_id in documents:
                raise ValueError(f"Duplicate document ID {doc_id!r} at line {line_number}")
            documents[doc_id] = Document(
                doc_id=doc_id,
                title=str(row.get("title", "") or ""),
                text=str(row.get("text", "") or ""),
                metadata=dict(row.get("metadata", {}) or {}),
            )
    return documents


def load_queries(dataset_dir: str | Path) -> Queries:
    queries: Queries = {}
    path = Path(dataset_dir) / "queries.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            query_id = str(row["_id"])
            if query_id in queries:
                raise ValueError(f"Duplicate query ID {query_id!r} at line {line_number}")
            queries[query_id] = str(row["text"])
    return queries


def load_qrels(dataset_dir: str | Path, split: str) -> Qrels:
    qrels: Qrels = {}
    path = Path(dataset_dir) / "qrels" / f"{split}.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected qrels header in {path}: {reader.fieldnames}")
        for row in reader:
            query_id = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = int(row["score"])
            existing = qrels.setdefault(query_id, {}).get(doc_id)
            if existing is not None and existing != score:
                raise ValueError(f"Conflicting qrel for {query_id}/{doc_id}")
            qrels[query_id][doc_id] = score
    return qrels


def load_dataset(
    name: str,
    split: str,
    data_dir: str | Path = "data",
    download: bool = True,
) -> tuple[Documents, Queries, Qrels]:
    spec = DATASETS[name]
    if split not in spec.available_splits:
        raise ValueError(f"{name} has no declared {split!r} split")
    root = download_dataset(name, data_dir) if download else Path(data_dir) / name
    documents = load_corpus(root)
    all_queries = load_queries(root)
    qrels = load_qrels(root, split)
    missing_queries = sorted(set(qrels) - set(all_queries))
    missing_docs = sorted(
        {doc_id for judgments in qrels.values() for doc_id in judgments} - set(documents)
    )
    if missing_queries or missing_docs:
        raise ValueError(
            f"Dataset mismatch: {len(missing_queries)} missing queries, "
            f"{len(missing_docs)} missing documents"
        )
    queries = {query_id: all_queries[query_id] for query_id in qrels}
    return documents, queries, qrels


def dataset_manifest(name: str, data_dir: str | Path = "data") -> dict[str, object]:
    root = Path(data_dir) / name
    spec = DATASETS[name]
    documents = load_corpus(root)
    queries = load_queries(root)
    split_counts = {
        split: {
            "queries": len(load_qrels(root, split)),
            "judgments": sum(len(v) for v in load_qrels(root, split).values()),
            "positive_judgments": sum(
                score > 0
                for judgments in load_qrels(root, split).values()
                for score in judgments.values()
            ),
        }
        for split in spec.available_splits
    }
    archive = Path(data_dir) / f"{name}.zip"
    return {
        "spec": asdict(spec),
        "archive_sha256": _digest(archive, "sha256") if archive.exists() else None,
        "documents": len(documents),
        "queries_total": len(queries),
        "splits": split_counts,
    }
