"""Comparable latency, storage, memory, and environment measurements."""

from __future__ import annotations

import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def file_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def current_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def measure_queries(
    queries: dict[str, str],
    search: Callable[[str], T],
    warmup: bool = True,
) -> tuple[dict[str, T], dict[str, float | int | None]]:
    if warmup and queries:
        search(next(iter(queries.values())))
    outputs: dict[str, T] = {}
    latencies_ms: list[float] = []
    rss_before = current_rss_bytes()
    for query_id, query in queries.items():
        start = time.perf_counter()
        outputs[query_id] = search(query)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    rss_after = current_rss_bytes()
    return outputs, {
        "queries": len(queries),
        "mean_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "total_latency_s": sum(latencies_ms) / 1000,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": (
            rss_after - rss_before
            if rss_before is not None and rss_after is not None
            else None
        ),
    }


def environment_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        manifest["physical_cpu_count"] = psutil.cpu_count(logical=False)
        manifest["total_memory_bytes"] = psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(0)
            manifest["cuda_runtime"] = torch.version.cuda
    except ImportError:
        manifest["torch"] = None
        manifest["cuda_available"] = False
    return manifest

