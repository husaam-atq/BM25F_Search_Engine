"""Paired uncertainty and effect-size analysis over queries."""

from __future__ import annotations

import math
from statistics import fmean, stdev

import numpy as np


def paired_comparison(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    metric: str = "AP",
    seed: int = 2026,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 20_000,
) -> dict[str, float | int | str]:
    query_ids = sorted(set(baseline) & set(candidate))
    if not query_ids:
        raise ValueError("No common queries for paired comparison")
    deltas = np.asarray(
        [candidate[q][metric] - baseline[q][metric] for q in query_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, len(deltas), size=(bootstrap_samples, len(deltas))
    )
    bootstrap_means = deltas[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    observed = abs(float(deltas.mean()))
    extreme = 0
    remaining = permutation_samples
    batch_size = 1000
    while remaining:
        size = min(batch_size, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(deltas)))
        permuted = np.abs((signs * deltas).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        remaining -= size
    p_value = (extreme + 1) / (permutation_samples + 1)

    delta_list = deltas.tolist()
    standard_deviation = stdev(delta_list) if len(delta_list) > 1 else 0.0
    effect = (
        float(deltas.mean()) / standard_deviation
        if standard_deviation > 0
        else 0.0
    )
    wins = int(np.count_nonzero(deltas > 1e-12))
    losses = int(np.count_nonzero(deltas < -1e-12))
    ties = len(deltas) - wins - losses
    interpretation = (
        "confidence interval excludes zero"
        if lower > 0 or upper < 0
        else "statistically uncertain at the 95% interval"
    )
    return {
        "metric": metric,
        "queries": len(query_ids),
        "baseline_mean": fmean(baseline[q][metric] for q in query_ids),
        "candidate_mean": fmean(candidate[q][metric] for q in query_ids),
        "mean_difference": float(deltas.mean()),
        "bootstrap_ci_95_low": float(lower),
        "bootstrap_ci_95_high": float(upper),
        "randomisation_p_two_sided": float(p_value),
        "paired_effect_size_dz": float(effect),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "interpretation": interpretation,
    }

