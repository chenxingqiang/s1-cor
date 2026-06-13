"""Shared calibration metrics for CoR CPU reports and ablations."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Tuple


def compute_ece(
    confidences: List[float],
    accuracies: List[float],
    n_bins: int = 5,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Expected calibration error with uniform bins on [0, 1]."""
    if not confidences:
        return 0.0, []

    buckets: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for conf, acc in zip(confidences, accuracies):
        idx = min(int(conf * n_bins), n_bins - 1)
        if conf >= 1.0:
            idx = n_bins - 1
        buckets[idx].append((conf, acc))

    ece = 0.0
    bin_stats: List[Dict[str, Any]] = []
    n = len(confidences)

    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        avg_conf = mean(c for c, _ in bucket)
        avg_acc = mean(a for _, a in bucket)
        weight = len(bucket) / n
        gap = abs(avg_conf - avg_acc)
        ece += weight * gap
        bin_stats.append(
            {
                "bin": i,
                "count": len(bucket),
                "avg_confidence": round(avg_conf, 4),
                "avg_actual_quality": round(avg_acc, 4),
                "gap": round(gap, 4),
            }
        )

    return ece, bin_stats
