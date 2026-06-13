"""CPU audit: self-rating vs heuristic intrinsic dimension scores."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from intrinsic_weights import DIMENSION_NAMES
from rewards.intrinsic import IntrinsicRewardCalculator
from rewards.self_rating import SelfRatingEvaluator, SelfRatingExtractor


def _pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _chain_text_from_row(row: Dict[str, Any], extract_chain_fn, extract_thinking_fn) -> str:
    chains = extract_chain_fn(row)
    if chains:
        return chains[-1]
    thinking = row.get("thinking_rated") or ""
    if not thinking and row.get("thinking_trajectories"):
        thinking = row["thinking_trajectories"][0]
    if not thinking:
        thinking = extract_thinking_fn(row.get("text_cor") or row.get("text") or "")
    return thinking


def build_correlation_report(
    rows: List[Dict[str, Any]],
    *,
    extract_chain_fn,
    extract_thinking_fn,
) -> Dict[str, Any]:
    extractor = SelfRatingExtractor()
    intrinsic = IntrinsicRewardCalculator()
    evaluator = SelfRatingEvaluator(calibration_bonus=0.2)

    per_dim_self: Dict[str, List[float]] = {d: [] for d in DIMENSION_NAMES}
    per_dim_actual: Dict[str, List[float]] = {d: [] for d in DIMENSION_NAMES}
    calibrations: List[float] = []
    rated_samples = 0

    for row in rows:
        chain = _chain_text_from_row(row, extract_chain_fn, extract_thinking_fn)
        if not chain:
            continue
        ratings = extractor.extract(chain)
        if not ratings:
            continue

        rated_samples += 1
        actual = intrinsic.get_actual_qualities(chain)
        quality = evaluator.evaluate_self_rating_quality(
            ratings, actual, final_answer_correct=True
        )
        calibrations.append(quality["overall_calibration"])

        for dim in DIMENSION_NAMES:
            if dim in ratings and dim in actual:
                per_dim_self[dim].append(ratings[dim].normalized)
                per_dim_actual[dim].append(actual[dim])

    pooled_self: List[float] = []
    pooled_actual: List[float] = []
    per_dimension: List[Dict[str, Any]] = []

    for dim in DIMENSION_NAMES:
        self_vals = per_dim_self[dim]
        actual_vals = per_dim_actual[dim]
        if not self_vals:
            continue
        pooled_self.extend(self_vals)
        pooled_actual.extend(actual_vals)
        mae = mean(abs(s - a) for s, a in zip(self_vals, actual_vals))
        dim_r = _pearson_r(self_vals, actual_vals)
        per_dimension.append(
            {
                "dimension": dim,
                "pairs": len(self_vals),
                "mean_self": round(mean(self_vals), 4),
                "mean_actual_heuristic": round(mean(actual_vals), 4),
                "mae": round(mae, 4),
                "pearson_r": round(dim_r, 4) if dim_r is not None else None,
            }
        )

    pooled_r = _pearson_r(pooled_self, pooled_actual)

    return {
        "samples_scanned": len(rows),
        "samples_with_self_rating": rated_samples,
        "self_rating_coverage": rated_samples / len(rows) if rows else 0.0,
        "mean_overall_calibration": round(mean(calibrations), 4) if calibrations else 0.0,
        "pooled_pearson_r": round(pooled_r, 4) if pooled_r is not None else None,
        "per_dimension": per_dimension,
        "most_miscalibrated_dimension": (
            max(per_dimension, key=lambda d: d["mae"])["dimension"] if per_dimension else None
        ),
    }
