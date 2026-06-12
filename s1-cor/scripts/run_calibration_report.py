#!/usr/bin/env python3
"""
CPU φ calibration proxy report (dual-coupling deferred → measurable ECE-style metric).

Buckets self-ratings from s1K-cor samples and measures alignment with heuristic
actual quality scores. No separate φ head — documents calibration before GPU training.

Usage:
    cd s1-cor
    python scripts/run_calibration_report.py --dataset deepseek --samples 30
    python scripts/run_calibration_report.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from data_utils import load_cor_dataset_from_disk
from reflection_parsing import extract_chain_sequence_from_sample
from rewards import RewardCalculator, RewardConfig
from rewards.self_rating import SelfRatingExtractor, SelfRatingEvaluator
from rewards.intrinsic import IntrinsicRewardCalculator
from validate_cor_logic import extract_thinking_from_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-rating calibration proxy (ECE)")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def _chain_text(sample: Dict[str, Any]) -> str:
    chains = extract_chain_sequence_from_sample(sample)
    if chains:
        return chains[-1]
    text = sample.get("text_cor") or sample.get("text") or ""
    thinking = extract_thinking_from_text(text)
    return thinking or sample.get("thinking_rated") or ""


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


def run_report(dataset: str, n_samples: int, n_bins: int) -> Dict[str, Any]:
    data_path = f"local_data/s1K_cor_{dataset}"
    ds = load_cor_dataset_from_disk(data_path)
    n = min(n_samples, len(ds))

    extractor = SelfRatingExtractor()
    evaluator = SelfRatingEvaluator(calibration_bonus=0.2)
    intrinsic = IntrinsicRewardCalculator()

    confidences: List[float] = []
    dim_accuracies: List[float] = []
    overall_cals: List[float] = []
    rated_count = 0

    for i in range(n):
        sample = ds[i]
        chain = _chain_text(sample)
        if not chain:
            continue

        ratings = extractor.extract(chain)
        if not ratings:
            continue

        rated_count += 1
        actual = intrinsic.get_actual_qualities(chain)
        quality = evaluator.evaluate_self_rating_quality(
            ratings, actual, final_answer_correct=True
        )

        avg_self = mean(r.normalized for r in ratings.values())
        confidences.append(avg_self)
        dim_accuracies.append(quality["overall_calibration"])
        overall_cals.append(quality["overall_calibration"])

    ece, bins = compute_ece(confidences, dim_accuracies, n_bins)

    return {
        "layer": "verify",
        "report": "calibration_proxy",
        "dataset": dataset,
        "samples_scanned": n,
        "samples_with_self_rating": rated_count,
        "ece_proxy": round(ece, 4),
        "mean_overall_calibration": round(mean(overall_cals), 4) if overall_cals else 0.0,
        "bins": bins,
        "notes": [
            "Proxy for theory.md φ calibration; GRPO still updates θ only.",
            "actual_quality uses heuristic intrinsic dims, not lm-eval correctness.",
            "Track ece_proxy across training checkpoints on GPU hosts.",
        ],
    }


def main() -> int:
    args = parse_args()
    report = run_report(args.dataset, args.samples, args.bins)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Calibration proxy ({report['dataset']})")
        print(f"  rated samples: {report['samples_with_self_rating']}/{report['samples_scanned']}")
        print(f"  ECE proxy:     {report['ece_proxy']:.4f}")
        print(f"  mean cal:      {report['mean_overall_calibration']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
