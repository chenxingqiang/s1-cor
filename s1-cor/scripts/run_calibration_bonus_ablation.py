#!/usr/bin/env python3
"""
CPU ablation over calibration_bonus α (theory.md high-high alignment bonus).

Measures ECE proxy and mean calibration vs RewardConfig.calibration_bonus
before GPU GRPO. Complements run_calibration_report.py single-point snapshot.

Usage:
    cd s1-cor
    python scripts/run_calibration_bonus_ablation.py --json --samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from calibration_metrics import compute_ece
from data_utils import load_cor_dataset_from_disk
from reflection_parsing import extract_chain_sequence_from_sample
from rewards.intrinsic import IntrinsicRewardCalculator
from rewards.self_rating import SelfRatingExtractor, SelfRatingEvaluator
from validate_cor_logic import extract_thinking_from_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration bonus α ablation (CPU)")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--alpha-values", default="0.0,0.1,0.2,0.4", help="calibration_bonus sweep")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _chain_text(sample: Dict[str, Any]) -> str:
    chains = extract_chain_sequence_from_sample(sample)
    if chains:
        return chains[-1]
    text = sample.get("text_cor") or sample.get("text") or ""
    thinking = extract_thinking_from_text(text)
    return thinking or sample.get("thinking_rated") or ""


def _collect_rows(dataset: str, n: int) -> List[Dict[str, Any]]:
    ds = load_cor_dataset_from_disk(f"local_data/s1K_cor_{dataset}")
    rows = []
    for i in range(min(n, len(ds))):
        sample = ds[i]
        chain = _chain_text(sample)
        if not chain:
            continue
        extractor = SelfRatingExtractor()
        ratings = extractor.extract(chain)
        if not ratings:
            continue
        intrinsic = IntrinsicRewardCalculator()
        actual = intrinsic.get_actual_qualities(chain)
        rows.append({"chain": chain, "ratings": ratings, "actual": actual})
    return rows


def run_ablation(dataset: str, n_samples: int, alphas: List[float]) -> Dict[str, Any]:
    rows = _collect_rows(dataset, n_samples)
    results: List[Dict[str, Any]] = []

    for alpha in alphas:
        evaluator = SelfRatingEvaluator(calibration_bonus=alpha)
        confidences: List[float] = []
        accuracies: List[float] = []
        overall_cals: List[float] = []

        for row in rows:
            quality = evaluator.evaluate_self_rating_quality(
                row["ratings"], row["actual"], final_answer_correct=True
            )
            avg_self = mean(r.normalized for r in row["ratings"].values())
            confidences.append(avg_self)
            accuracies.append(quality["overall_calibration"])
            overall_cals.append(quality["overall_calibration"])

        ece, _ = compute_ece(confidences, accuracies)
        results.append(
            {
                "calibration_bonus": alpha,
                "samples_with_self_rating": len(rows),
                "ece_proxy": round(ece, 4),
                "mean_overall_calibration": round(mean(overall_cals), 4) if overall_cals else 0.0,
            }
        )

    best = min(results, key=lambda r: r["ece_proxy"]) if results else None

    return {
        "layer": "verify",
        "report": "calibration_bonus_ablation",
        "dataset": dataset,
        "samples_requested": n_samples,
        "sweep": results,
        "best_by_ece": best,
        "notes": [
            "α is RewardConfig.calibration_bonus (high-high alignment bonus).",
            "Proxy only; φ head still deferred — GRPO updates θ only.",
        ],
    }


def main() -> int:
    args = parse_args()
    alphas = _parse_float_list(args.alpha_values)
    payload = run_ablation(args.dataset, args.samples, alphas)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Calibration bonus ablation ({payload['dataset']})")
        for row in payload["sweep"]:
            print(
                f"  α={row['calibration_bonus']:.2f} "
                f"ECE={row['ece_proxy']:.4f} "
                f"mean_cal={row['mean_overall_calibration']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
