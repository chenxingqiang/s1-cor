#!/usr/bin/env python3
"""
CPU report: self-rating vs heuristic intrinsic r_d alignment.

Documents five_dim_intrinsic partial tier — model self-assessment vs rule-based
actual quality proxies before GPU training.

Usage:
    cd s1-cor
    python scripts/run_self_rating_intrinsic_correlation_report.py --json --samples 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from data_utils import load_cor_dataset_from_disk
from reflection_parsing import extract_chain_sequence_from_sample
from self_rating_intrinsic_correlation_audit import build_correlation_report
from validate_cor_logic import extract_thinking_from_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-rating vs intrinsic correlation (CPU)")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def run_report(dataset: str, n_samples: int) -> Dict[str, Any]:
    ds = load_cor_dataset_from_disk(f"local_data/s1K_cor_{dataset}")
    rows = [ds[i] for i in range(min(n_samples, len(ds)))]
    stats = build_correlation_report(
        rows,
        extract_chain_fn=extract_chain_sequence_from_sample,
        extract_thinking_fn=extract_thinking_from_text,
    )
    return {
        "layer": "verify",
        "report": "self_rating_intrinsic_correlation",
        "dataset": dataset,
        **stats,
        "notes": [
            "actual = heuristic IntrinsicRewardCalculator r_d (not learned Q_phi).",
            "High MAE / low Pearson → self-rating poorly aligned with proxies.",
            "Complements run_calibration_report.py ECE proxy.",
        ],
    }


def main() -> int:
    args = parse_args()
    payload = run_report(args.dataset, args.samples)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Self-rating vs intrinsic ({payload['dataset']}, "
            f"rated={payload['samples_with_self_rating']}/{payload['samples_scanned']})"
        )
        print(f"  pooled Pearson r: {payload['pooled_pearson_r']}")
        print(f"  mean calibration: {payload['mean_overall_calibration']:.4f}")
        for row in payload["per_dimension"]:
            print(
                f"  {row['dimension']}: MAE={row['mae']:.3f} "
                f"r={row['pearson_r']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
