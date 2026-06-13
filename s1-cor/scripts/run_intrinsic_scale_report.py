#!/usr/bin/env python3
"""
CPU report: R_ext vs R_int component scale on local CoR data.

Helps pick λ before GPU GRPO when five_dim_intrinsic uses heuristic scorers.
Complements run_intrinsic_dim_ablation.py (w_d sensitivity).

Usage:
    cd s1-cor
    python scripts/run_intrinsic_scale_report.py --json --samples 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from data_utils import load_cor_dataset_from_disk
from reflection_parsing import extract_chain_sequence_from_sample
from rewards import RewardCalculator, RewardConfig
from validate_cor_logic import extract_thinking_from_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intrinsic vs external reward scale (CPU)")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--lambda-intrinsic", type=float, default=1.0)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def _thinking_from_row(row: Dict[str, Any]) -> str:
    chains = extract_chain_sequence_from_sample(row)
    if chains:
        return chains[-1]
    thinking = row.get("thinking_rated") or ""
    if not thinking and row.get("thinking_trajectories"):
        thinking = row["thinking_trajectories"][0]
    if not thinking:
        thinking = extract_thinking_from_text(row.get("text_cor") or row.get("text") or "")
    return thinking


def run_report(dataset: str, n_samples: int, lambda_intrinsic: float) -> Dict[str, Any]:
    ds = load_cor_dataset_from_disk(f"local_data/s1K_cor_{dataset}")
    rows = [ds[i] for i in range(min(n_samples, len(ds)))]

    cfg = RewardConfig(lambda_intrinsic=lambda_intrinsic)
    calc = RewardCalculator(cfg)

    ext_vals: List[float] = []
    int_vals: List[float] = []
    imp_vals: List[float] = []
    conv_vals: List[float] = []
    total_vals: List[float] = []
    multi_round = 0

    for row in rows:
        gt = row.get("attempt") or row.get("solution") or ""
        chains = extract_chain_sequence_from_sample(row)
        if len(chains) > 1:
            multi_round += 1
            out = calc.calculate_reflection_reward(chains, gt, gt)
        else:
            thinking = _thinking_from_row(row)
            out = calc.calculate_total_reward(thinking, gt, gt)

        ext_vals.append(out.external_reward)
        int_vals.append(out.intrinsic_reward)
        imp_vals.append(out.improvement_reward)
        conv_vals.append(out.convergence_reward)
        total_vals.append(out.total_reward)

    mean_ext = mean(ext_vals) if ext_vals else 0.0
    mean_int = mean(int_vals) if int_vals else 0.0
    mean_imp = mean(imp_vals) if imp_vals else 0.0
    mean_conv = mean(conv_vals) if conv_vals else 0.0
    mean_total = mean(total_vals) if total_vals else 0.0

    int_contrib = lambda_intrinsic * mean_int
    lambda_hint = None
    if mean_int > 1e-6:
        lambda_hint = round(min(2.0, max(0.25, mean_ext / mean_int)), 3)

    return {
        "layer": "verify",
        "report": "intrinsic_scale",
        "dataset": dataset,
        "samples": len(rows),
        "lambda_intrinsic": lambda_intrinsic,
        "mean_external": round(mean_ext, 4),
        "mean_intrinsic": round(mean_int, 4),
        "mean_improvement": round(mean_imp, 4),
        "mean_convergence": round(mean_conv, 4),
        "mean_total": round(mean_total, 4),
        "intrinsic_contribution": round(int_contrib, 4),
        "intrinsic_fraction_of_total": round(int_contrib / mean_total, 4) if mean_total else 0.0,
        "multi_round_samples": multi_round,
        "suggested_lambda_intrinsic": lambda_hint,
        "notes": [
            "Heuristic five_dim_intrinsic; suggested_lambda balances |R_ext| vs |R_int| on local data.",
            "Use with run_intrinsic_dim_ablation.py before GPU GRPO.",
            "φ / token-level CoR still deferred — see docs/FIVE_DIM_INTRINSIC.md.",
        ],
    }


def main() -> int:
    args = parse_args()
    payload = run_report(args.dataset, args.samples, args.lambda_intrinsic)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Intrinsic scale ({payload['dataset']}, n={payload['samples']})")
        print(f"  mean R_ext={payload['mean_external']:.4f} R_int={payload['mean_intrinsic']:.4f}")
        print(f"  mean R_total={payload['mean_total']:.4f}")
        print(f"  suggested λ={payload['suggested_lambda_intrinsic']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
