#!/usr/bin/env python3
"""
CPU ablation sweep over CoR reward hyperparameters (no training).

Supports paper-style sensitivity analysis before GPU GRPO runs:
  - lambda_intrinsic (λ)
  - improvement_weight (μ)
  - convergence_alpha (α)

Usage:
    cd s1-cor
    python scripts/run_ablation_sweep.py --dataset deepseek --samples 20
    python scripts/run_ablation_sweep.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from statistics import mean
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from data_utils import load_cor_dataset_from_disk
from rewards import RewardCalculator, RewardConfig
from reflection_parsing import extract_chain_sequence_from_sample
from validate_cor_logic import extract_thinking_from_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoR reward hyperparameter ablation (CPU)")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=10)
    p.add_argument(
        "--lambda-values", default="0.0,0.5,1.0,2.0",
        help="Comma-separated λ sweep",
    )
    p.add_argument(
        "--mu-values", default="0.0,0.25,0.5,1.0",
        help="Comma-separated μ sweep",
    )
    p.add_argument(
        "--alpha-values", default="0.5,1.0,2.0",
        help="Comma-separated α (convergence) sweep",
    )
    p.add_argument("--json", action="store_true", help="Print JSON only")
    return p.parse_args()


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def load_rows(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    path = f"local_data/s1K_cor_{dataset_name}"
    ds = load_cor_dataset_from_disk(path)
    return [ds[i] for i in range(min(n, len(ds)))]


def _thinking_from_row(row: Dict[str, Any]) -> str:
    thinking = row.get("thinking_rated") or row.get("thinking_trajectories", [""])[0]
    if not thinking:
        thinking = extract_thinking_from_text(row.get("text", "") or row.get("text_cor", ""))
    return thinking


def mean_reward_for_config(
    rows: List[Dict[str, Any]],
    cfg: RewardConfig,
) -> Dict[str, float]:
    calc = RewardCalculator(cfg)

    totals: List[float] = []
    externals: List[float] = []
    intrinsics: List[float] = []
    improves: List[float] = []
    converges: List[float] = []

    for row in rows:
        thinking = _thinking_from_row(row)
        ground_truth = row.get("attempt", "") or row.get("solution", "")
        chains = extract_chain_sequence_from_sample(row)
        if len(chains) > 1:
            out = calc.calculate_reflection_reward(
                chain_sequence=chains,
                final_answer=ground_truth,
                ground_truth=ground_truth,
            )
        else:
            out = calc.calculate_total_reward(
                thinking_chain=thinking,
                answer=ground_truth,
                ground_truth=ground_truth,
            )
        totals.append(out.total_reward)
        externals.append(out.external_reward)
        intrinsics.append(out.intrinsic_reward)
        improves.append(out.improvement_reward)
        converges.append(out.convergence_reward)

    return {
        "mean_total": mean(totals),
        "mean_external": mean(externals),
        "mean_intrinsic": mean(intrinsics),
        "mean_improvement": mean(improves),
        "mean_convergence": mean(converges),
        "n_samples": len(rows),
    }


def main() -> int:
    args = parse_args()
    lambdas = _parse_float_list(args.lambda_values)
    mus = _parse_float_list(args.mu_values)
    alphas = _parse_float_list(args.alpha_values)

    rows = load_rows(args.dataset, args.samples)
    results: List[Dict[str, Any]] = []

    for lam, mu, alpha in product(lambdas, mus, alphas):
        cfg = RewardConfig(
            lambda_intrinsic=lam,
            improvement_weight=mu,
            convergence_alpha=alpha,
        )
        stats = mean_reward_for_config(rows, cfg)
        entry = {
            "lambda_intrinsic": lam,
            "improvement_weight": mu,
            "convergence_alpha": alpha,
            **stats,
        }
        results.append(entry)

    payload = {
        "dataset": args.dataset,
        "samples": len(rows),
        "sweep": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Ablation on {len(rows)} samples ({args.dataset})")
        for r in results:
            print(
                f"  λ={r['lambda_intrinsic']:.2f} μ={r['improvement_weight']:.2f} "
                f"α={r['convergence_alpha']:.2f} → total={r['mean_total']:.4f} "
                f"(ext={r['mean_external']:.3f} int={r['mean_intrinsic']:.3f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
