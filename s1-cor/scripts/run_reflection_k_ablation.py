#!/usr/bin/env python3
"""
CPU ablation over reflection depth K (design.md §9 table).

Truncates parsed chain_sequence to K snapshots and measures mean
reward components — proxy for +Reflection (K=2) vs (K=3) before GPU training.

Also reports stage presets aligned with design.md Expected Results:
  - sft_baseline:      λ=0, μ=0  (external only dominant)
  - cor_self_rating:   λ=1, μ=0  (+ CoR intrinsic)
  - cor_reflection_k:  λ=1, μ=0.5, chains truncated to K

Usage:
    cd s1-cor
    python scripts/run_reflection_k_ablation.py --samples 30 --json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reflection rounds K ablation (CPU)")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--k-values", default="1,2,3,4", help="Max chain snapshots per sample")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _reward_for_chains(
    calc: RewardCalculator,
    chains: List[str],
    ground_truth: str,
) -> Dict[str, float]:
    if len(chains) <= 1:
        out = calc.calculate_total_reward(
            thinking_chain=chains[0] if chains else "",
            answer=ground_truth,
            ground_truth=ground_truth,
        )
    else:
        out = calc.calculate_reflection_reward(
            chain_sequence=chains,
            final_answer=ground_truth,
            ground_truth=ground_truth,
        )
    return {
        "total": out.total_reward,
        "external": out.external_reward,
        "intrinsic": out.intrinsic_reward,
        "improvement": out.improvement_reward,
        "convergence": out.convergence_reward,
        "rounds": float(out.reflection_rounds),
    }


def _mean_stats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: mean(r[k] for r in rows) for k in keys}


def run_k_ablation(rows: List[Dict[str, Any]], k_values: List[int]) -> List[Dict[str, Any]]:
    calc = RewardCalculator(
        RewardConfig(lambda_intrinsic=1.0, improvement_weight=0.5, convergence_weight=0.1)
    )
    results: List[Dict[str, Any]] = []

    for k in k_values:
        per_sample: List[Dict[str, float]] = []
        for row in rows:
            chains = extract_chain_sequence_from_sample(row)
            if not chains:
                continue
            truncated = chains[: max(1, k)]
            gt = row.get("attempt", "") or row.get("solution", "")
            per_sample.append(_reward_for_chains(calc, truncated, gt))

        stats = _mean_stats(per_sample)
        results.append(
            {
                "K": k,
                "n_samples": len(per_sample),
                **{f"mean_{key}": val for key, val in stats.items()},
            }
        )
    return results


def run_stage_presets(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """design.md §9 row names as reward-config presets (CPU proxy, not AIME scores)."""
    presets = [
        ("sft_baseline", RewardConfig(lambda_intrinsic=0.0, improvement_weight=0.0)),
        ("cor_self_rating", RewardConfig(lambda_intrinsic=1.0, improvement_weight=0.0)),
        ("cor_reflection", RewardConfig(lambda_intrinsic=1.0, improvement_weight=0.5)),
    ]
    out: List[Dict[str, Any]] = []

    for name, cfg in presets:
        calc = RewardCalculator(cfg)
        per_sample: List[Dict[str, float]] = []
        for row in rows:
            chains = extract_chain_sequence_from_sample(row)
            gt = row.get("attempt", "") or row.get("solution", "")
            if cfg.improvement_weight > 0 and len(chains) > 1:
                per_sample.append(_reward_for_chains(calc, chains, gt))
            else:
                thinking = chains[-1] if chains else ""
                per_sample.append(
                    _reward_for_chains(calc, [thinking], gt)
                )
        stats = _mean_stats(per_sample)
        out.append({"stage": name, "n_samples": len(per_sample), **stats})

    return out


def main() -> int:
    args = parse_args()
    k_values = _parse_int_list(args.k_values)

    path = f"local_data/s1K_cor_{args.dataset}"
    ds = load_cor_dataset_from_disk(path)
    rows = [ds[i] for i in range(min(args.samples, len(ds)))]

    payload = {
        "dataset": args.dataset,
        "samples": len(rows),
        "reflection_k_sweep": run_k_ablation(rows, k_values),
        "design_md_stage_presets": run_stage_presets(rows),
        "paper_benchmark_note": "AIME/MATH/GPQA numbers require GPU eval; see scripts/check_eval_readiness.py",
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Reflection K ablation ({len(rows)} samples)")
        for row in payload["reflection_k_sweep"]:
            print(
                f"  K={row['K']}: mean_total={row.get('mean_total', 0):.4f} "
                f"improve={row.get('mean_improvement', 0):.4f} rounds={row.get('mean_rounds', 0):.1f}"
            )
        print("\nStage presets (reward proxy):")
        for row in payload["design_md_stage_presets"]:
            print(f"  {row['stage']}: mean_total={row.get('mean_total', 0):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
