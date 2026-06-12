#!/usr/bin/env python3
"""
CPU ablation over five intrinsic dimensions (design.md / theory.md §2).

Documents heuristic R_int sensitivity before GPU training:
  - uniform: w_d = 0.2 each (paper default)
  - emphasize_<dim>: single dimension weight = 1.0
  - drop_<dim>: zero one dimension, renormalize others

Usage:
    cd s1-cor
    python scripts/run_intrinsic_dim_ablation.py --samples 20 --json
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

DIMS = ("consistency", "completeness", "accuracy", "clarity", "format")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Five-dim intrinsic weight ablation (CPU)")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=15)
    p.add_argument(
        "--presets",
        default="uniform,emphasize,drop",
        help="Comma-separated: uniform, emphasize, drop",
    )
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


def _weight_presets(mode: str) -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    if mode == "uniform":
        presets.append(
            {
                "name": "uniform_w0.2",
                "weights": {d: 0.2 for d in DIMS},
            }
        )
    elif mode == "emphasize":
        for dim in DIMS:
            presets.append(
                {
                    "name": f"emphasize_{dim}",
                    "weights": {d: (1.0 if d == dim else 0.0) for d in DIMS},
                }
            )
    elif mode == "drop":
        for dim in DIMS:
            presets.append(
                {
                    "name": f"drop_{dim}",
                    "weights": {d: (0.0 if d == dim else 0.2) for d in DIMS},
                }
            )
    return presets


def _mean_intrinsic_stats(
    rows: List[Dict[str, Any]],
    calc: RewardCalculator,
) -> Dict[str, float]:
    intrinsics: List[float] = []
    totals: List[float] = []
    dim_sums = {d: 0.0 for d in DIMS}

    for row in rows:
        thinking = _thinking_from_row(row)
        gt = row.get("attempt") or row.get("solution") or ""
        chains = extract_chain_sequence_from_sample(row)
        if len(chains) > 1:
            out = calc.calculate_reflection_reward(chains, gt, gt)
        else:
            out = calc.calculate_total_reward(thinking, gt, gt)
        intrinsics.append(out.intrinsic_reward)
        totals.append(out.total_reward)
        for d, v in out.dimension_scores.items():
            if d in dim_sums:
                dim_sums[d] += v

    n = len(rows) or 1
    return {
        "mean_intrinsic": mean(intrinsics) if intrinsics else 0.0,
        "mean_total": mean(totals) if totals else 0.0,
        "mean_dim_scores": {d: dim_sums[d] / n for d in DIMS},
        "n_samples": len(rows),
    }


def run_ablation(dataset: str, n_samples: int, preset_modes: List[str]) -> Dict[str, Any]:
    path = f"local_data/s1K_cor_{dataset}"
    ds = load_cor_dataset_from_disk(path)
    rows = [ds[i] for i in range(min(n_samples, len(ds)))]

    preset_defs: List[Dict[str, Any]] = []
    for mode in preset_modes:
        preset_defs.extend(_weight_presets(mode.strip()))

    results: List[Dict[str, Any]] = []
    for preset in preset_defs:
        cfg = RewardConfig(
            lambda_intrinsic=1.0,
            dimension_weights=preset["weights"],
        )
        calc = RewardCalculator(cfg)
        stats = _mean_intrinsic_stats(rows, calc)
        results.append(
            {
                "preset": preset["name"],
                "weights": preset["weights"],
                **stats,
            }
        )

    uniform = next((r for r in results if r["preset"] == "uniform_w0.2"), None)
    sensitivities: Dict[str, float] = {}
    if uniform:
        base = uniform["mean_intrinsic"]
        for dim in DIMS:
            emph = next((r for r in results if r["preset"] == f"emphasize_{dim}"), None)
            if emph:
                sensitivities[dim] = round(emph["mean_intrinsic"] - base, 4)

    return {
        "layer": "verify",
        "report": "intrinsic_dim_ablation",
        "dataset": dataset,
        "samples": len(rows),
        "presets": results,
        "emphasis_delta_vs_uniform": sensitivities,
        "notes": [
            "Heuristic dimension scorers; not learned Q_phi.",
            "Use before GPU GRPO to sanity-check λ·R_int scale vs R_ext.",
            "Default weights match IntrinsicRewardCalculator.DEFAULT_WEIGHTS.",
        ],
    }


def main() -> int:
    args = parse_args()
    modes = [m.strip() for m in args.presets.split(",") if m.strip()]
    payload = run_ablation(args.dataset, args.samples, modes)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Intrinsic dim ablation ({payload['dataset']}, n={payload['samples']})")
        for row in payload["presets"]:
            print(
                f"  {row['preset']}: R_int={row['mean_intrinsic']:.4f} "
                f"R_total={row['mean_total']:.4f}"
            )
        if payload["emphasis_delta_vs_uniform"]:
            print("  emphasis Δ vs uniform:", payload["emphasis_delta_vs_uniform"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
