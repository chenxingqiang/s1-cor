#!/usr/bin/env python3
"""
CPU theory verification for CoR on Qwen2.5-0.5B (smallest open scale).

Combines design.md §9 stage ladder (SFT → +CoR → +Reflection) and
reflection depth K ablation as reward proxies — not AIME benchmark scores.
GPU full pipeline: docs/MODEL_05B_TEST.md, train/grpo_05b.sh.

Usage:
    cd s1-cor
    python scripts/run_05b_theory_verify.py --json
    make loop-05b-theory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_cor_dataset_from_disk
from run_reflection_k_ablation import run_k_ablation, run_stage_presets

MODEL_SIZE = "0.5B"
HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Aligned with train/sft_small.py MODEL_CONFIGS["0.5B"] and run_scale_experiments.sh
MODEL_TRAINING_CONFIG: Dict[str, Any] = {
    "model_size": MODEL_SIZE,
    "hf_model_id": HF_MODEL_ID,
    "sft": {
        "batch_size": 4,
        "grad_accum": 4,
        "max_length": 4096,
        "lr": 2e-5,
        "entry": "python train/sft_small.py --model_size 0.5B",
    },
    "grpo": {
        "batch_size": 4,
        "grad_accum": 4,
        "block_size": 4096,
        "num_generations": 8,
        "lr": 5e-6,
        "lambda_intrinsic": 1.0,
        "self_rating_weight": 0.2,
        "improvement_weight": 0.5,
        "convergence_weight": 0.1,
        "max_reflection_rounds": 3,
        "entry": "bash train/grpo_05b.sh",
        "scale_entry": "bash train/run_scale_experiments.sh 0.5B",
    },
    "vram_note_gb": "~1GB (single GPU)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="0.5B CoR theory CPU verify")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--k-values", default="1,2,3")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit 1 if theory checks fail")
    return p.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _stage_totals(presets: List[Dict[str, Any]]) -> Dict[str, float]:
    return {row["stage"]: float(row.get("mean_total", 0.0)) for row in presets}


def evaluate_theory_checks(
    stage_presets: List[Dict[str, Any]],
    k_sweep: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool]:
    totals = _stage_totals(stage_presets)
    sft = totals.get("sft_baseline", 0.0)
    cor = totals.get("cor_self_rating", 0.0)
    refl = totals.get("cor_reflection", 0.0)

    k_by_k = {int(row["K"]): float(row.get("mean_total", 0.0)) for row in k_sweep}
    k_min = min(k_by_k) if k_by_k else 1
    k_max = max(k_by_k) if k_by_k else 1

    checks: Dict[str, Any] = {
        "stage_ladder_cor_ge_sft": {
            "ok": cor >= sft - 1e-9,
            "detail": f"cor_self_rating={cor:.4f} vs sft_baseline={sft:.4f}",
        },
        "stage_ladder_reflection_ge_cor": {
            "ok": refl >= cor - 1e-9,
            "detail": f"cor_reflection={refl:.4f} vs cor_self_rating={cor:.4f}",
        },
        "reflection_k_non_decreasing_max": {
            "ok": k_by_k.get(k_max, 0.0) >= k_by_k.get(k_min, 0.0) - 1e-9,
            "detail": f"K={k_min} total={k_by_k.get(k_min, 0):.4f} vs K={k_max} total={k_by_k.get(k_max, 0):.4f}",
        },
    }
    all_ok = all(c["ok"] for c in checks.values())
    return checks, all_ok


def build_report(dataset: str, samples: int, k_values: List[int]) -> Dict[str, Any]:
    path = f"local_data/s1K_cor_{dataset}"
    ds = load_cor_dataset_from_disk(path)
    rows = [ds[i] for i in range(min(samples, len(ds)))]

    k_sweep = run_k_ablation(rows, k_values)
    stage_presets = run_stage_presets(rows)
    theory_checks, theory_ok = evaluate_theory_checks(stage_presets, k_sweep)

    return {
        "layer": "verify",
        "report": "05b_theory_verify",
        "model": MODEL_TRAINING_CONFIG,
        "dataset": dataset,
        "samples": len(rows),
        "reflection_k_sweep": k_sweep,
        "design_md_stage_presets": stage_presets,
        "theory_checks": theory_checks,
        "theory_ok": theory_ok,
        "interpretation": {
            "cpu_proxy": "Reward means on bundled chain_sequence; not model-generated 0.5B outputs.",
            "expected_ladder": "sft_baseline ≤ cor_self_rating ≤ cor_reflection (design.md §9)",
            "gpu_next": "bash train/grpo_05b.sh after SFT; eval via eval/commands.sh cor-0.5B line",
        },
        "paper_benchmark_note": "AIME/MATH/GPQA require GPU ckpt + lm_eval; see docs/MODEL_05B_TEST.md",
    }


def main() -> int:
    args = parse_args()
    k_values = _parse_int_list(args.k_values)
    payload = build_report(args.dataset, args.samples, k_values)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"0.5B theory verify ({payload['samples']} samples, {args.dataset})")
        print(f"Model: {HF_MODEL_ID}")
        for row in payload["design_md_stage_presets"]:
            print(f"  {row['stage']}: mean_total={row.get('mean_total', 0):.4f}")
        print("Theory checks:")
        for name, chk in payload["theory_checks"].items():
            mark = "OK" if chk["ok"] else "FAIL"
            print(f"  [{mark}] {name}: {chk['detail']}")

    if args.strict and not payload["theory_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
