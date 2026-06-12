#!/usr/bin/env python3
"""
CPU smoke for GRPO reward_fn wiring (no GPU / no model load).

Validates create_reward_fn with string vs math R_ext on local s1K-cor samples
before launching torchrun on a GPU host.

Usage:
    cd s1-cor
    python scripts/run_grpo_reward_smoke.py --samples 5
    python scripts/run_grpo_reward_smoke.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from statistics import mean
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO reward_fn CPU smoke")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def run_smoke(dataset: str, n_samples: int) -> Dict[str, Any]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

    from data_utils import load_cor_dataset_from_disk
    from grpo import CoRTrainingConfig, create_reward_fn

    data_path = f"local_data/s1K_cor_{dataset}"
    ds = load_cor_dataset_from_disk(data_path)
    n = min(n_samples, len(ds))

    cfg_string = CoRTrainingConfig(use_math_grader=False, enable_reflection=True)
    cfg_math = CoRTrainingConfig(use_math_grader=True, enable_reflection=True)
    fn_string = create_reward_fn(cfg_string, enable_logging=False)
    fn_math = create_reward_fn(cfg_math, enable_logging=False)

    deltas: List[float] = []
    disagree = 0

    for i in range(n):
        sample = ds[i]
        completion = sample.get("attempt") or sample.get("solution") or ""
        gt = sample.get("solution") or sample.get("attempt") or ""
        if not completion or not gt:
            continue

        r_str = fn_string([completion], reference_answer=[gt])[0]
        r_math = fn_math([completion], reference_answer=[gt])[0]
        deltas.append(r_math - r_str)
        if abs(r_math - r_str) > 1e-6:
            disagree += 1

    evaluated = len(deltas)
    return {
        "layer": "verify",
        "report": "grpo_reward_smoke",
        "dataset": dataset,
        "samples_requested": n,
        "samples_evaluated": evaluated,
        "use_math_grader_wired": True,
        "mean_reward_delta_math_minus_string": round(mean(deltas), 6) if deltas else 0.0,
        "reward_disagreement_count": disagree,
        "grpo_flags": {
            "USE_MATH_GRADER": "set to 1 in grpo.sh / run_cor_pipeline.sh for GPU training",
            "cli": "--use_math_grader=True on train/grpo.py",
        },
        "notes": [
            "Uses bundled attempt vs solution as completion/GT proxy.",
            "Non-zero delta indicates math grader changes R_ext component.",
            "Full benchmark still requires GPU ckpt + eval/commands.sh.",
        ],
    }


def main() -> int:
    args = parse_args()
    if args.json:
        logging.disable(logging.CRITICAL)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    report = run_smoke(args.dataset, args.samples)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"GRPO reward smoke ({report['dataset']}, n={report['samples_evaluated']})")
        print(f"  mean Δ(math-string): {report['mean_reward_delta_math_minus_string']}")
        print(f"  disagreements:       {report['reward_disagreement_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
