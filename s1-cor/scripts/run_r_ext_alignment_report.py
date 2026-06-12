#!/usr/bin/env python3
"""
CPU report: training R_ext (string match) vs eval-aligned math grading gap.

Compares default RewardCalculator external reward with math grader on local CoR data.
Documents train/eval disagreement before GPU benchmark runs.

Usage:
    cd s1-cor
    python scripts/run_r_ext_alignment_report.py --dataset deepseek --samples 20
    python scripts/run_r_ext_alignment_report.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from answer_grading import extract_answer_from_completion
from data_utils import load_cor_dataset_from_disk
from rewards import RewardCalculator, RewardConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R_ext train vs eval grading alignment report")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def _completion_from_sample(sample: Dict[str, Any]) -> str:
    return sample.get("attempt") or sample.get("solution") or ""


def _ground_truth(sample: Dict[str, Any]) -> str:
    return sample.get("solution") or sample.get("attempt") or ""


def run_report(dataset: str, n_samples: int) -> Dict[str, Any]:
    data_path = f"local_data/s1K_cor_{dataset}"
    ds = load_cor_dataset_from_disk(data_path)
    n = min(n_samples, len(ds))

    string_calc = RewardCalculator(RewardConfig(use_math_grader=False))
    math_calc = RewardCalculator(RewardConfig(use_math_grader=True))

    disagreements: List[Dict[str, Any]] = []
    string_correct = 0
    math_correct = 0
    both_correct = 0
    both_wrong = 0

    for i in range(n):
        sample = ds[i]
        completion = _completion_from_sample(sample)
        gt = _ground_truth(sample)
        pred = extract_answer_from_completion(completion)

        r_string = string_calc.calculate_external_reward(pred, gt)
        r_math = math_calc.calculate_external_reward(pred, gt)

        if r_string > 0.5:
            string_correct += 1
        if r_math > 0.5:
            math_correct += 1
        if r_string > 0.5 and r_math > 0.5:
            both_correct += 1
        elif r_string <= 0.5 and r_math <= 0.5:
            both_wrong += 1

        if (r_string > 0.5) != (r_math > 0.5):
            disagreements.append(
                {
                    "index": i,
                    "string_reward": r_string,
                    "math_reward": r_math,
                    "pred_excerpt": pred[:120],
                    "gt_excerpt": extract_answer_from_completion(gt)[:120],
                }
            )

    return {
        "layer": "verify",
        "report": "r_ext_alignment",
        "dataset": dataset,
        "samples": n,
        "string_match_accuracy": string_correct / n if n else 0.0,
        "math_grader_accuracy": math_correct / n if n else 0.0,
        "agreement_rate": 1.0 - (len(disagreements) / n if n else 0.0),
        "disagreement_count": len(disagreements),
        "disagreements_sample": disagreements[:5],
        "notes": [
            "Pred=attempt, GT=solution (formatting gap between model output and reference).",
            "Enable RewardConfig.use_math_grader=True or USE_MATH_GRADER=1 for GRPO.",
            "MATH/GPQA OpenAI judge in eval/commands.sh remains a separate gap.",
        ],
    }


def main() -> int:
    args = parse_args()
    report = run_report(args.dataset, args.samples)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"R_ext alignment ({report['dataset']}, n={report['samples']})")
        print(f"  string match acc: {report['string_match_accuracy']:.3f}")
        print(f"  math grader acc:  {report['math_grader_accuracy']:.3f}")
        print(f"  agreement:        {report['agreement_rate']:.3f}")
        print(f"  disagreements:    {report['disagreement_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
