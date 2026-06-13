#!/usr/bin/env python3
"""
CPU report: training answer_grading vs lm-eval pre-OpenAI extraction path.

Measures when MATH500/GPQA eval would rely on OpenAI after regex/boxed fail.
See docs/TRAIN_EVAL_GRADING.md.

Usage:
    cd s1-cor
    python scripts/run_eval_grading_path_report.py --json --samples 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from data_utils import load_cor_dataset_from_disk
from eval_grading_path_audit import build_path_alignment_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vs eval pre-OpenAI grading path")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def _rows_from_dataset(dataset: str, n: int) -> List[Dict[str, Any]]:
    ds = load_cor_dataset_from_disk(f"local_data/s1K_cor_{dataset}")
    return [ds[i] for i in range(min(n, len(ds)))]


def run_report(dataset: str, n_samples: int) -> Dict[str, Any]:
    rows = _rows_from_dataset(dataset, n_samples)
    alignment = build_path_alignment_report(rows)
    return {
        "layer": "verify",
        "report": "eval_grading_path",
        "dataset": dataset,
        **alignment,
        "notes": [
            "train path: answer_grading.extract_answer_from_completion + sympy.",
            "eval path: lm-eval openai_math pre-OpenAI order (boxed → Answer: regex).",
            "openai_fallback_likely_count = train ok but eval pre-OpenAI fail.",
        ],
    }


def main() -> int:
    args = parse_args()
    payload = run_report(args.dataset, args.samples)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Eval grading path ({payload['dataset']}, n={payload['samples']})")
        print(f"  train math acc:     {payload['train_math_accuracy']:.3f}")
        print(f"  eval pre-OAI acc:   {payload['eval_pre_openai_accuracy']:.3f}")
        print(f"  path agreement:     {payload['path_agreement_rate']:.3f}")
        print(f"  OpenAI likely:      {payload['openai_fallback_likely_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
