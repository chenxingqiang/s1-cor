#!/usr/bin/env python3
"""
CPU contract gate for five_dim_intrinsic (matrix partial tier).

Aggregates self-rating correlation + dimension weight ablation into
contract_checks JSON for loop-perceive / publication honesty.

Usage:
    cd s1-cor
    python scripts/run_five_dim_contract_report.py --json --strict
    make loop-five-dim-contract
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_cor_dataset_from_disk
from five_dim_contract_audit import build_five_dim_contract_report
from reflection_parsing import extract_chain_sequence_from_sample
from run_intrinsic_dim_ablation import run_ablation
from self_rating_intrinsic_correlation_audit import build_correlation_report
from validate_cor_logic import extract_thinking_from_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Five-dim intrinsic contract (CPU)")
    p.add_argument("--dataset", default="deepseek", choices=["deepseek", "full", "gemini_test"])
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit 1 if contract_ok is false")
    return p.parse_args()


def run_report(dataset: str, n_samples: int) -> dict:
    ds = load_cor_dataset_from_disk(f"local_data/s1K_cor_{dataset}")
    rows = [ds[i] for i in range(min(n_samples, len(ds)))]

    correlation = build_correlation_report(
        rows,
        extract_chain_fn=extract_chain_sequence_from_sample,
        extract_thinking_fn=extract_thinking_from_text,
    )
    ablation = run_ablation(dataset, n_samples, ["uniform", "emphasize", "drop"])
    payload = build_five_dim_contract_report(correlation, ablation)
    payload["dataset"] = dataset
    payload["samples"] = len(rows)
    return payload


def main() -> int:
    args = parse_args()
    payload = run_report(args.dataset, args.samples)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Five-dim contract ({payload['samples']} samples)")
        print(f"  contract_ok: {payload['contract_ok']}")
        for name, chk in payload["contract_checks"].items():
            mark = "OK" if chk["ok"] else "FAIL"
            print(f"  [{mark}] {name}: {chk['detail']}")

    if args.strict and not payload["contract_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
