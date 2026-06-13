#!/usr/bin/env python3
"""
CPU audit for benchmark reproduction chain (design.md §9 / README Results).

Aggregates check_eval_readiness + compare_eval_to_paper fixture smoke.
Does not run vLLM or download checkpoints.

Usage:
    cd s1-cor
    python scripts/run_benchmark_reproduction_report.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from benchmark_reproduction_audit import build_audit_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark reproduction CPU audit")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()

    report = build_audit_report(tolerance=args.tolerance)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Benchmark reproduction audit")
        print(f"  cpu_audit_ok: {report['cpu_audit_ok']}")
        print(f"  gpu_reproduction_ready: {report['gpu_reproduction_ready']}")
        for step in report["reproduction_steps"]:
            mark = "✓" if step["ready_on_host"] else "✗"
            print(f"  {mark} {step['id']}: {step['title']}")
        if report["readiness"]["blockers"]:
            print("  blockers:")
            for b in report["readiness"]["blockers"][:3]:
                print(f"    - {b}")
    return 0 if report["cpu_audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
