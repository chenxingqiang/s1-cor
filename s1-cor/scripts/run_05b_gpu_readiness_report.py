#!/usr/bin/env python3
"""
CPU bridge report: 0.5B theory (R19) → GPU SFT/GRPO/eval (R21).

Documents ordered pipeline + host blockers without downloading weights.

Usage:
    cd s1-cor
    python scripts/run_05b_gpu_readiness_report.py --json
    make loop-05b-gpu-ready
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from scale_05b_gpu_audit import build_scale_05b_gpu_report

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _run_theory_verify(samples: int = 5) -> bool | None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_05b_theory_verify.py",
            "--json",
            "--strict",
            "--samples",
            str(samples),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    try:
        data = json.loads(proc.stdout)
        return bool(data.get("theory_ok"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="0.5B GPU readiness bridge (CPU)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-theory", action="store_true")
    parser.add_argument("--theory-samples", type=int, default=5)
    parser.add_argument("--strict", action="store_true", help="Exit 1 if cpu_bridge_ok is false")
    args = parser.parse_args()

    theory_ok = None if args.skip_theory else _run_theory_verify(args.theory_samples)
    report = build_scale_05b_gpu_report(theory_ok=theory_ok)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("0.5B GPU readiness bridge")
        print(f"  cpu_bridge_ok: {report['cpu_bridge_ok']}")
        print(f"  theory_ok: {report['theory_ok']}")
        print(f"  gpu_eval_ready: {report['gpu_eval_ready']}")
        for step in report["pipeline_steps"]:
            mark = "✓" if step["ready_on_host"] else "✗"
            print(f"  {mark} {step['id']}: {step['title']}")

    if args.strict and not report["cpu_bridge_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
