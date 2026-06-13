#!/usr/bin/env python3
"""
Product Loop Layer 4 — Verify (产品循环验证).

CPU evidence for reward chain + GRPO wiring (not meta-loop pytest gate).
Runs existing single-purpose report scripts; not a full pipeline orchestrator.

See docs/LOOPS.md (product loops vs meta loop_verify).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

# (script relative to s1-cor/, extra args, display name)
PRODUCT_CHECKS: List[Tuple[str, List[str], str]] = [
    ("scripts/run_grpo_reward_smoke.py", ["--json", "--samples", "3"], "grpo_reward_smoke"),
    ("scripts/run_r_ext_alignment_report.py", ["--json", "--samples", "5"], "r_ext_alignment"),
    ("scripts/run_calibration_report.py", ["--json", "--samples", "5"], "calibration_proxy"),
    (
        "scripts/run_ablation_sweep.py",
        [
            "--json",
            "--samples",
            "3",
            "--lambda-values",
            "0,1",
            "--mu-values",
            "0,0.5",
            "--alpha-values",
            "1",
        ],
        "ablation_sweep_mini",
    ),
    (
        "scripts/run_intrinsic_dim_ablation.py",
        ["--json", "--samples", "3", "--presets", "uniform,drop"],
        "intrinsic_dim_ablation_mini",
    ),
    (
        "scripts/run_calibration_bonus_ablation.py",
        ["--json", "--samples", "3", "--alpha-values", "0.0,0.2"],
        "calibration_bonus_ablation_mini",
    ),
    (
        "scripts/run_eval_openai_grader_report.py",
        ["--json"],
        "eval_openai_grader_audit",
    ),
    (
        "scripts/run_intrinsic_scale_report.py",
        ["--json", "--samples", "3"],
        "intrinsic_scale_mini",
    ),
    (
        "scripts/run_benchmark_reproduction_report.py",
        ["--json"],
        "benchmark_reproduction_audit",
    ),
    (
        "scripts/run_eval_grading_path_report.py",
        ["--json", "--samples", "5"],
        "eval_grading_path_mini",
    ),
]


def _parse_json_stdout(stdout: str) -> Dict[str, Any]:
    stdout = stdout.strip()
    if not stdout:
        raise json.JSONDecodeError("empty stdout", stdout, 0)
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        if start < 0:
            raise
        return json.loads(stdout[start:])


def run_checks(quiet: bool = False) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    failed: List[str] = []

    def _log(msg: str) -> None:
        if not quiet:
            print(msg)

    for script, extra, name in PRODUCT_CHECKS:
        cmd = [sys.executable, script, *extra]
        _log(f"▶ {name}...")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        row: Dict[str, Any] = {"exit_code": proc.returncode, "cmd": " ".join(cmd)}

        if proc.returncode != 0:
            failed.append(name)
            row["error"] = (proc.stderr or proc.stdout)[-500:]
            results[name] = row
            _log(f"✗ {name} failed (exit {proc.returncode})")
            continue

        try:
            row["report"] = _parse_json_stdout(proc.stdout)
            results[name] = row
            _log(f"✓ {name}")
        except json.JSONDecodeError as exc:
            failed.append(name)
            row["error"] = f"JSON parse: {exc}"
            results[name] = row
            _log(f"✗ {name} invalid JSON")

    return {
        "layer": "verify",
        "loop": "product",
        "ok": len(failed) == 0,
        "failed": failed,
        "checks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Product loop layer 4: verify")
    parser.add_argument("--json", action="store_true", help="Print JSON summary only")
    args = parser.parse_args()

    if args.json:
        logging.disable(logging.CRITICAL)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    summary = run_checks(quiet=args.json)

    if args.json:
        print(json.dumps(summary, indent=2))
    elif summary["ok"]:
        print("\nloop_product_verify: OK (product loop layer 4)")
    else:
        print(f"\nloop_product_verify: FAILED ({', '.join(summary['failed'])})")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
