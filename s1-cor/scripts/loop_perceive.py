#!/usr/bin/env python3
"""
Meta Loop Layer 1 — Perceive (感知).

Emits a JSON snapshot for backlog / strategy. Does NOT run Implement or merge PRs.
See docs/LOOPS.md (meta loop vs product loops).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MATRIX = REPO / "docs" / "theory_code_matrix.yaml"


def _run(cmd: List[str], cwd: Path | None = None) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "cmd": " ".join(cmd),
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
    }


def _matrix_tier_counts() -> Dict[str, int]:
    if not MATRIX.is_file():
        return {}
    text = MATRIX.read_text(encoding="utf-8")
    counts: Dict[str, int] = {}
    for tier in re.findall(r"^\s+tier:\s*(\S+)", text, re.MULTILINE):
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def build_snapshot(pytest_quick: bool = True, run_pytest: bool = True) -> Dict[str, Any]:
    tiers = _matrix_tier_counts()
    backlog: List[str] = []

    if tiers.get("deferred", 0):
        backlog.append(f"matrix deferred items: {tiers['deferred']}")
    if tiers.get("partial", 0):
        backlog.append(f"matrix partial items: {tiers['partial']}")
    if tiers.get("heuristic", 0):
        backlog.append(f"matrix heuristic items: {tiers['heuristic']}")

    if run_pytest:
        pytest_cmd = [sys.executable, "-m", "pytest", "train/", "-q", "--tb=no"]
        if pytest_quick:
            pytest_cmd.append("-x")
        pytest_result = _run(pytest_cmd)
        if pytest_result["exit_code"] != 0:
            backlog.insert(0, "pytest train/ failing")
    else:
        pytest_result = {"exit_code": None, "skipped": True}

    readiness_json: Dict[str, Any] = {}
    full = subprocess.run(
        [sys.executable, "scripts/check_eval_readiness.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    try:
        readiness_json = json.loads(full.stdout)
        if not readiness_json.get("ready_for_benchmark_eval"):
            backlog.append(
                "benchmark eval not ready: "
                + "; ".join(readiness_json.get("blockers", [])[:3])
            )
    except json.JSONDecodeError:
        backlog.append("check_eval_readiness JSON parse failed")

    return {
        "layer": "perceive",
        "meta_loop": "AGENTS.md five-layer; see docs/LOOPS.md",
        "product_loops": {
            "reflection_parsing": "train/reflection_parsing.py",
            "reward_formula": "train/rewards/calculator.py",
            "grpo": "train/grpo.py",
        },
        "matrix_tiers": tiers,
        "pytest_train": {
            "exit_code": pytest_result["exit_code"],
            "ok": pytest_result["exit_code"] == 0 if run_pytest else None,
            "skipped": not run_pytest,
        },
        "eval_readiness": readiness_json,
        "backlog_hints": backlog,
        "next_layer": "strategy — pick ONE item from backlog_hints; run loop_verify after implement",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Meta loop layer 1: perceive")
    parser.add_argument("--json", action="store_true", help="Print JSON snapshot")
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Matrix + readiness only (faster; use loop_verify for pytest)",
    )
    args = parser.parse_args()

    snap = build_snapshot(run_pytest=not args.skip_pytest)

    if args.json:
        print(json.dumps(snap, indent=2))
    else:
        print("Meta Loop — Perceive")
        print(f"  matrix tiers: {snap['matrix_tiers']}")
        if snap["pytest_train"]["skipped"]:
            print("  pytest train/: skipped (use loop_verify)")
        else:
            print(f"  pytest train/: {'OK' if snap['pytest_train']['ok'] else 'FAIL'}")
        if snap["backlog_hints"]:
            print("  backlog:")
            for h in snap["backlog_hints"]:
                print(f"    - {h}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
