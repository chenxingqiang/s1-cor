#!/usr/bin/env python3
"""
Meta Loop Layer 1 — Perceive (感知).

Emits a JSON snapshot for backlog / strategy. Does NOT run Implement or merge PRs.
See docs/LOOPS.md (meta loop vs product loops).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MATRIX = REPO / "docs" / "theory_code_matrix.yaml"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from loop_matrix import matrix_gaps, matrix_tier_counts, parse_matrix_components  # noqa: E402


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
    return matrix_tier_counts(parse_matrix_components(MATRIX))


def _parse_json_stdout(stdout: str) -> Dict[str, Any]:
    stdout = stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        if start < 0:
            return {}
        return json.loads(stdout[start:])


def _product_loop_snapshots(include: bool = True) -> Dict[str, Any]:
    """Layer-1 summaries from product-loop report scripts (small sample counts)."""
    if not include:
        return {"skipped": True}

    specs = [
        ("grpo_reward_smoke", "scripts/run_grpo_reward_smoke.py", ["--json", "--samples", "3"]),
        ("r_ext_alignment", "scripts/run_r_ext_alignment_report.py", ["--json", "--samples", "5"]),
        ("calibration_proxy", "scripts/run_calibration_report.py", ["--json", "--samples", "5"]),
        (
            "ablation_sweep_mini",
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
        ),
        (
            "intrinsic_dim_ablation_mini",
            "scripts/run_intrinsic_dim_ablation.py",
            ["--json", "--samples", "3", "--presets", "uniform,drop"],
        ),
        (
            "calibration_bonus_ablation_mini",
            "scripts/run_calibration_bonus_ablation.py",
            ["--json", "--samples", "3", "--alpha-values", "0.0,0.2"],
        ),
        (
            "eval_openai_grader_audit",
            "scripts/run_eval_openai_grader_report.py",
            ["--json"],
        ),
    ]
    out: Dict[str, Any] = {}
    for key, script, args in specs:
        proc = subprocess.run(
            [sys.executable, script, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            out[key] = {"ok": False, "exit_code": proc.returncode}
            continue
        report = _parse_json_stdout(proc.stdout)
        out[key] = {
            "ok": True,
            "report": report.get("report", key),
            "summary": _summarize_product_report(key, report),
        }
    return out


def _summarize_product_report(key: str, report: Dict[str, Any]) -> Dict[str, Any]:
    if key == "grpo_reward_smoke":
        return {
            "mean_reward_delta": report.get("mean_reward_delta_math_minus_string"),
            "disagreements": report.get("reward_disagreement_count"),
        }
    if key == "r_ext_alignment":
        return {
            "agreement_rate": report.get("agreement_rate"),
            "disagreement_count": report.get("disagreement_count"),
            "math_fixes_string": report.get("math_fixes_string"),
            "recommended_training_grader": report.get("recommended_training_grader"),
        }
    if key == "calibration_proxy":
        return {
            "ece_proxy": report.get("ece_proxy"),
            "rated_samples": report.get("samples_with_self_rating"),
        }
    if key == "ablation_sweep_mini":
        sweep = report.get("sweep") or []
        return {
            "configs": len(sweep),
            "best_mean_total": max((r.get("mean_total", 0) for r in sweep), default=0),
        }
    if key == "intrinsic_dim_ablation_mini":
        presets = report.get("presets") or []
        return {
            "presets": len(presets),
            "uniform_mean_intrinsic": next(
                (p.get("mean_intrinsic") for p in presets if p.get("preset") == "uniform_w0.2"),
                None,
            ),
        }
    if key == "calibration_bonus_ablation_mini":
        sweep = report.get("sweep") or []
        best = report.get("best_by_ece") or {}
        return {
            "alphas": len(sweep),
            "best_alpha": best.get("calibration_bonus"),
            "best_ece": best.get("ece_proxy"),
        }
    if key == "eval_openai_grader_audit":
        smoke = report.get("regex_extraction_smoke") or {}
        return {
            "regex_smoke_ok": smoke.get("ok"),
            "openai_api_key_set": report.get("openai_api_key_set"),
            "ready_for_openai_eval": report.get("ready_for_openai_eval"),
        }
    return {}


def build_snapshot(
    pytest_quick: bool = True,
    run_pytest: bool = True,
    include_product: bool = True,
) -> Dict[str, Any]:
    tiers = _matrix_tier_counts()
    gaps = matrix_gaps(parse_matrix_components(MATRIX))
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

    product_snapshots = _product_loop_snapshots(include=include_product)
    for key, snap in product_snapshots.items():
        if isinstance(snap, dict) and snap.get("ok") is False:
            backlog.append(f"product loop snapshot failed: {key}")

    return {
        "layer": "perceive",
        "meta_loop": "AGENTS.md five-layer; see docs/LOOPS.md",
        "product_loops": {
            "reflection_parsing": "train/reflection_parsing.py",
            "reward_formula": "train/rewards/calculator.py",
            "grpo": "train/grpo.py",
            "verify_entry": "make loop-product-verify",
        },
        "product_loop_snapshots": product_snapshots,
        "matrix_tiers": tiers,
        "matrix_gaps": [{"id": g["id"], "tier": g.get("tier"), "verify": g.get("verify")} for g in gaps],
        "pytest_train": {
            "exit_code": pytest_result["exit_code"],
            "ok": pytest_result["exit_code"] == 0 if run_pytest else None,
            "skipped": not run_pytest,
        },
        "eval_readiness": readiness_json,
        "backlog_hints": backlog,
        "next_layer": "strategy — make loop-strategy; pick ONE ranked gap; loop_verify after implement",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Meta loop layer 1: perceive")
    parser.add_argument("--json", action="store_true", help="Print JSON snapshot")
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Matrix + readiness only (faster; use loop_verify for pytest)",
    )
    parser.add_argument(
        "--skip-product",
        action="store_true",
        help="Skip product-loop report snapshots (faster perceive)",
    )
    args = parser.parse_args()

    snap = build_snapshot(
        run_pytest=not args.skip_pytest,
        include_product=not args.skip_product,
    )

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
