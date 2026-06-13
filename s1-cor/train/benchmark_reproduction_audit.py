"""CPU audit for benchmark reproduction chain (partial → documented GPU path)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_eval_to_paper import compare_scores, load_scores_from_path  # noqa: E402
from eval_repro_common import (  # noqa: E402
    FIXTURE_LM_EVAL,
    PAPER_TARGETS,
    REPRODUCTION_STEPS,
    SMOKE_RESULTS_DIR,
)
from check_eval_readiness import build_report as build_readiness  # noqa: E402


def _step_status(readiness: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = readiness.get("checks") or {}
    steps: List[Dict[str, Any]] = []
    for spec in REPRODUCTION_STEPS:
        sid = spec["id"]
        if sid == "train":
            ok = bool(checks.get("any_default_checkpoint"))
            blocker = None if ok else "no ckpts/cor-grpo (or cor-sft) with config.json"
        elif sid == "readiness":
            ok = readiness.get("ready_for_benchmark_eval", False)
            blocker = None if ok else "; ".join(readiness.get("blockers", [])[:2])
        elif sid == "eval":
            ok = (
                checks.get("cuda_available")
                and checks.get("vllm_installed")
                and checks.get("any_default_checkpoint")
            )
            blocker = None if ok else "needs CUDA + vllm + checkpoint"
        elif sid == "compare":
            ok = FIXTURE_LM_EVAL.is_file()
            blocker = None if ok else "missing compare fixture"
        else:
            ok = False
            blocker = "unknown step"

        steps.append(
            {
                **spec,
                "ready_on_host": ok,
                "blocker": blocker,
            }
        )
    return steps


def build_audit_report(tolerance: float = 5.0) -> Dict[str, Any]:
    readiness = build_readiness()
    steps = _step_status(readiness)

    fixture_scores = load_scores_from_path(FIXTURE_LM_EVAL) if FIXTURE_LM_EVAL.is_file() else {}
    fixture_comparison = compare_scores(fixture_scores, tolerance) if fixture_scores else []
    fixture_ok = bool(fixture_scores) and all(r["status"] == "pass" for r in fixture_comparison)

    smoke_scores: Dict[str, float] = {}
    smoke_comparison: List[Dict[str, Any]] = []
    if SMOKE_RESULTS_DIR.is_dir():
        smoke_scores = load_scores_from_path(SMOKE_RESULTS_DIR)
        if smoke_scores:
            smoke_comparison = compare_scores(smoke_scores, tolerance)

    checks = readiness.get("checks") or {}
    cpu_audit_ok = (
        fixture_ok
        and checks.get("lm_eval_harness_present")
        and checks.get("commands_sh_present")
    )

    return {
        "layer": "verify",
        "report": "benchmark_reproduction_audit",
        "paper_targets": PAPER_TARGETS,
        "readiness": {
            "ready_for_benchmark_eval": readiness.get("ready_for_benchmark_eval"),
            "blockers": readiness.get("blockers", []),
            "checks": checks,
        },
        "reproduction_steps": steps,
        "fixture_compare_ok": fixture_ok,
        "fixture_comparison": fixture_comparison,
        "smoke_results_dir": str(SMOKE_RESULTS_DIR.relative_to(SCRIPT_DIR.parent)),
        "smoke_results_present": bool(smoke_scores),
        "smoke_comparison": smoke_comparison or None,
        "cpu_audit_ok": cpu_audit_ok,
        "gpu_reproduction_ready": readiness.get("ready_for_benchmark_eval", False),
        "notes": [
            "cpu_audit_ok validates compare_eval_to_paper + harness paths on CPU.",
            "gpu_reproduction_ready requires CUDA + vllm + ckpt + OPENAI_API_KEY.",
            "Run make loop-eval-smoke to populate results/eval_smoke_dummy (non-paper scores).",
        ],
    }
