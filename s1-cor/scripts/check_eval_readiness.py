#!/usr/bin/env python3
"""
Check prerequisites for reproducing README / design.md benchmark numbers.

CPU-safe gate before running s1-cor/eval/commands.sh on a GPU host.
Does not run lm_eval or download checkpoints.

Usage:
    cd s1-cor
    python scripts/check_eval_readiness.py
    python scripts/check_eval_readiness.py --json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "eval" / "lm-evaluation-harness"
COMMANDS = REPO_ROOT / "eval" / "commands.sh"

# design.md §9 / README primary row (CoR-32B, 1K samples)
PAPER_TARGETS = {
    "AIME24": 56.7,
    "MATH500": 93.0,
    "GPQA": 59.6,
}

DEFAULT_CHECKPOINTS = [
    "ckpts/cor-grpo",
    "ckpts/cor-sft",
    "ckpts/cor-32B",
]


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _package_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _checkpoint_status() -> List[Dict[str, Any]]:
    rows = []
    for rel in DEFAULT_CHECKPOINTS:
        path = REPO_ROOT / rel
        rows.append(
            {
                "path": rel,
                "exists": path.is_dir(),
                "has_config": (path / "config.json").is_file() if path.is_dir() else False,
            }
        )
    return rows


def build_report() -> Dict[str, Any]:
    ckpts = _checkpoint_status()
    any_ckpt = any(c["exists"] and c["has_config"] for c in ckpts)

    checks = {
        "cuda_available": _has_cuda(),
        "vllm_installed": _package_available("vllm"),
        "lm_eval_installed": _package_available("lm_eval"),
        "lm_eval_harness_present": HARNESS.is_dir(),
        "commands_sh_present": COMMANDS.is_file(),
        "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "any_default_checkpoint": any_ckpt,
        "sympy_available": _package_available("sympy"),
    }

    blockers: List[str] = []
    if not checks["cuda_available"]:
        blockers.append("CUDA not available (required for vLLM eval)")
    if not checks["vllm_installed"]:
        blockers.append("vllm not installed")
    if not checks["lm_eval_installed"]:
        blockers.append("lm_eval not installed (pip install -e eval/lm-evaluation-harness[math,vllm])")
    if not checks["any_default_checkpoint"]:
        blockers.append("no default checkpoint under ckpts/ (train or download weights)")
    if not checks["openai_api_key_set"]:
        blockers.append("OPENAI_API_KEY unset (MATH/GPQA grading in commands.sh)")

    ready = len(blockers) == 0

    return {
        "ready_for_benchmark_eval": ready,
        "blockers": blockers,
        "checks": checks,
        "checkpoints": ckpts,
        "paper_targets_design_md": PAPER_TARGETS,
        "eval_entrypoint": "cd s1-cor/eval/lm-evaluation-harness && bash ../commands.sh",
        "repro_doc": "docs/EVAL_REPRODUCTION.md",
        "compare_script": "python scripts/compare_eval_to_paper.py --results-dir <lm_eval_output>",
        "notes": [
            "CPU VMs can run this script only; full reproduction needs GPU + ckpt + vLLM.",
            "Compare lm_eval JSON to paper_targets after training CoR-GRPO on 1K s1K-cor.",
            "GRPO training: USE_MATH_GRADER=1 enables eval-aligned R_ext (requires sympy).",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark eval readiness gate")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("CoR benchmark eval readiness")
        print("=" * 40)
        for key, ok in report["checks"].items():
            mark = "✅" if ok else "❌"
            print(f"  {mark} {key}")
        print()
        if report["blockers"]:
            print("Blockers:")
            for b in report["blockers"]:
                print(f"  - {b}")
        else:
            print("✅ Ready to run eval/commands.sh on this host.")
        print(f"\nPaper targets (design.md §9): {report['paper_targets_design_md']}")

    return 0 if report["ready_for_benchmark_eval"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
