"""CPU audit: 0.5B scale CPU theory → GPU training/eval bridge."""

from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_repro_common import (  # noqa: E402
    COMMANDS,
    SCALE_05B_CHECKPOINTS,
    SCALE_05B_PIPELINE,
)
from check_eval_readiness import build_report as build_readiness  # noqa: E402


def _path_ready(rel: str) -> Dict[str, Any]:
    path = REPO_ROOT / rel
    is_dir = path.is_dir()
    return {
        "path": rel,
        "exists": is_dir or path.is_file(),
        "has_config": (path / "config.json").is_file() if is_dir else False,
    }


def _glob_any(pattern: str) -> bool:
    return bool(glob.glob(str(REPO_ROOT / pattern)))


def _commands_has_05b_eval() -> bool:
    if not COMMANDS.is_file():
        return False
    text = COMMANDS.read_text(encoding="utf-8")
    return "cor-0.5B" in text and "0.5B" in text


def _pipeline_step_rows(readiness: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = readiness.get("checks") or {}
    rows: List[Dict[str, Any]] = []

    for spec in SCALE_05B_PIPELINE:
        sid = spec["id"]
        artifact = spec.get("artifact_glob", "")

        if sid == "sft_05b":
            ok = _glob_any(artifact) or _path_ready("train/colab_minimal.sh")["exists"]
            blocker = None if _glob_any(artifact) else "run colab_minimal.sh sft on GPU"
        elif sid == "grpo_05b":
            ok = _glob_any(artifact)
            blocker = None if ok else "run grpo_05b.sh after SFT"
        elif sid == "readiness":
            ok = readiness.get("ready_for_benchmark_eval", False)
            blocker = None if ok else "; ".join(readiness.get("blockers", [])[:2])
        elif sid == "eval_05b":
            ok = (
                checks.get("cuda_available")
                and checks.get("vllm_installed")
                and (_glob_any("ckpts/cor-0.5B*") or _glob_any("ckpts/cor-0.5B"))
            )
            blocker = None if ok else "needs CUDA + vllm + cor-0.5B ckpt"
        elif sid == "compare":
            ok = (REPO_ROOT / "scripts" / "compare_eval_to_paper.py").is_file()
            blocker = None if ok else "missing compare_eval_to_paper.py"
        else:
            ok = False
            blocker = "unknown step"

        rows.append(
            {
                **spec,
                "ready_on_host": ok,
                "blocker": blocker,
            }
        )
    return rows


def evaluate_cpu_bridge_checks() -> Dict[str, bool]:
    train_dir = REPO_ROOT / "train"
    return {
        "grpo_05b_script": (train_dir / "grpo_05b.sh").is_file(),
        "colab_minimal_script": (train_dir / "colab_minimal.sh").is_file(),
        "sft_small_05b": (train_dir / "sft_small.py").is_file(),
        "commands_sh_05b_line": _commands_has_05b_eval(),
        "model_05b_doc": (REPO_ROOT.parent / "docs" / "MODEL_05B_TEST.md").is_file(),
        "sympy_available": importlib.util.find_spec("sympy") is not None,
    }


def build_scale_05b_gpu_report(
    theory_ok: bool | None = None,
) -> Dict[str, Any]:
    readiness = build_readiness()
    cpu_checks = evaluate_cpu_bridge_checks()
    ckpts = [_path_ready(rel) for rel in SCALE_05B_CHECKPOINTS]
    any_05b_ckpt = any(c["exists"] and c["has_config"] for c in ckpts) or _glob_any(
        "ckpts/cor-0.5B*"
    )

    cpu_bridge_ok = all(cpu_checks.values()) and theory_ok is not False
    gpu_blockers = list(readiness.get("blockers") or [])
    if not any_05b_ckpt:
        gpu_blockers.append("no 0.5B GRPO checkpoint under ckpts/cor-0.5B*")

    pipeline = _pipeline_step_rows(readiness)

    return {
        "layer": "verify",
        "report": "05b_gpu_readiness",
        "matrix_id": "scale_05b_validation",
        "cpu_bridge_ok": cpu_bridge_ok,
        "cpu_bridge_checks": cpu_checks,
        "theory_ok": theory_ok,
        "gpu_eval_ready": readiness.get("ready_for_benchmark_eval", False) and any_05b_ckpt,
        "any_05b_checkpoint": any_05b_ckpt,
        "checkpoints_05b": ckpts,
        "readiness": {
            "checks": readiness.get("checks"),
            "blockers": readiness.get("blockers"),
        },
        "gpu_blockers": gpu_blockers,
        "pipeline_steps": pipeline,
        "recommended_env": {
            "WANDB_DISABLED": "true",
            "USE_MATH_GRADER": "1",
            "REF_MODEL": "ckpts/sft-0.5B-colab",
        },
        "eval_command_hint": (
            "eval/commands.sh cor-0.5B line — set pretrained= to GRPO output_dir"
        ),
        "notes": [
            "cpu_bridge_ok = scripts/docs/theory gate on CPU; not benchmark scores.",
            "gpu_eval_ready requires CUDA + vllm + cor-0.5B ckpt + OPENAI_API_KEY.",
            "0.5B lm_eval is scale validation; paper AIME targets are 32B design.md goals.",
        ],
    }
