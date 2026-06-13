"""CPU audit helpers for eval-only OpenAI grading (MATH500 / GPQA)."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_SH = REPO_ROOT / "eval" / "commands.sh"
HARNESS = REPO_ROOT / "eval" / "lm-evaluation-harness"

OPENAI_EVAL_TASKS = ("openai_math", "gpqa_diamond_openai")
ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*(.*)", re.DOTALL)


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def parse_openai_tasks_from_commands(commands_path: Path | None = None) -> List[str]:
    """Return unique lm_eval task names in commands.sh that use OpenAI grading."""
    path = commands_path or COMMANDS_SH
    if not path.is_file():
        return []

    found: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "OPENAI_API_KEY" not in line and "openai_math" not in line and "gpqa" not in line:
            continue
        for task in OPENAI_EVAL_TASKS:
            if task in line and task not in found:
                found.append(task)
    return found


def smoke_regex_answer_extraction() -> Dict[str, Any]:
    """CPU-only: regex path used before OpenAI fallback in lm-eval utils."""
    cases = [
        ("Answer: 42", "42"),
        ("thinking...\nAnswer: \\boxed{7}", "\\boxed{7}"),
        ("Answer: Here is the solution:\n\n10", "Here is the solution:\n\n10"),
    ]
    rows = []
    ok = True
    for text, expected_tail in cases:
        matches = ANSWER_PATTERN.findall(text)
        got = matches[-1].strip() if matches else ""
        passed = got == expected_tail or expected_tail in got
        rows.append({"input_excerpt": text[:40], "extracted": got[:60], "ok": passed})
        ok = ok and passed
    return {"ok": ok, "cases": rows}


def build_audit_report() -> Dict[str, Any]:
    tasks = parse_openai_tasks_from_commands()
    regex_smoke = smoke_regex_answer_extraction()

    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False

    key_set = bool(os.environ.get("OPENAI_API_KEY"))
    harness_ok = HARNESS.is_dir()
    lm_eval_ok = _package_available("lm_eval")

    blockers: List[str] = []
    if not key_set:
        blockers.append("OPENAI_API_KEY unset (required for openai_math / gpqa_diamond_openai)")
    if not cuda:
        blockers.append("CUDA not available (vLLM eval)")
    if not harness_ok:
        blockers.append("lm-evaluation-harness missing")
    if not lm_eval_ok:
        blockers.append("lm_eval package not installed")

    return {
        "layer": "verify",
        "report": "eval_openai_grader_audit",
        "openai_eval_tasks": tasks,
        "commands_sh": str(COMMANDS_SH.relative_to(REPO_ROOT)),
        "harness_present": harness_ok,
        "lm_eval_installed": lm_eval_ok,
        "openai_api_key_set": key_set,
        "cuda_available": cuda,
        "regex_extraction_smoke": regex_smoke,
        "cpu_math_grader_available": _package_available("sympy"),
        "ready_for_openai_eval": key_set and cuda and harness_ok and lm_eval_ok,
        "blockers": blockers,
        "notes": [
            "Eval-only; training R_ext uses answer_grading.py (see docs/TRAIN_EVAL_GRADING.md).",
            "Full MATH500/GPQA scoring calls OpenAI when regex/boxed extraction is insufficient.",
            "PROCESSOR=gpt-4o-mini required in commands.sh for openai_math.",
        ],
    }
