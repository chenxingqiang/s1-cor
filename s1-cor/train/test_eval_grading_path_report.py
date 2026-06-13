"""Tests for eval grading path audit."""

import json
import subprocess
import sys
from pathlib import Path

from eval_grading_path_audit import (
    build_path_alignment_report,
    extract_lm_eval_pre_openai_answer,
)

ROOT = Path(__file__).resolve().parents[1]


def test_extract_lm_eval_pre_openai_boxed():
    text = "reasoning...\n\\boxed{42}"
    assert extract_lm_eval_pre_openai_answer(text) == "42"


def test_extract_lm_eval_pre_openai_answer_regex():
    text = "work\nAnswer: 7"
    assert extract_lm_eval_pre_openai_answer(text) == "7"


def test_build_path_alignment_report_agreement():
    rows = [
        {"attempt": "Answer: 42", "solution": "42"},
        {"attempt": "\\boxed{7}", "solution": "7"},
    ]
    report = build_path_alignment_report(rows)
    assert report["samples"] == 2
    assert report["path_agreement_rate"] == 1.0


def test_eval_grading_path_report_script_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval_grading_path_report.py",
            "--json",
            "--samples",
            "3",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "eval_grading_path"
    assert "path_agreement_rate" in data
    assert data["samples"] == 3
