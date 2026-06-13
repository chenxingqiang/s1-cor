"""Tests for eval OpenAI grader audit."""

import json
import subprocess
import sys
from pathlib import Path

from eval_openai_grader_audit import (
    build_audit_report,
    parse_openai_tasks_from_commands,
    smoke_regex_answer_extraction,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parse_openai_tasks_from_commands():
    tasks = parse_openai_tasks_from_commands()
    assert "openai_math" in tasks
    assert "gpqa_diamond_openai" in tasks


def test_smoke_regex_answer_extraction():
    result = smoke_regex_answer_extraction()
    assert result["ok"] is True
    assert len(result["cases"]) >= 2


def test_build_audit_report_shape():
    report = build_audit_report()
    assert report["report"] == "eval_openai_grader_audit"
    assert "regex_extraction_smoke" in report
    assert report["regex_extraction_smoke"]["ok"] is True


def test_eval_openai_grader_report_script_json():
    proc = subprocess.run(
        [sys.executable, "scripts/run_eval_openai_grader_report.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "eval_openai_grader_audit"
    assert data["regex_extraction_smoke"]["ok"] is True
