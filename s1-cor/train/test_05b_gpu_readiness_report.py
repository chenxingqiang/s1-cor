"""Tests for 0.5B GPU readiness bridge report."""

import json
import subprocess
import sys
from pathlib import Path

from scale_05b_gpu_audit import build_scale_05b_gpu_report, evaluate_cpu_bridge_checks

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_bridge_checks_pass_on_repo():
    checks = evaluate_cpu_bridge_checks()
    assert checks["grpo_05b_script"] is True
    assert checks["commands_sh_05b_line"] is True
    assert checks["model_05b_doc"] is True


def test_build_scale_05b_gpu_report_structure():
    report = build_scale_05b_gpu_report(theory_ok=True)
    assert report["report"] == "05b_gpu_readiness"
    assert report["cpu_bridge_ok"] is True
    assert len(report["pipeline_steps"]) == 5
    assert report["matrix_id"] == "scale_05b_validation"


def test_05b_gpu_readiness_script_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_05b_gpu_readiness_report.py",
            "--json",
            "--strict",
            "--theory-samples",
            "3",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["cpu_bridge_ok"] is True
    assert data["theory_ok"] is True
