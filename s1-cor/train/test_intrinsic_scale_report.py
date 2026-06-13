"""Tests for intrinsic scale report."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_intrinsic_scale_report_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_intrinsic_scale_report.py",
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
    assert data["report"] == "intrinsic_scale"
    assert data["samples"] == 3
    assert "suggested_lambda_intrinsic" in data
    assert data["mean_total"] >= 0.0
