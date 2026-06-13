"""Tests for R_ext alignment report script."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r_ext_alignment_report_json():
    proc = subprocess.run(
        [sys.executable, "scripts/run_r_ext_alignment_report.py", "--json", "--samples", "3"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "r_ext_alignment"
    assert "agreement_rate" in data
    assert data["samples"] == 3
    assert "recommended_training_grader" in data
    assert "math_fixes_string" in data
