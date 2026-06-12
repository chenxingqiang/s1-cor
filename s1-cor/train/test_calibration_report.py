"""Tests for calibration proxy report script."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_report_json():
    proc = subprocess.run(
        [sys.executable, "scripts/run_calibration_report.py", "--json", "--samples", "5"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "calibration_proxy"
    assert "ece_proxy" in data
