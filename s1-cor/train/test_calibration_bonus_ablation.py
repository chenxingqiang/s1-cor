"""Tests for calibration bonus ablation script."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_bonus_ablation_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_calibration_bonus_ablation.py",
            "--json",
            "--samples",
            "5",
            "--alpha-values",
            "0.0,0.2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "calibration_bonus_ablation"
    assert len(data["sweep"]) == 2
    assert data["best_by_ece"] is not None
