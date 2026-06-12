"""Tests for five-dim intrinsic ablation script."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_intrinsic_dim_ablation_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_intrinsic_dim_ablation.py",
            "--json",
            "--samples",
            "3",
            "--presets",
            "uniform,emphasize",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "intrinsic_dim_ablation"
    assert len(data["presets"]) == 6
    uniform = next(p for p in data["presets"] if p["preset"] == "uniform_w0.2")
    assert uniform["mean_intrinsic"] >= 0.0
    assert "consistency" in data["emphasis_delta_vs_uniform"]
