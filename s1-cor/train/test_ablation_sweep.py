"""Smoke test for scripts/run_ablation_sweep.py (CPU)."""

import json
import subprocess
import sys
from pathlib import Path


def test_ablation_sweep_json():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_ablation_sweep.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "deepseek",
            "--samples",
            "2",
            "--lambda-values",
            "0.0,1.0",
            "--mu-values",
            "0.0,0.5",
            "--alpha-values",
            "1.0",
            "--json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert data["samples"] == 2
    assert len(data["sweep"]) == 4
    assert all("mean_total" in r for r in data["sweep"])
