"""Tests for GRPO reward smoke script."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_grpo_reward_smoke_json():
    proc = subprocess.run(
        [sys.executable, "scripts/run_grpo_reward_smoke.py", "--json", "--samples", "3"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "grpo_reward_smoke"
    assert data["samples_evaluated"] >= 1
    assert data["use_math_grader_wired"] is True
