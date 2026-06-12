"""Tests for compare_eval_to_paper.py."""

import json
import pytest
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lm_eval_sample_results.json"


def test_compare_fixture_passes_within_tolerance():
    script = ROOT / "scripts" / "compare_eval_to_paper.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--results", str(FIXTURE), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert data["all_pass"] is True
    assert data["observed"]["AIME24"] == pytest.approx(56.7, abs=0.1)
    assert data["observed"]["MATH500"] == pytest.approx(93.0, abs=0.1)
