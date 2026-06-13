"""Tests for benchmark reproduction audit."""

import json
import subprocess
import sys
from pathlib import Path

from benchmark_reproduction_audit import build_audit_report

ROOT = Path(__file__).resolve().parents[1]


def test_build_audit_report_fixture_compare_ok():
    report = build_audit_report()
    assert report["report"] == "benchmark_reproduction_audit"
    assert report["fixture_compare_ok"] is True
    assert report["cpu_audit_ok"] is True
    assert len(report["reproduction_steps"]) == 4


def test_benchmark_reproduction_report_script_json():
    proc = subprocess.run(
        [sys.executable, "scripts/run_benchmark_reproduction_report.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["cpu_audit_ok"] is True
    assert data["paper_targets"]["MATH500"] == 93.0
