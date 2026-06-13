"""Tests for deferred claims audit."""

import json
import subprocess
import sys
from pathlib import Path

from deferred_claims_audit import build_deferred_claims_report

ROOT = Path(__file__).resolve().parents[1]


def test_build_deferred_claims_report_ok():
    report = build_deferred_claims_report()
    assert report["report"] == "deferred_claims_audit"
    assert report["deferred_count"] >= 2
    assert report["audit_ok"] is True
    ids = {e["id"] for e in report["entries"]}
    assert "token_level_reward_chain" in ids
    assert "dual_coupling_phi" in ids


def test_deferred_claims_report_script_json():
    proc = subprocess.run(
        [sys.executable, "scripts/run_deferred_claims_report.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["audit_ok"] is True
