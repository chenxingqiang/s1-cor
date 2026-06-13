"""Tests for publication readiness audit."""

import json
import subprocess
import sys
from pathlib import Path

from publication_readiness_audit import build_publication_readiness_report

ROOT = Path(__file__).resolve().parents[1]


def test_publication_readiness_audit_after_r18_docs():
    report = build_publication_readiness_report()
    assert report["report"] == "publication_readiness_audit"
    # R18 lands README + design fixes + fixture label — audit should pass
    assert report["audit_ok"] is True, report.get("issues")


def test_publication_readiness_script_json():
    proc = subprocess.run(
        [sys.executable, "scripts/run_publication_readiness_report.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["audit_ok"] is True
