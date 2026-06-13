"""Tests for self-rating vs intrinsic correlation report."""

import json
import subprocess
import sys
from pathlib import Path

from self_rating_intrinsic_correlation_audit import build_correlation_report

ROOT = Path(__file__).resolve().parents[1]


def _fake_chain(row):
    return [row.get("chain", "")] if row.get("chain") else []


def _fake_thinking(text):
    return text


def test_build_correlation_report_synthetic():
    rows = [
        {
            "chain": (
                "Step 1: work.\n"
                "[Self-Rating: Consistency=8/10, Completeness=6/10, "
                "Accuracy=7/10, Clarity=7/10]\n"
                "Therefore answer is 1."
            ),
        }
    ]
    report = build_correlation_report(
        rows,
        extract_chain_fn=_fake_chain,
        extract_thinking_fn=_fake_thinking,
    )
    assert report["samples_with_self_rating"] == 1
    assert report["per_dimension"]
    assert report["pooled_pearson_r"] is not None


def test_self_rating_intrinsic_correlation_script_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_self_rating_intrinsic_correlation_report.py",
            "--json",
            "--samples",
            "5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "self_rating_intrinsic_correlation"
    assert data["samples_with_self_rating"] >= 1
    assert "pooled_pearson_r" in data
