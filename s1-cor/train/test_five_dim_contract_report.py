"""Tests for five_dim_contract audit and report script."""

import json
import subprocess
import sys
from pathlib import Path

from five_dim_contract_audit import build_five_dim_contract_report, evaluate_five_dim_contract

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_five_dim_contract_passes_minimal():
    correlation = {
        "samples_with_self_rating": 3,
        "pooled_pearson_r": 0.15,
    }
    ablation = {
        "most_sensitive_dimension": "accuracy",
        "drop_delta_vs_uniform": {"accuracy": -0.05},
        "emphasis_delta_vs_uniform": {"format": 0.02},
    }
    checks, ok = evaluate_five_dim_contract(correlation, ablation)
    assert ok is True
    assert checks["five_dimensions_defined"]["ok"] is True
    assert checks["nonzero_ablation_delta"]["ok"] is True


def test_five_dim_contract_report_script_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_five_dim_contract_report.py",
            "--json",
            "--strict",
            "--samples",
            "5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["report"] == "five_dim_contract"
    assert data["contract_ok"] is True
    assert data["matrix_id"] == "five_dim_intrinsic"
    for name, chk in data["contract_checks"].items():
        assert chk["ok"], f"{name}: {chk['detail']}"


def test_build_five_dim_contract_report_structure():
    report = build_five_dim_contract_report(
        {"samples_with_self_rating": 1, "pooled_pearson_r": 0.1},
        {
            "most_sensitive_dimension": "clarity",
            "drop_delta_vs_uniform": {"clarity": -0.01},
            "emphasis_delta_vs_uniform": {},
        },
    )
    assert report["matrix_tier"] == "partial"
    assert "honest_claim" in report
