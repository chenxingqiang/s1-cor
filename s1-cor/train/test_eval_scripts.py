"""Smoke tests for eval readiness and reflection-K ablation scripts."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_check_eval_readiness_json():
    script = ROOT / "scripts" / "check_eval_readiness.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    data = json.loads(proc.stdout)
    assert "ready_for_benchmark_eval" in data
    assert "blockers" in data
    assert "paper_targets_design_md" in data
    assert data["paper_targets_design_md"]["AIME24"] == 56.7
    assert isinstance(data["ready_for_benchmark_eval"], bool)
    assert isinstance(data["blockers"], list)


def test_reflection_k_ablation_json():
    script = ROOT / "scripts" / "run_reflection_k_ablation.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--samples",
            "3",
            "--k-values",
            "1,2",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert len(data["reflection_k_sweep"]) == 2
    assert len(data["design_md_stage_presets"]) == 3
    stages = {r["stage"] for r in data["design_md_stage_presets"]}
    assert stages == {"sft_baseline", "cor_self_rating", "cor_reflection"}
    for row in data["design_md_stage_presets"]:
        assert row.get("mean_total", 0) > 0, f"{row['stage']} should have positive mean_total"


def test_loop_perceive_json():
    script = ROOT / "scripts" / "loop_perceive.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json", "--skip-pytest"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    data = json.loads(proc.stdout)
    assert data["layer"] == "perceive"
    assert "matrix_tiers" in data
    assert "product_loops" in data
    assert data["pytest_train"]["skipped"] is True
