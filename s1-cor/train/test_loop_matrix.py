"""Tests for loop_matrix.yaml parsing."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from loop_matrix import matrix_gaps, parse_matrix_components, rank_strategy_candidates


def test_parse_matrix_finds_partial_gaps():
    comps = parse_matrix_components()
    ids = {c["id"] for c in comps}
    assert "external_reward" in ids
    assert "five_dim_intrinsic" in ids
    five = next(c for c in comps if c["id"] == "five_dim_intrinsic")
    assert five["tier"] == "partial"
    gaps = matrix_gaps(comps)
    tiers = {g["tier"] for g in gaps}
    assert "partial" in tiers
    assert "deferred" in tiers


def test_rank_eval_openai_before_benchmark_on_cpu():
    """benchmark_reproduction is GPU-blocked on CPU; eval_openai_grader ranks higher."""
    gaps = matrix_gaps(parse_matrix_components())
    ranked = rank_strategy_candidates(gaps, cuda_available=False, pytest_ok=True)
    eval_rank = next(r["priority_rank"] for r in ranked if r["id"] == "eval_openai_grader")
    bench_rank = next(r["priority_rank"] for r in ranked if r["id"] == "benchmark_reproduction")
    assert eval_rank < bench_rank


def test_external_reward_tier_implemented():
    comps = parse_matrix_components()
    ext = next(c for c in comps if c["id"] == "external_reward")
    assert ext["tier"] == "implemented"
    assert "external_reward" not in {g["id"] for g in matrix_gaps(comps)}


def test_rank_partial_before_deferred_on_cpu():
    gaps = matrix_gaps(parse_matrix_components())
    ranked = rank_strategy_candidates(gaps, cuda_available=False, pytest_ok=True)
    partial_rank = next(
        r["priority_rank"] for r in ranked if r["id"] == "five_dim_intrinsic"
    )
    deferred_rank = next(
        r["priority_rank"] for r in ranked if r["id"] == "token_level_reward_chain"
    )
    assert partial_rank < deferred_rank
