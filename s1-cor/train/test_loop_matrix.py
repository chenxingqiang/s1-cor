"""Tests for loop_matrix.yaml parsing."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from loop_matrix import matrix_gaps, parse_matrix_components, rank_strategy_candidates


def test_parse_matrix_finds_partial_and_heuristic():
    comps = parse_matrix_components()
    ids = {c["id"] for c in comps}
    assert "external_reward" in ids
    assert "five_dim_intrinsic" in ids
    gaps = matrix_gaps(comps)
    tiers = {g["tier"] for g in gaps}
    assert "partial" in tiers
    assert "heuristic" in tiers


def test_rank_partial_before_heuristic_on_cpu():
    gaps = matrix_gaps(parse_matrix_components())
    ranked = rank_strategy_candidates(gaps, cuda_available=False, pytest_ok=True)
    partial_ranks = [
        r["priority_rank"]
        for r in ranked
        if r["tier"] == "partial" and r.get("cpu_actionable")
    ]
    heuristic_rank = next(
        r["priority_rank"] for r in ranked if r["id"] == "five_dim_intrinsic"
    )
    assert partial_ranks
    assert min(partial_ranks) < heuristic_rank


def test_rank_heuristic_before_deferred_on_cpu():
    gaps = matrix_gaps(parse_matrix_components())
    ranked = rank_strategy_candidates(gaps, cuda_available=False, pytest_ok=True)
    assert ranked[0]["tier"] in ("partial", "heuristic")
    heuristic_rank = next(
        r["priority_rank"] for r in ranked if r["id"] == "five_dim_intrinsic"
    )
    deferred_rank = next(
        r["priority_rank"] for r in ranked if r["id"] == "token_level_reward_chain"
    )
    assert heuristic_rank < deferred_rank
