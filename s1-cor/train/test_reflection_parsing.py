"""Tests for multi-round chain extraction."""

from data_utils import load_cor_dataset_from_disk
from reflection_parsing import (
    extract_chain_sequence_from_sample,
    extract_reflection_rounds,
    split_thinking_by_self_ratings,
)


class TestExplicitRounds:
    def test_two_round_markers(self):
        text = (
            "[Round 1]\n"
            "Step 1: guess.\n"
            "[Round 2]\n"
            "Step 1: better.\n"
        )
        chains = extract_reflection_rounds(text)
        assert len(chains) == 2


class TestSelfRatingSplits:
    def test_multiple_self_ratings_yield_cumulative_snapshots(self):
        thinking = (
            "Part A reasoning.\n"
            "[Self-Rating: Consistency=4/10, Accuracy=3/10]\n"
            "Part B reasoning.\n"
            "[Self-Rating: Consistency=8/10, Accuracy=7/10]\n"
        )
        chains = split_thinking_by_self_ratings(thinking)
        assert len(chains) == 2
        assert "Part A" in chains[0]
        assert "Part A" in chains[1]
        assert "Part B" in chains[1]
        assert len(chains[1]) > len(chains[0])

    def test_single_self_rating_returns_one_chain(self):
        thinking = "Only one block.\n[Self-Rating: Consistency=5/10]"
        assert len(split_thinking_by_self_ratings(thinking)) == 1


class TestDatasetRows:
    def test_deepseek_sample_has_multi_round_chain(self):
        ds = load_cor_dataset_from_disk("local_data/s1K_cor_deepseek")
        chains = extract_chain_sequence_from_sample(ds[0])
        assert len(chains) >= 2
