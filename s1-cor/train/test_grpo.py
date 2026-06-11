"""Tests for GRPO reward function wiring."""

import pytest

from grpo import CoRTrainingConfig, create_reward_fn, prepare_dataset
from reflection_parsing import extract_reflection_rounds


class TestGrpoRewardFn:
    def setup_method(self):
        self.config = CoRTrainingConfig(enable_reflection=True)
        self.reward_fn = create_reward_fn(self.config, enable_logging=False)

    def test_single_round_returns_one_reward_per_completion(self):
        completions = [
            "Step 1: Think.\nTherefore, the answer is 42",
            "Step 1: Work it out.\nAnswer: 7",
        ]
        rewards = self.reward_fn(
            completions,
            ground_truths=["42", "7"],
        )
        assert len(rewards) == 2
        assert all(r > 0 for r in rewards)

    def test_self_rating_checkpoints_trigger_reflection_reward(self):
        completion = (
            "Step 1: weak guess.\n"
            "[Self-Rating: Consistency=3/10, Accuracy=2/10]\n"
            "Step 2: improved work.\n"
            "[Self-Rating: Consistency=8/10, Accuracy=8/10]\n"
            "Therefore, the answer is 42"
        )
        rewards = self.reward_fn([completion], ground_truths=["42"])
        assert len(rewards) == 1
        assert rewards[0] > 0

    def test_multi_round_does_not_double_count_rewards(self):
        completion = (
            "[Round 1]\n"
            "Step 1: initial guess.\n"
            "[Round 2]\n"
            "Step 1: revised reasoning.\n"
            "Therefore, the answer is 42"
        )
        assert len(extract_reflection_rounds(completion)) == 2

        rewards = self.reward_fn(
            [completion],
            ground_truths=["42"],
        )
        assert len(rewards) == 1
        assert rewards[0] > 0

    def test_no_ground_truth_uses_intrinsic_reward(self):
        rewards = self.reward_fn(
            ["Step 1: Think carefully.\nTherefore, x = 1."],
            ground_truths=[None],
        )
        assert len(rewards) == 1
        assert 0 <= rewards[0] <= 2.0

    def test_reference_answer_column_from_grpo_dataset(self):
        completion = "Step 1: Think.\nTherefore, the answer is 42"
        rewards = self.reward_fn(
            [completion],
            reference_answer=["42"],
        )
        assert len(rewards) == 1
        assert rewards[0] >= 1.0

    def test_prepare_dataset_loads_local_cor_data(self):
        dataset = prepare_dataset("local_data/s1K_cor_deepseek")
        assert len(dataset) > 0
        assert "prompt" in dataset.column_names
        assert "reference_answer" in dataset.column_names
        assert dataset[0]["reference_answer"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
