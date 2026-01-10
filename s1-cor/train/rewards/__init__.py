"""
Chain of Reward (CoR) - Reward Calculation Module

Implements multi-dimensional reward system for CoR training.
Supports CoR-GRPO dual coupling and self-reflection.
Based on DESIGN.md and THEORY.md specifications.
"""

from .calculator import RewardCalculator, RewardConfig, RewardOutput
from .self_rating import SelfRatingExtractor, SelfRatingEvaluator, SelfRating
from .intrinsic import (
    IntrinsicRewardCalculator,
    ConsistencyReward,
    CompletenessReward,
    ClarityReward,
    FormatReward,
    AccuracyReward,
    # NEW: Self-reflection rewards
    ReflectionReward,
    ImprovementRewardCalculator,
    ConvergenceRewardCalculator,
)
from .training_logger import CoRTrainingLogger, get_logger, log_cor_reward

__all__ = [
    # Core
    "RewardCalculator",
    "RewardConfig",
    "RewardOutput",
    # Self-rating
    "SelfRatingExtractor", 
    "SelfRatingEvaluator",
    "SelfRating",
    # Intrinsic dimensions
    "IntrinsicRewardCalculator",
    "ConsistencyReward",
    "CompletenessReward",
    "ClarityReward",
    "FormatReward",
    "AccuracyReward",
    # Self-reflection
    "ReflectionReward",
    "ImprovementRewardCalculator",
    "ConvergenceRewardCalculator",
    # Logging
    "CoRTrainingLogger",
    "get_logger",
    "log_cor_reward",
]
