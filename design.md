# Chain of Reward (CoR) Design Document

## 1. Overview

CoR is a RL framework distributing reward signals along reasoning chains with **endogenous self-evaluation**.

## 2. Reward Design

### Total Reward
R(c) = R_ext(c) + λ × R_int(c)

### Intrinsic Reward (5 dimensions + self-rating)
R_int(c) = Σ w_d × r_d + w_self × r_self

| Dimension | Weight |
|-----------|--------|
| Consistency | 0.2 |
| Completeness | 0.2 |
| Accuracy | 0.2 |
| Clarity | 0.2 |
| Format | 0.2 |
| Self-rating | 0.2 |

### Self-Rating Calibration
r_self = (1/D) × Σ cal_d(self_rating_d/10, actual_d)
cal_d(u, v) = 1 - |u - v|

## 3. Implementation

s1-cor/train/rewards/ contains:
- calculator.py: RewardCalculator
- self_rating.py: SelfRatingExtractor
- intrinsic.py: IntrinsicRewardCalculator

## 4. Training

cd s1-cor/train && bash grpo.sh

Parameters: N=8, λ=1.0, β=0.01, δ=0.2

## 5. Results

| Model | AIME24 | MATH500 | GPQA |
|-------|--------|---------|------|
| w/o CoR | 50.0 | 92.6 | 56.6 |
| CoR-32B | 56.7 | 93.0 | 59.6 |

See theory.md for mathematical foundations.
