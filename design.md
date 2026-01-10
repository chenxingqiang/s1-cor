# Chain of Reward (CoR) Design Document

## 1. Overview

CoR is a RL framework with **dual coupling** between reward signals and policy optimization, enabling **self-reflection** and **iterative intelligence growth**.

### Core Innovation
- **Endogenous Self-Evaluation**: Model generates multi-dimensional self-ratings
- **CoR-GRPO Dual Coupling**: Bidirectional co-evolution of rewards and policy
- **Self-Reflection Loop**: Iterative improvement through reflection

## 2. Reward Design

### 2.1 Total Reward (Extended)
```
R(c) = R_ext(c) + λ·R_int(c) + μ·R_improve(c) + ν·R_converge(c)
```

### 2.2 Intrinsic Reward (5 dimensions + self-rating)
```
R_int(c) = Σ w_d × r_d + w_self × r_self
```

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Consistency | 0.2 | Logical coherence |
| Completeness | 0.2 | Step comprehensiveness |
| Accuracy | 0.2 | Factual correctness |
| Clarity | 0.2 | Reasoning clarity |
| Format | 0.2 | Structural correctness |
| Self-rating | 0.2 | Calibration quality |

### 2.3 Self-Rating Calibration
```
r_self = (1/D) × Σ cal_d(self_rating_d/10, actual_d)
cal_d(u, v) = 1 - |u - v|
```

### 2.4 Improvement Reward (NEW)
```
R_improve(c_k, c_{k+1}) = Q(c_{k+1}) - Q(c_k)
```

### 2.5 Convergence Reward (NEW)
```
R_converge = -|c_{k+1} - c_k|
```

## 3. Dual Coupling Theory

### 3.1 Structure
```
       CoR (Reward)  ←──→  GRPO (Policy)
             ↓                  ↓
    Self-rating quality    Policy improvement
             ↓                  ↓
       Better rewards  →  Better reasoning
```

### 3.2 Dynamics
- **Coupling 1**: CoR rewards drive GRPO policy updates
- **Coupling 2**: Better policy improves self-rating calibration
- **Result**: Synergistic co-evolution

## 4. Self-Reflection Framework

### 4.1 Reflection Operator
```
c_{k+1} = T(c_k, self_rating_k, x)
```

### 4.2 Convergence Guarantee
If T is contraction mapping (γ < 1), iteration converges to optimal c*.

### 4.3 Intelligence Growth
```
E[Q(c_K) - Q(c_0)] = μ · (1-γ^K)/(1-γ)
```
More reflection → Higher quality reasoning

## 5. Implementation

### 5.1 Core Modules
```
s1-cor/train/rewards/
├── __init__.py
├── calculator.py      # RewardCalculator (extended)
├── self_rating.py     # SelfRatingExtractor, Evaluator
├── intrinsic.py       # 5 dimension rewards + ReflectionReward
└── reflection.py      # NEW: Reflection loop manager
```

### 5.2 Key Classes

**RewardCalculator** (extended):
```python
class RewardCalculator:
    def calculate_total_reward(self, thinking_chain, answer, ground_truth):
        external = self.calculate_external_reward(answer, ground_truth)
        intrinsic, dim_scores = self.calculate_intrinsic_reward(thinking_chain)
        return external + λ * intrinsic
    
    def calculate_improvement_reward(self, c_old, c_new):
        return self.quality(c_new) - self.quality(c_old)
```

**ReflectionManager** (NEW):
```python
class ReflectionManager:
    def reflect_and_improve(self, chain, self_rating, x, max_rounds=3):
        for k in range(max_rounds):
            low_dims = self.identify_weak_dimensions(self_rating)
            reflection = self.generate_reflection(chain, low_dims)
            new_chain = self.generate_improved_chain(x, reflection)
            improvement = self.calculate_improvement(chain, new_chain)
            if improvement < threshold:
                break
            chain = new_chain
        return chain
```

## 6. Training Pipeline

### 6.1 SFT Phase
Train model to generate self-ratings and reflection patterns.

### 6.2 GRPO Phase (with Reflection)
```
for each input x:
    c_0 = generate(x)
    for k = 0 to K-1:
        self_rating_k = extract(c_k)
        c_{k+1} = reflect_and_improve(c_k, self_rating_k)
        R_improve_k = Q(c_{k+1}) - Q(c_k)
    
    R_total = R_ext + λ·R_int + μ·Σ R_improve_k
    update_policy(R_total)
```

### 6.3 Launch Commands
```bash
# SFT with reflection data
python train/sft_small.py --dataset hf --push_to_hub

# GRPO with dual coupling
bash train/grpo.sh
```

## 7. Data Format

### 7.1 Single-Round (Current)
```
<thinking>
...reasoning...
[Self-Rating: Consistency=7/10, Accuracy=8/10, ...]
</thinking>
<answer>...</answer>
```

### 7.2 Multi-Round Reflection (NEW)
```
[Round 1]
<thinking>...initial reasoning...</thinking>
[Self-Rating: Consistency=4/10, Accuracy=3/10, ...]

[Reflection]
My accuracy is low (3/10). The error is in step 2...

[Round 2]
<thinking>...corrected reasoning...</thinking>
[Self-Rating: Consistency=8/10, Accuracy=9/10, ...]

[Convergence: Δ=+4.5, Stop=True]

<answer>...</answer>
```

## 8. Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Intrinsic weight | λ | 1.0 | R_int coefficient |
| Improvement weight | μ | 0.5 | R_improve coefficient |
| Convergence weight | ν | 0.1 | R_converge coefficient |
| Reflection rounds | K | 2-3 | Max iterations |
| KL penalty | β | 0.01 | Reference regularization |
| Candidates | N | 8 | GRPO group size |

## 9. Expected Results

| Model | AIME24 | MATH500 | GPQA |
|-------|--------|---------|------|
| Baseline (SFT) | 50.0 | 92.6 | 56.6 |
| + CoR (self-rating) | 53.3 | 92.8 | 58.0 |
| + Reflection (K=2) | 56.7 | 93.0 | 59.6 |
| + Reflection (K=3) | 58.0 | 93.2 | 60.2 |

## 10. Key Theorems

1. **Synergistic Gain**: Dual coupling evolves faster than independent components
2. **Convergence**: Contraction mapping guarantees fixed-point convergence
3. **Monotonic Improvement**: Quality increases with each reflection round
4. **Lyapunov Stability**: System energy decreases, ensuring continuous evolution

See `theory.md` for full mathematical proofs.
