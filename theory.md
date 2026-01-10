# Chain of Reward (CoR) — Mathematical Theory

## Notation

| Symbol | Definition |
|--------|------------|
| x | Input prompt/question |
| c | Complete reasoning chain (y_think, y_answer) |
| π_θ | Policy parameterized by θ |
| R_ext(c) | External (task) reward |
| R_int(c) | Intrinsic (thinking quality) reward |
| λ | Weight balancing intrinsic vs external rewards |
| K | Number of reflection rounds |
| T | Self-reflection operator |

## 1. Core Reward Formula

```
R(c) = R_ext(c) + λ·R_int(c)
```

Where:
- R_ext: Binary correctness (sparse)
- R_int: Multi-dimensional quality + self-rating calibration (dense)

## 2. Intrinsic Reward (5 dimensions + self-rating)

```
R_int(c) = Σ_{d=1}^5 w_d·r_d(y_think) + w_self·r_self_rating_quality
```

Dimensions: Consistency, Completeness, Accuracy, Clarity, Format (w_d = 0.2 each)

## 3. Self-Rating Calibration

```
cal_d(u, v) = 1 - |u - v|
r_self = (1/D) Σ cal_d(self_rating_d/10, actual_d)
```

---

# Part II: CoR-GRPO Dual Coupling Theory

## 4. Dual Coupling Structure

CoR and GRPO form a bidirectional coupled evolutionary system:

```
┌─────────────────────────────────────────────────────┐
│              CoR-GRPO Dual Coupling                 │
├─────────────────────────────────────────────────────┤
│   ┌─────────┐       Coupling 1        ┌─────────┐  │
│   │   CoR   │  ────────────────────►  │  GRPO   │  │
│   │  Reward │   R(θ, φ) signal        │ Policy  │  │
│   │  Chain  │  ◄────────────────────  │ Optim.  │  │
│   └─────────┘       Coupling 2        └─────────┘  │
│               Improved calibration φ                │
└─────────────────────────────────────────────────────┘
```

## 5. Coupling Dynamics Equations

**Coupling 1: CoR → GRPO**
```
θ_{t+1} = θ_t + α · ∇_θ E[R_CoR(θ_t, φ_t)]
```

**Coupling 2: GRPO → CoR**
```
φ_{t+1} = φ_t + β · ∇_φ E[cal(g_φ(c), f_θ(c))]
```

## 6. Theorem: Synergistic Gain

```
d/dt(Q_θ + Q_φ) > dQ_θ/dt|_{φ=const} + dQ_φ/dt|_{θ=const}
```

The coupled system evolves faster than independent components.

---

# Part III: Self-Reflection Framework

## 7. Iterative Refinement Operator

```
c_{k+1} = T_θ(c_k, self_rating_k, x)
```

## 8. Contraction Mapping Convergence

If T is a contraction: d(T(c), T(c')) ≤ γ·d(c, c'), then:
```
lim_{k→∞} c_k = c* (unique fixed point)
```

## 9. Improvement Reward

```
R_improve(c_k, c_{k+1}) = Q(c_{k+1}) - Q(c_k)
```

## 10. Cumulative Intelligence Theorem

If E[R_improve] = μ > 0, then after K rounds:
```
E[Q(c_K) - Q(c_0)] = μ · (1-γ^K)/(1-γ)
```

**More reasoning → More intelligent**

## 11. Extended Reward with Reflection

```
R_CoR_Reflect = R_ext + λ·R_int + μ·R_improve + ν·R_converge
```

## 12. Lyapunov Stability

```
V(θ, φ) = -E[R_total(θ, φ)]
dV/dt < 0  →  Continuous evolution guaranteed
```

---

# Part IV: Implementation

## 13. Algorithm: CoR-Reflect Training

```
for each batch:
    # 1. Initial generation
    c^{(0)} ~ π_θ(·|x)
    
    # 2. Multi-round reflection
    for k = 0 to K-1:
        self_rating_k = extract_ratings(c^{(k)})
        reflection = π_θ(reflect | c^{(k)}, self_rating_k)
        c^{(k+1)} ~ π_θ(·| x, reflection)
        R_improve^{(k)} = Q(c^{(k+1)}) - Q(c^{(k)})
    
    # 3. Total reward
    R_total = R_ext + λ·R_int + μ·Σ_k R_improve^{(k)}
    
    # 4. GRPO update
    θ ← θ + α·∇_θ J(θ)
```

## 14. Data Format

```
[Round 1]
<thinking>...initial reasoning...</thinking>
[Self-Rating: Consistency=4/10, Accuracy=3/10, ...]

[Reflection]
Accuracy is low. Error in step 2...

[Round 2]
<thinking>...corrected reasoning...</thinking>
[Self-Rating: Consistency=8/10, Accuracy=9/10, ...]

[Convergence: Improvement=+4.5, Continue=No]
```

## 15. Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ | 1.0 | Intrinsic weight |
| K | 2-3 | Reflection rounds |
| μ | 0.5 | Improvement weight |
| ν | 0.1 | Convergence weight |

## References

1. Schulman et al. (2017). PPO
2. Ng et al. (1999). Reward Shaping
3. Wei et al. (2022). Chain-of-Thought
4. Banach Fixed-Point Theorem
5. Lyapunov Stability Theory
