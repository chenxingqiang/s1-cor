# Chain of Reward (CoR) — Mathematical Theory

## Notation

| Symbol | Definition |
|--------|------------|
| x | Input prompt/question |
| c | Complete reasoning chain (y_think, y_answer) |
| τ | Trajectory {(s₀,a₀), ..., (s_T,a_T)} |
| s_t | State at time t |
| a_t | Action (token) at time t |
| π_θ | Policy parameterized by θ |
| π_ref | Reference policy (SFT model) |
| R_ext(c) | External (task) reward |
| R_int(c) | Intrinsic (thinking quality) reward |
| λ | Weight balancing intrinsic vs external rewards |
| w_d | Weight for dimension d |
| γ | Discount factor |
| β | KL penalty coefficient |
| δ | Clipping range parameter |
| ε | Small constant for numerical stability |
| N | Number of candidates per group |
| D | Number of reward dimensions |

## 1. Problem Formalization

**Definition 1** (Reasoning Chain). Given input x, policy π_θ generates:
- Thinking process: y_think = (a₁, a₂, ..., a_{T_think})
- Final answer: y_answer = (a_{T_think+1}, ..., a_T)
- Complete chain: c = (y_think, y_answer)

**Objective**: Find π*_θ that maximizes expected total reward:

```
J(π_θ) = E_{c~π_θ(·|x), x~D} [R(c)]
```

## 2. Reward Decomposition

### 2.1 Total Reward

```
R(c) = R_ext(c) + λ·R_int(c)
```

### 2.2 External Reward (Sparse)

```
R_ext(c) = I[y_answer = y_gt] = { 1 if correct, 0 otherwise }
```

### 2.3 Intrinsic Reward (Dense, Multi-dimensional)

```
R_int(c) = Σ_{d=1}^D w_d·r_d(y_think) + w_self·r_self_rating_quality
```

where r_d: Y_think → [0,1] measures quality along dimension d.

**Dimensions**:
- r_consistency: Logical coherence
- r_completeness: Step comprehensiveness  
- r_accuracy: Factual correctness
- r_clarity: Reasoning clarity
- r_format: Structural correctness
- r_self_rating_quality: Quality of self-evaluation (endogenous)

### 2.4 Reward Chain (Step-level)

```
CoR(τ) = Σ_{t=0}^T γ^t r_int(s_t, a_t, s_{t+1})
```

## 3. Endogenous Self-Evaluation

### 3.1 Self-Rating

During thinking, model generates:
```
self_rating_d ∈ [0, 10], d = 1, ..., D
```

Example: `[Self-Rating: Consistency=8/10, Completeness=9/10]`

### 3.2 Calibration Function

```
cal_d(u, v) = 1 - |u - v|, where u, v ∈ [0,1]
```

- u = self_rating_d / 10 (normalized self-rating)
- v = actual_quality_d (evaluated by external metric)

### 3.3 Self-Rating Quality Reward

```
r_self_rating_quality = (1/D) Σ_{d=1}^D cal_d(self_rating_d/10, actual_quality_d)
```

**Enhanced calibration** (optional bonus):
```
cal_d^enhanced(u, v) = cal_d(u, v) + α·I[u > 0.8 and v > 0.8]
```

## 4. GRPO Algorithm

### Algorithm 1: GRPO for CoR

For each input x:

1. **Sample**: Generate N candidates {c^(i)}_{i=1}^N from π_θ(·|x)

2. **Reward**: Compute R(c^(i)) = R_ext(c^(i)) + λ·R_int(c^(i))

3. **Statistics**: 
   ```
   μ_R = (1/N) Σ R(c^(i))
   σ²_R = (1/(N-1)) Σ (R(c^(i)) - μ_R)²
   ```

4. **Advantage** (normalized):
   ```
   A^(i) = (R(c^(i)) - μ_R) / (σ_R + ε)
   ```

5. **Decomposed Advantage** (optional):
   ```
   A^(i)_ext = (R_ext(c^(i)) - μ_{R_ext}) / (σ_{R_ext} + ε)
   A^(i)_int = (R_int(c^(i)) - μ_{R_int}) / (σ_{R_int} + ε)
   A^(i)_total = A^(i)_ext + λ·A^(i)_int
   ```

### GRPO Objective Function

```
J(θ) = E_{x~D} [(1/N) Σ_{i=1}^N min(r_i·A^(i), clip(r_i, 1-δ, 1+δ)·A^(i))] - β·D_KL(π_θ || π_ref)
```

where:
- r_i = π_θ(c^(i)|x) / π_{θ_old}(c^(i)|x) is the importance ratio
- clip(r, a, b) = max(a, min(b, r))

## 5. Theoretical Results

### Theorem 1: Unbiasedness of Group-Normalized Advantages

The policy gradient estimate using A^(i) is unbiased:

```
E[∇_θ log π_θ(c^(i)|x)·A^(i)] = E[∇_θ log π_θ(c^(i)|x)·(R(c^(i)) - μ_R)]
```

**Proof**: The scaling factor 1/(σ_R + ε) depends only on group statistics, not individual c^(i). By the baseline property from REINFORCE, subtracting μ_R (independent of individual samples) does not introduce bias. □

### Theorem 2: Clipped Objective Lower Bounds Surrogate

```
min(r_i·A^(i), clip(r_i, 1-δ, 1+δ)·A^(i)) ≤ r_i·A^(i)
```

with equality when r_i ∈ [1-δ, 1+δ]. This provides trust-region constraint.

**Proof**: Case analysis on sign of A^(i) and value of r_i shows the min operation always returns a value ≤ r_i·A^(i). □

### Theorem 3: Potential-Based Shaping Preserves Optimal Policies

If r_int(s, a) = γ·Φ(s') - Φ(s) for some potential Φ: S → R, then optimal policies are invariant.

**Proof** (Ng et al., 1999): Define Q'^*(s,a) = Q^*(s,a) + Φ(s). Since Φ(s) is independent of action a:
```
argmax_a Q'^*(s, a) = argmax_a Q^*(s, a)
```
Therefore optimal policies are preserved. □

## 6. Bellman Extension

### Extended Value Function

```
Q^π(s, a) = E_{s'~P(·|s,a)} [r_int(s, a) + γ·E_{a'~π(·|s')} Q^π(s', a') + λ·r_ext(s, a)]
```

### Potential-Based Intrinsic Reward

```
r_int(s, a) = E_{s'~P(·|s,a)} [φ(s') - φ(s)]
```

where the potential function is:
```
φ(s) = Σ_{k=1}^K w_k·f_k(s)
```

with features f_k measuring reasoning quality.

**Corollary**: Process-quality features as potential provide denser feedback without altering optimal policies.

## 7. Convergence Analysis

### Assumptions

- **A1** (Bounded Rewards): |R_ext(c)| ≤ R_max, |R_int(c)| ≤ R_max
- **A2** (Support Overlap): supp(π_θ) ⊆ supp(π_{θ_old})
- **A3** (Reference Regularization): D_KL(π_θ || π_ref) < ∞
- **A4** (Finite Horizon): P(T < ∞) = 1
- **A5** (Lipschitz): π_θ is L-Lipschitz in θ

### Theorem 4: Monotone Improvement

Under A1-A5, if λ ≥ 0 and β > 0:
```
J(π_{θ_{k+1}}) ≥ J(π_{θ_k}) - O(δ²)
```

### Theorem 5: Convergence to Local Optimum

Under Robbins-Monro conditions (Σα_k = ∞, Σα²_k < ∞), GRPO converges to a local optimum almost surely.

## 8. Calibration Improvement

### Proposition 1: Endogenous Self-Evaluation Improves Calibration

If R_int includes r_self_rating_quality and λ > 0, then maximizing expected return encourages alignment between self-ratings and actual quality.

**Proof**:
1. R(c) is monotone increasing in each cal_d when λ > 0:
   ```
   ∂R(c)/∂cal_d = λ·w_self/D > 0
   ```

2. Higher total reward → higher advantages A^(i)

3. GRPO increases probability of candidates with higher advantages:
   ```
   π_{θ_{k+1}}(c|x) ∝ π_{θ_k}(c|x)·exp(α·A^(c))
   ```

4. Therefore, candidates with better calibration have higher probability, improving calibration over iterations. □

## 9. Multi-Dimensional Scoring

### Definition: Dimension Scoring Function

```
f_d(τ) = g_d({h_{d,t}(s_t, a_t)}_{t=0}^T)
```

where:
- h_{d,t}: S × A → R is feature extractor for dimension d at step t
- g_d: R^{T+1} → [0,1] is aggregation function

### Consistency Constraint (Optional)

```
L_consistency = E[(sign(Σ_d w_d·f_d(τ)) - sign(R_ext(τ)))²]
```

Ensures high-quality reasoning aligns with correct answers.

## 10. Training Algorithm

### Algorithm 2: CoR-GRPO Training

```
for each batch {x_j}_{j=1}^B:
    # 1. Generate candidates
    for each x_j: c_j^(i) ~ π_θ(·|x_j), i=1,...,N
    
    # 2. Extract self-ratings from y_think
    
    # 3. Compute rewards
    R_ext = I[y_answer = y_gt]
    r_d = quality_metric_d(y_think)
    r_self = (1/D) Σ cal_d(self_rating_d/10, r_d)
    R_int = Σ w_d·r_d + w_self·r_self
    R = R_ext + λ·R_int
    
    # 4. Compute advantages
    μ_R, σ_R = group_statistics(R)
    A^(i) = (R^(i) - μ_R) / (σ_R + ε)
    
    # 5. Compute objective
    J(θ) = (1/B) Σ_j (1/N) Σ_i min(r_j^(i)·A_j^(i), clip(...)·A_j^(i)) - β·D_KL
    
    # 6. Update
    θ ← θ + α·∇_θ J(θ)
```

## 11. Practical Recommendations

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| λ | 1.0 | [0.5, 2.0] |
| β | 0.01 | [0.01, 0.1] |
| δ | 0.2 | [0.1, 0.3] |
| N | 8 | [4, 16] |
| D_KL (monitor) | - | [0.01, 0.1] |

**Tips**:
- Normalize rewards to [0,1] for stability
- Monitor KL divergence for stable training
- Start with rule-based intrinsic rewards, then consider learned

## 12. Connections to Existing Theory

### RL Theory Extensions

- **Multi-time-scale rewards**: Dense intrinsic + sparse external
- **Potential-based shaping**: Reasoning quality as potential
- **Group-relative optimization**: Theoretical guarantees for policy improvement

### CoT Reasoning Theory

- **Quantitative measures**: Dimension scoring functions f_d(τ)
- **Optimization framework**: Reward chain CoR(τ)
- **Explicit-to-implicit transfer**: Endogenous self-evaluation improves meta-cognition

## 13. References

1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
2. Ng et al. (1999). "Policy Invariance under Reward Transformations"
3. Wei et al. (2022). "Chain-of-Thought Prompting"
4. Wang et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning"
5. TRL Documentation: GRPOTrainer, GRPOConfig
