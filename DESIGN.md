# Design Document: GRPO-based Chain of Reward (CoR) Implementation

**Phase**: Design  
**Date**: 2025-01-XX  
**Status**: Pending Approval

## 1. Overview

This document outlines the design for implementing **Chain of Reward (CoR)** based on **Group Relative Policy Optimization (GRPO)** framework, building upon the s1 project architecture. The implementation will extend s1's training pipeline to incorporate fine-grained reward signals along the reasoning chain, enabling optimization of both final answers and intermediate thinking steps.

## 2. Objectives

### 2.1 Primary Goals
- Extend s1's training framework to support GRPO-based reinforcement learning
- Implement reward chain calculation combining external (sparse) and intrinsic (dense) rewards
- Design multi-dimensional scoring system for intrinsic rewards
- Maintain compatibility with existing s1 infrastructure (data format, evaluation pipeline)

### 2.2 Success Criteria
- GRPO training script that can train models with CoR
- Reward calculation module supporting multi-dimensional intrinsic rewards
- Integration with existing s1 evaluation benchmarks (AIME24, MATH500, GPQA Diamond)
- Performance improvement over baseline s1 model on reasoning tasks

## 3. Architecture Design

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     CoR Training Pipeline                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌──────────▼──────────┐  ┌──────▼──────────┐
│ Data Loading   │  │  Generation Module  │  │ Reward Module   │
│                │  │                     │  │                 │
│ - Load dataset │  │ - Generate N chains │  │ - External RM   │
│ - Format CoT   │  │ - Sample with      │  │ - Intrinsic RM  │
│   sequences    │  │   temperature      │  │ - Multi-dim     │
│                │  │                     │  │   scoring       │
└────────┬───────┘  └──────────┬──────────┘  └──────┬──────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   GRPO Trainer    │
                    │                   │
                    │ - Advantage calc  │
                    │ - Policy update   │
                    │ - KL penalty      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Model Checkpoint │
                    └───────────────────┘
```

### 3.2 Core Modules

#### 3.2.1 Reward Calculation Module (`train/rewards.py`)

**Responsibilities**:
- Calculate external task reward (sparse, based on final answer correctness)
- **Extract and evaluate model's self-ratings** (endogenous evaluation)
- Calculate intrinsic thinking rewards (dense, multi-dimensional)
- Combine rewards with configurable weighting

**Key Innovation: Endogenous Self-Evaluation** (per target.md Section 112-116):
The model generates multi-dimensional self-ratings during thinking, which are then evaluated:
1. Model generates self-ratings: `[逻辑一致性: 8/10, 步骤完整性: 9/10, 事实准确性: 7/10]`
2. Reward calculator evaluates quality of these self-ratings
3. Self-rating quality contributes to intrinsic rewards

**Key Classes**:
```python
class RewardCalculator:
    def __init__(self, lambda_intrinsic: float = 1.0):
        self.lambda_intrinsic = lambda_intrinsic
    
    def calculate_external_reward(self, answer: str, ground_truth: str) -> float:
        """Calculate sparse reward based on final answer correctness."""
        # Returns 1.0 if correct, 0.0 if incorrect
        
    def extract_self_ratings(self, thinking_chain: str) -> dict:
        """Extract multi-dimensional self-ratings from thinking chain.
        
        Looks for patterns like:
        - [逻辑一致性: 8/10]
        - [Self-Rating: Consistency=8/10, Completeness=9/10]
        - [评分: {consistency: 0.8, completeness: 0.9}]
        
        Returns dict with extracted ratings per dimension.
        """
        
    def evaluate_self_rating_quality(self, self_ratings: dict, 
                                     thinking_chain: str, 
                                     final_answer: str,
                                     ground_truth: str) -> dict:
        """Evaluate quality of model's self-ratings.
        
        Quality metrics:
        1. Consistency: Do self-ratings align with actual thinking quality?
        2. Calibration: Are high self-ratings associated with correct answers?
        3. Completeness: Are all expected dimensions rated?
        4. Reasonableness: Are self-ratings in plausible ranges?
        
        Returns dict with quality scores for each metric.
        """
        
    def calculate_intrinsic_reward(self, thinking_chain: str, 
                                   self_ratings: dict = None) -> dict:
        """Calculate dense multi-dimensional intrinsic rewards.
        
        Returns dict with rewards for:
        - consistency: logical coherence of reasoning
        - self_rating_quality: quality of model's self-evaluation (NEW)
        - self_rating_calibration: alignment of self-ratings with correctness (NEW)
        - confidence: alignment of self-ratings with final answer
        - format: structural correctness
        - step_completeness: reasoning step quality
        
        If self_ratings are provided, uses them; otherwise extracts from chain.
        """
        
    def calculate_total_reward(self, chain: Chain, ground_truth: str) -> float:
        """Calculate combined reward: R(c) = R_ext(c) + λ * R_int(c)"""
        # Extract self-ratings first
        self_ratings = self.extract_self_ratings(chain.thinking)
        # Then calculate intrinsic reward including self-rating quality
        r_int = self.calculate_intrinsic_reward(chain.thinking, self_ratings)
        # Combine with external reward
        ...
```

#### 3.2.2 GRPO Training Module (`train/grpo.py`)

**Responsibilities**:
- Use TRL's `GRPOTrainer` for policy optimization
- Integrate reward calculator with TRL's reward function interface
- Configure training parameters via `GRPOConfig`
- Generate multiple candidate chains per input (handled by TRL)

**Implementation using TRL**:
```python
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainingArguments

class CoRGRPOTrainer:
    def __init__(
        self,
        model,
        reward_calculator: RewardCalculator,
        train_dataset,
        config: GRPOConfig
    ):
        # Wrap reward calculator to match TRL's interface
        def reward_func(completions, prompts=None, **kwargs):
            """TRL-compatible reward function."""
            rewards = []
            for completion, prompt in zip(completions, prompts or [None]*len(completions)):
                # Extract thinking chain and answer from completion
                chain = self._parse_completion(completion)
                # Get ground truth from dataset if available
                ground_truth = kwargs.get('ground_truth', None)
                
                # Calculate total reward
                total_reward = reward_calculator.calculate_total_reward(
                    chain, ground_truth
                )
                rewards.append(total_reward)
            return rewards
        
        # Initialize TRL's GRPOTrainer
        self.trainer = GRPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            reward_funcs=reward_func,
        )
    
    def train(self):
        """Start training."""
        return self.trainer.train()
    
    def _parse_completion(self, completion: str) -> Chain:
        """Parse completion string into Chain object."""
        # Extract thinking and answer parts
        # Format: <|im_start|>assistant\n{thinking}<|im_end|>\n{answer}
        ...
```

**Key Design Decision**: Use TRL's built-in `GRPOTrainer` instead of implementing from scratch, as it:
- Handles candidate generation automatically
- Implements advantage calculation internally
- Supports FSDP and multi-node training
- Has proven stability and performance

**Customization Points**:
- Reward function wrapper to integrate our multi-dimensional reward calculator
- Data collator for CoR format (thinking chain + answer separation)
- Custom logging for reward distribution analysis

#### 3.2.3 Prompt Template for Endogenous Self-Evaluation

**Key Design**: Model must generate self-ratings during thinking (per target.md).

**Prompt Template**:
```python
SYSTEM_PROMPT = """You are a helpful assistant that thinks step-by-step and evaluates your own reasoning.

During your thinking process, you should:
1. Break down the problem into steps
2. For each major step, provide self-ratings in the format:
   [Self-Rating: Dimension=Score/10]
   
   Required dimensions:
   - Consistency: Does this step logically follow from previous steps? (0-10)
   - Completeness: Is this step comprehensive? (0-10)
   - Accuracy: Are the facts/calculations correct? (0-10)
   - Clarity: Is the reasoning clear? (0-10)

3. After completing all steps, provide a summary self-rating

Example format:
Step 1: ...
[Self-Rating: Consistency=8/10, Completeness=9/10, Accuracy=7/10, Clarity=8/10]

Step 2: ...
[Self-Rating: Consistency=9/10, Completeness=8/10, Accuracy=9/10, Clarity=9/10]

Summary: [Overall Quality: 8.5/10]"""
```

**Training Data Format**:
During SFT phase, training examples should include self-ratings in thinking chain:
```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
{thinking_with_self_ratings}
<|im_end|>\n{answer}
```

#### 3.2.4 Data Format Extensions

**Current s1 Format**:
- Text field containing: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{thinking}<|im_end|>`

**Extended Format for CoR**:
- Add fields for:
  - `ground_truth`: Correct answer for external reward
  - `thinking_chain`: Explicit thinking steps with self-ratings
  - `self_ratings_extracted`: Parsed self-ratings (optional, for training analysis)
  - `multi_dim_scores`: Pre-computed scores (optional, for training reward model)

## 4. Reward Design

### 4.1 External Reward (R_ext)

**Definition**: 
```
R_ext(c) = I(y_answer == y_ground_truth)
```

- Binary reward: 1.0 if correct, 0.0 if incorrect
- Evaluated using same grading protocol as s1 (Claude 3.5 Sonnet comparison)

### 4.2 Intrinsic Reward (R_int)

**Multi-dimensional decomposition** (per target.md):
```
R_int(c) = Σ_d w_d * r_d(y_think)
```

**Reward Chain Formulation** (from target.md Section 1.2):
The reward chain is formally defined as:
```
COR(τ) = Σ_{t=0}^T γ^t r_int(s_t, a_t, s_{t+1})
```
where γ is the discount factor and r_int is the intrinsic reward function for each step.

For our implementation, we aggregate step-level rewards into sequence-level reward:
```
R_int(c) = Σ_d w_d * r_d(y_think)  # Aggregated over all steps
```

**Dimensions**:

#### 4.2.1 Consistency Reward (r_consistency)
- **Purpose**: Ensure reasoning steps logically lead to final answer
- **Calculation**: 
  - Extract intermediate conclusions from thinking chain
  - Check if final answer is derivable from reasoning steps
  - Score: 1.0 if consistent, 0.0 if contradictory

#### 4.2.2 Self-Rating Quality Reward (r_self_rating_quality) ⭐ **NEW - Core Endogenous Feature**

- **Purpose**: Reward model for accurately evaluating its own thinking quality
- **Core Innovation**: Model evaluates itself, then we evaluate the quality of its evaluation
- **Calculation**:
  1. Extract self-ratings from thinking chain
  2. Evaluate actual thinking quality using rule-based metrics
  3. Compare self-ratings with actual quality:
     - Calibration: Are high self-ratings associated with high actual quality?
     - Consistency: Do self-ratings match actual dimension scores?
     - Reasonableness: Are self-ratings in plausible ranges?
  4. Score based on alignment between self-assessment and actual quality
  
- **Mathematical Formulation** (per target.md Section 112-116):
```
r_self_rating_quality = Σ_d w_d * calibration_d(self_rating_d, actual_quality_d)
```

Where `calibration_d` measures how well model's self-rating for dimension d aligns with actual quality d.

- **Implementation**:
```python
def evaluate_self_rating_quality(self_ratings, actual_quality):
    """Evaluate quality of model's self-evaluation."""
    calibration_scores = {}
    for dimension in self_ratings:
        model_rating = self_ratings[dimension] / 10.0  # Normalize to [0,1]
        actual_score = actual_quality[dimension]
        
        # Calibration: how close is self-rating to actual?
        calibration_scores[dimension] = 1.0 - abs(model_rating - actual_score)
        
        # Bonus: if model correctly identifies high quality → reward
        if model_rating > 0.8 and actual_score > 0.8:
            calibration_scores[dimension] += 0.2
    
    return sum(calibration_scores.values()) / len(calibration_scores)
```

#### 4.2.3 Confidence Reward (r_confidence)
- **Purpose**: Reward high self-confidence when answer is correct
- **Calculation**:
  - Extract self-rating scores from thinking chain (required now)
  - If final answer correct AND self-rating high → reward
  - Encourages "knowing when you know"
  - **Updated**: Now uses self-ratings generated by model (endogenous), not external evaluation

#### 4.2.3 Format Reward (r_format)
- **Purpose**: Ensure proper CoT structure
- **Calculation**:
  - Check for thinking markers (`<|im_start|>think`, etc.)
  - Verify separation between thinking and final answer
  - Score: 1.0 if properly formatted, 0.0 otherwise

#### 4.2.4 Step Completeness Reward (r_steps)
- **Purpose**: Reward comprehensive reasoning steps
- **Calculation**:
  - Count reasoning steps (e.g., numbered steps, explicit transitions)
  - Reward models that break down problems systematically
  - Score: min(1.0, num_steps / expected_steps)

#### 4.2.5 Potential Function-based Reward (per target.md Section 3.2)

**Purpose**: Based on Bellman equation extension (target.md Section 3.1), we define intrinsic reward as potential difference:
```
r_int(s, a) = E[φ(s') - φ(s)]
```

Where φ(s) is a potential function measuring reasoning quality.

**Potential Function** (from target.md):
```
φ(s) = Σ_k w_k * f_k(s)
```

Where f_k(s) are reasoning quality features (logical consistency, step completeness, etc.).

**Implementation**:
```python
def calculate_potential_function(state: ReasoningState) -> float:
    """Calculate φ(s) = Σ_k w_k * f_k(s)"""
    features = {
        'logical_consistency': check_logical_consistency(state),
        'step_completeness': count_completeness(state),
        'factual_accuracy': check_factual_accuracy(state),
        ...
    }
    weights = {...}  # Learned or fixed
    return sum(weights[k] * features[k] for k in features)

def calculate_step_reward(prev_state, action, next_state):
    """r_int(s, a) = φ(s') - φ(s)"""
    return calculate_potential_function(next_state) - calculate_potential_function(prev_state)
```

### 4.3 Reward Model Implementation

**Option A: Rule-based (Initial Implementation)**
- Use predefined rules and heuristics
- Fast, deterministic, no training required
- Baseline for validation

**Option B: Learned Reward Model (Future)**
- Train a separate model to evaluate thinking quality
- Requires annotated thinking chains
- More flexible but computationally expensive

**Initial Implementation**: Start with Option A, design interface for future Option B integration.

## 5. GRPO Algorithm Details

### 5.1 Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Step 1: Generate candidates
        chains = []
        for prompt in batch:
            candidates = trainer.generate_candidates(
                prompt, 
                num_candidates=N
            )
            chains.extend(candidates)
        
        # Step 2: Calculate rewards
        rewards = []
        for chain in chains:
            total_reward = reward_calculator.calculate_total_reward(
                chain, 
                ground_truth
            )
            rewards.append(total_reward)
        
        # Step 3: Calculate advantages
        advantages = trainer.calculate_advantages(rewards)
        
        # Step 4: Compute loss and update
        old_log_probs = compute_log_probs(chains, old_model)
        loss = trainer.compute_loss(chains, advantages, old_log_probs)
        
        # Step 5: Optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Step 6: Update old model (for next iteration)
        old_model.load_state_dict(model.state_dict())
```

### 5.2 Advantage Calculation

**Mathematical Formulation** (from target.md):
```
A^(i) = (R(i) - μ_R) / (σ_R + ε)
```

Where:
- R(i) = R(c(i)) is the total reward for candidate i
- μ_R and σ_R are the mean and standard deviation of rewards in the group
- ε is a small constant for numerical stability

**Implementation Note**: TRL's GRPOTrainer handles advantage calculation internally. However, for CoR with intrinsic rewards, we need to ensure the reward function returns the combined reward:

```python
def calculate_total_reward(chain: Chain, ground_truth: str) -> float:
    """Calculate R(c) = R_ext(c) + λ * R_int(c)"""
    r_ext = reward_calculator.calculate_external_reward(
        chain.answer, ground_truth
    )
    r_int = reward_calculator.calculate_intrinsic_reward(chain.thinking)
    total = r_ext + lambda_intrinsic * r_int
    return total
```

**Multi-dimensional Intrinsic Advantage** (from target.md Section 2.2):
For fine-grained control, we can compute per-dimension advantages:
```
A^(i)_int = (1/D) * Σ_d α_d * (R^(i)_int,d - μ_R_int,d) / (σ_R_int,d + ε)
```

Then combine:
```
A^(i)_total = A^(i)_ext + λ * A^(i)_int
```

**Implementation Strategy**: For initial version, compute total reward and let TRL handle advantage normalization. For advanced version, we can customize advantage calculation.

**Endogenous Self-Evaluation in Advantage** (per target.md):
The advantage calculation naturally incorporates self-rating quality:
- Models with better self-assessment capabilities will have higher intrinsic rewards
- Higher intrinsic rewards lead to higher total rewards
- Higher total rewards translate to higher advantages
- GRPO optimizes for models that both think well AND evaluate themselves well

This creates a virtuous cycle: models learn to think better AND to evaluate their thinking better.

### 5.3 GRPO Loss Function

**Mathematical Formulation** (from target.md Section 2.1):
```
J(θ) = E_{x~X} [
    (1/N) * Σ_{i=1}^N [
        min(
            (π_θ(c^(i)|x) / π_θ_old(c^(i)|x)) * A^(i)_total,
            clip(π_θ(c^(i)|x) / π_θ_old(c^(i)|x), 1-δ, 1+δ) * A^(i)_total
        )
    ]
] - β * D_KL(π_θ || π_ref)
```

Where:
- π_θ is the current policy
- π_θ_old is the old policy (for importance sampling)
- π_ref is the reference policy (for KL penalty)
- A^(i)_total = A^(i)_ext + λ * A^(i)_int (total advantage with intrinsic rewards)
- β controls KL penalty strength

**Implementation**: TRL's GRPOTrainer implements this internally. We configure via `GRPOConfig`:

```python
training_args = GRPOConfig(
    # Clipping
    epsilon=0.2,  # clip_epsilon = δ
    
    # KL penalty
    beta=0.01,  # β for KL divergence penalty
    
    # Other parameters
    num_generations=4,  # N (number of candidates per input)
    ...
)
```

**Extended Objective with Intrinsic Rewards** (from target.md):
The key modification is that A^(i)_total now includes intrinsic advantages:
```
A^(i)_total = A^(i)_ext + λ * A^(i)_int
```

Where A^(i)_int is computed from multi-dimensional intrinsic rewards per target.md Section 2.2.

## 6. Integration with s1 Infrastructure

### 6.1 Data Compatibility
- Use existing s1K dataset format
- Extend with ground truth labels (already available in evaluation)
- Support both SFT and GRPO training from same data

### 6.2 Training Script Structure
- Extend `train/sft.py` → create `train/grpo.py`
- Reuse FSDP configuration from s1
- Support multi-node training (same as s1)

### 6.3 Evaluation Integration
- Use existing `eval/lm-evaluation-harness` setup
- Add CoR-specific metrics (thinking quality, reward distribution)
- Maintain compatibility with AIME24, MATH500, GPQA benchmarks

## 7. Implementation Plan

### Phase 1: Core Components (Week 1)
1. Implement `RewardCalculator` class
   - External reward calculation
   - **Self-rating extraction** (NEW - core endogenous feature)
   - **Self-rating quality evaluation** (NEW - core endogenous feature)
   - Basic intrinsic rewards (rule-based)
   - Total reward combination

2. Design prompt template for self-evaluation
   - System prompt requiring self-ratings during thinking
   - Example format for multi-dimensional ratings
   - Integration with existing s1 format

3. Implement `GRPOTrainer` skeleton using TRL
   - Candidate generation (handled by TRL)
   - Advantage calculation (handled by TRL)
   - Reward function wrapper for TRL interface

### Phase 2: Training Integration (Week 2)
3. Prepare training data with self-evaluations
   - **Create SFT dataset** with examples containing self-ratings
   - Use GPT-4o or similar to generate thinking chains with self-ratings
   - Filter high-quality examples (per target.md Section 90-95)

4. Integrate GRPO into training pipeline
   - Extend data loading for CoR format with self-ratings
   - Implement training loop with TRL's GRPOTrainer
   - Add logging for self-rating quality metrics

5. Testing and validation
   - Unit tests for self-rating extraction
   - Unit tests for self-rating quality evaluation
   - Integration tests for training loop
   - Validate on small dataset first

### Phase 3: Evaluation and Paper (Week 3-4)
5. Evaluation setup
   - Benchmark on AIME24, MATH500, GPQA
   - Compare against baseline s1 model
   - Analyze reward distribution and scaling

6. Paper writing
   - Extend s1 paper structure
   - Add CoR methodology section
   - Include theoretical derivation from target.md

## 8. Configuration

### 8.1 Hyperparameters

**Using TRL's GRPOConfig**:
```python
from trl import GRPOConfig

training_args = GRPOConfig(
    # Model & Output
    output_dir="ckpts/s1-cor-grpo",
    
    # GRPO specific (TRL parameters)
    num_generations=4,  # Number of candidates per input (N)
    epsilon=0.2,  # Clip epsilon (δ)
    beta=0.01,  # KL penalty coefficient
    
    # Reward configuration (our custom parameters)
    # These are handled via reward function, not config
    
    # Training (standard Transformers TrainingArguments)
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    
    # Generation
    max_completion_length=32768,  # max_tokens for generation
    generation_kwargs={
        "temperature": 1.0,  # For candidate generation diversity
        "top_p": 0.95,
        "do_sample": True,
    },
    
    # FSDP (reuse s1's config)
    fsdp="full_shard auto_wrap",
    fsdp_config="train/fsdp_config_qwen.json",
    bf16=True,
    
    # Logging
    logging_steps=1,
    report_to="wandb",
)
```

**Custom Configuration for CoR**:
```python
@dataclass
class CoRConfig:
    """Custom configuration for Chain of Reward training."""
    # Reward parameters
    lambda_intrinsic: float = 1.0  # Weight for intrinsic rewards
    reward_dimensions: List[str] = field(default_factory=lambda: [
        "consistency", "confidence", "format", "step_completeness", "potential"
    ])
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "consistency": 0.25,
        "confidence": 0.25,
        "format": 0.15,
        "step_completeness": 0.15,
        "potential": 0.20,
    })
    
    # Potential function parameters (from target.md Section 3.2)
    use_potential_function: bool = True
    potential_features: List[str] = field(default_factory=lambda: [
        "logical_consistency",
        "step_completeness",
        "factual_accuracy"
    ])
    
    # Discount factor for reward chain (from target.md Section 1.2)
    gamma: float = 0.99  # Discount factor for step-level rewards
    
    # Consistency constraint (from target.md Section 5.2)
    use_consistency_loss: bool = True
    consistency_loss_weight: float = 0.1
```

## 9. Testing Strategy

### 9.1 Unit Tests
- `test_rewards.py`: Test reward calculation for each dimension
- `test_advantages.py`: Test advantage normalization
- `test_grpo_loss.py`: Test loss computation

### 9.2 Integration Tests
- `test_training_loop.py`: End-to-end training on small dataset
- `test_evaluation.py`: Integration with evaluation pipeline

### 9.3 Validation
- Compare reward distributions across training epochs
- Verify KL divergence stays within acceptable bounds
- Check that intrinsic rewards correlate with external rewards

## 10. Additional Components from target.md

### 10.1 Bellman Equation Extension (from target.md Section 3.1)

**Extended Bellman Optimality Equation**:
```
Q^*(s, a) = E[r_int(s, a) + γ * max_{a'} Q^*(s', a') + λ * r_ext(s, a)]
```

Where r_int(s, a) is defined via potential function:
```
r_int(s, a) = E[φ(s') - φ(s)]
```

**Implementation Consideration**: For our sequence-level reward, we aggregate step-level potential differences.

### 10.2 Consistency Constraint (from target.md Section 5.2)

**Mathematical Formulation**:
```
L_consistency = E[(sign(Σ_d w_d * f_d(τ)) - sign(R_ext(τ)))^2]
```

**Purpose**: Ensure reasoning process aligns with final decision.

**Implementation**:
```python
def compute_consistency_loss(intrinsic_scores, external_reward):
    """Compute consistency constraint loss."""
    intrinsic_aggregate = sum(w_d * f_d for w_d, f_d in intrinsic_scores.items())
    sign_intrinsic = torch.sign(intrinsic_aggregate)
    sign_extrinsic = torch.sign(external_reward)
    return (sign_intrinsic - sign_extrinsic) ** 2
```

**Integration**: Can be added as additional loss term in training loop (if TRL allows custom loss components).

### 10.3 Convergence Theorems (from target.md Section 4)

**Theorem 1 (Policy Improvement)**: With intrinsic rewards, policy improvement is guaranteed if reward functions are bounded and properly weighted.

**Theorem 2 (Convergence Conditions)**:
1. Reward functions bounded: r_int(s,a) ≤ R_max, r_ext(s,a) ≤ R_max
2. Policy space is compact
3. Learning rate satisfies Robbins-Monro conditions

**Practical Implications**: 
- Normalize all rewards to [0, 1] range
- Use appropriate learning rate scheduling
- Monitor KL divergence to ensure policy doesn't diverge too far

## 11. Open Questions & Risks

### 11.1 Open Questions
1. **Reward model training**: Should we train a separate RM or use rule-based initially?
   - **Decision**: Start with rule-based, design for future RM integration

2. **Multi-dimensional weight tuning**: How to determine optimal w_d weights?
   - **Decision**: Use equal weights initially, tune via ablation studies

3. **KL penalty strength**: What β value maintains stability?
   - **Decision**: Start with 0.01 (standard PPO range), adjust based on training stability

4. **Potential function features**: Which f_k(s) features are most effective?
   - **Decision**: Start with logical consistency and step completeness, expand based on empirical results

5. **Discount factor γ**: How to choose for step-level reward aggregation?
   - **Decision**: Use γ=0.99 initially (standard RL), but may need tuning for reasoning tasks

### 11.2 Risks & Mitigation
- **Risk**: Training instability from reward scale mismatch
  - **Mitigation**: Normalize rewards to [0, 1], careful λ tuning, monitor reward distribution
  
- **Risk**: Computationally expensive (N candidates per sample)
  - **Mitigation**: Start with small N=4, optimize generation batching, use vLLM if available
  
- **Risk**: Overfitting to reward dimensions
  - **Mitigation**: KL penalty (β), validation on held-out set, early stopping
  
- **Risk**: TRL's GRPOTrainer may not support all our customizations
  - **Mitigation**: Start with standard GRPO, gradually add customizations via reward function wrapper. If needed, extend TRL's trainer or fork for advanced features

## 12. Success Metrics

### 12.1 Training Metrics
- Reward improvement over epochs
- KL divergence stability
- Training loss convergence

### 12.2 Evaluation Metrics
- Accuracy on AIME24, MATH500, GPQA (vs baseline s1)
- Thinking quality scores (human evaluation)
- Reward distribution analysis

### 12.3 Paper Metrics
- Clear theoretical contribution (from target.md)
- Empirical validation of CoR benefits
- Comparison with baseline methods

---

## 13. Endogenous Reward: Key Innovation ⭐

### 13.1 Core Concept

**Endogenous Reward** means the model actively evaluates its own thinking during reasoning:
1. Model generates self-ratings: `[Self-Rating: Consistency=8/10, Completeness=9/10]`
2. Reward calculator evaluates quality of these self-ratings
3. Self-rating quality contributes to intrinsic rewards

**This creates a virtuous cycle**:
- Models learn to think better
- Models learn to evaluate their thinking better
- Better self-evaluation → Better intrinsic rewards → Better thinking

### 13.2 Implementation Status

✅ **Design Complete**: 
- Self-rating extraction (Section 3.2.1)
- Self-rating quality evaluation (Section 4.2.2)
- Prompt template for self-evaluation (Section 3.2.3)
- Training pipeline with self-ratings (Section 7)

⏳ **Implementation Pending**:
- Code implementation of self-rating extraction
- Code implementation of self-rating quality evaluation
- SFT data generation with self-ratings

**See also**: `DESIGN_ENDOGENOUS.md` for detailed design of endogenous rewards.

## Approval

**Design Status**: Updated with Endogenous Reward Feature

**Key Addition**: Model self-evaluation during thinking is now a core component of the design, addressing the fundamental requirement from target.md Section 112-116.

**Next Steps**: Upon approval, proceed to Development phase with Phase 1 implementation, starting with self-rating extraction and evaluation.

**Questions for Review**:
1. ✅ Does the endogenous reward design capture the core idea?
2. ✅ Is the self-rating format suitable for training?
3. ✅ Should we prioritize rule-based or learned reward model? (Start with rule-based)

