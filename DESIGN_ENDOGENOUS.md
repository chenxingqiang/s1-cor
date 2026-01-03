# Endogenous Reward Design: Model Self-Evaluation

## Core Innovation: Model Evaluates Itself

This document clarifies the **endogenous reward** design, addressing the key requirement from target.md: **models should evaluate their own thinking during the reasoning process**.

## Problem with Initial Design

**Initial Design** (Missing Endogenous Feature):
- External `RewardCalculator` evaluates thinking chain after generation
- Model generates thinking → External calculator evaluates → Reward assigned
- Model doesn't actively think about its own reasoning quality

**This is NOT truly endogenous** - it's still external evaluation, just more fine-grained.

## True Endogenous Reward (Per target.md)

**Core Idea** (from target.md Section 112-116):
> "引导模型在思考时，为自己在多个子维度上的表现进行自我评分"
> (Guide models to self-rate their performance on multiple sub-dimensions during thinking)

**Key Requirement** (from target.md Section 94):
> "训练模型学会生成包含自我评估步骤（如[Self-Rating: 8/10]）的思考链格式"
> (Train models to generate thinking chain format that includes self-evaluation steps)

## Implementation Design

### 1. Self-Rating Generation (During Thinking)

**Prompt Template**:
```
System: You are an assistant that thinks step-by-step and evaluates your reasoning.

During thinking, provide self-ratings:
[Self-Rating: Dimension=Score/10]

Required dimensions:
- Consistency: Logical coherence (0-10)
- Completeness: Step comprehensiveness (0-10)
- Accuracy: Factual correctness (0-10)
- Clarity: Reasoning clarity (0-10)

Example:
Step 1: Let me analyze the problem...
[Self-Rating: Consistency=8/10, Completeness=9/10, Accuracy=7/10, Clarity=8/10]

Step 2: Based on Step 1, I conclude...
[Self-Rating: Consistency=9/10, Completeness=8/10, Accuracy=9/10, Clarity=9/10]
```

### 2. Self-Rating Extraction

```python
def extract_self_ratings(thinking_chain: str) -> dict:
    """Extract self-ratings from thinking chain.
    
    Supports multiple formats:
    - [Self-Rating: Consistency=8/10, Completeness=9/10]
    - [逻辑一致性: 8/10, 步骤完整性: 9/10]
    - [评分: {consistency: 0.8, completeness: 0.9}]
    """
    ratings = {}
    # Pattern matching to extract ratings
    # Returns: {'consistency': 0.8, 'completeness': 0.9, ...}
    return ratings
```

### 3. Self-Rating Quality Evaluation

**Core Innovation**: We evaluate the model's evaluation of itself.

```python
def evaluate_self_rating_quality(
    self_ratings: dict,
    actual_quality: dict,
    final_answer_correct: bool
) -> dict:
    """Evaluate quality of model's self-assessment.
    
    Metrics:
    1. Calibration: Do self-ratings align with actual quality?
       - If model rates Consistency=8/10 and actual is 8/10 → high calibration
    2. Calibration with correctness: Do high self-ratings correlate with correct answers?
       - If model rates highly AND answer is correct → reward
    3. Consistency: Are self-ratings internally consistent?
       - If all dimensions rated similarly → consistent
    4. Reasonableness: Are self-ratings in plausible ranges?
       - Extreme ratings (0/10 or 10/10) might be less reasonable
    """
    quality_scores = {}
    
    # Calibration: alignment with actual quality
    for dim in self_ratings:
        model_score = self_ratings[dim] / 10.0
        actual_score = actual_quality[dim]
        quality_scores[f'{dim}_calibration'] = 1.0 - abs(model_score - actual_score)
    
    # Calibration with correctness
    avg_self_rating = sum(self_ratings.values()) / (10.0 * len(self_ratings))
    if final_answer_correct and avg_self_rating > 0.7:
        quality_scores['correctness_calibration'] = avg_self_rating
    elif not final_answer_correct and avg_self_rating < 0.5:
        quality_scores['correctness_calibration'] = 1.0 - avg_self_rating
    else:
        quality_scores['correctness_calibration'] = 0.0
    
    return quality_scores
```

### 4. Intrinsic Reward Calculation

**Updated Formula**:
```
R_int(c) = Σ_d w_d * r_d(y_think) + w_self * r_self_rating_quality
```

Where:
- `r_d(y_think)`: Traditional quality metrics (consistency, completeness, etc.)
- `r_self_rating_quality`: Quality of model's self-evaluation (NEW)

**Key Insight**: Models are rewarded for:
1. Thinking well (traditional intrinsic rewards)
2. **Evaluating their thinking well** (self-rating quality) ⭐

### 5. Training Pipeline

**Phase 1: SFT with Self-Ratings**
- Generate training examples with self-ratings in thinking chain
- Use GPT-4o to create high-quality examples
- Train model to generate self-ratings during thinking

**Phase 2: GRPO with Self-Rating Quality**
- Model generates thinking with self-ratings
- Reward calculator:
  1. Extracts self-ratings
  2. Evaluates actual thinking quality
  3. Calculates self-rating quality (calibration)
  4. Combines into intrinsic reward
- GRPO optimizes for models that both think well AND evaluate themselves well

## Why This is Truly Endogenous

**Before** (External Evaluation):
```
Model → Generates thinking
External Calculator → Evaluates thinking → Reward
```

**After** (Endogenous Evaluation):
```
Model → Generates thinking + Self-evaluates thinking
External Calculator → Evaluates:
  1. Thinking quality (as before)
  2. Self-evaluation quality (NEW) → Reward
```

**Key Difference**: Model actively participates in evaluation by generating self-ratings. The reward includes quality of this self-evaluation, creating incentive for models to develop accurate self-assessment capabilities.

## Alignment with target.md

✅ **Section 94**: "训练模型学会生成包含自我评估步骤的思考链格式"
- Implemented via prompt template and SFT

✅ **Section 112-116**: "模型在思考时，从多个维度分别评分"
- Implemented via self-rating extraction

✅ **Section 208-216**: "奖励模型的自我评估"
- Implemented via self-rating quality evaluation

## Implementation Status

- ✅ Design complete
- ⏳ Implementation pending (Phase 1)
- ⏳ Testing pending (Phase 2)

