# CoR Training Data Pipeline

This directory contains scripts for building and processing training data for the CoR (Chain of Reward) framework.

## Data Generation Scripts

### 1. Rule-Based Self-Rating (`build_cor_dataset.py`)
Fast, rule-based self-rating generation using heuristics.

```bash
python data/build_cor_dataset.py --output_path local_data/s1K_cor_full
```

Features:
- Analyzes thinking quality using pattern matching
- Generates ratings for: Consistency, Completeness, Accuracy, Clarity
- ~600 examples/second processing speed

### 2. LLM-Enhanced Self-Rating (`build_cor_dataset_deepseek.py`)
High-quality self-rating using DeepSeek Chat API.

```bash
# Requires DEEPSEEK_API_KEY in environment
python data/build_cor_dataset_deepseek.py --output_path local_data/s1K_cor_deepseek
```

Features:
- Uses DeepSeek Chat for evaluation
- More accurate quality assessments
- ~8 seconds per example

### 3. Validation (`validate_cor_dataset.py`)
Validate dataset quality and format.

```bash
python data/validate_cor_dataset.py --dataset_path local_data/s1K_cor_full
```

## Dataset Format

### Input Fields
- `question`: Problem statement
- `thinking_trajectories`: List of thinking chains
- `attempt`: Model's answer
- `solution`: Reference solution

### Output Fields (Added by CoR scripts)
- `thinking_rated`: Thinking with embedded self-ratings
- `overall_quality_score`: Float 0-1 overall quality
- `text_cor`: Pre-formatted training text
- `has_self_ratings`: Boolean flag
- `rating_method`: "rule" or "deepseek"

### Self-Rating Format
```
[Self-Rating: Consistency=X/10, Completeness=X/10, Accuracy=X/10, Clarity=X/10]
```

### Training Text Format (`text_cor`)
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<|im_start|>think
{thinking_with_ratings}<|im_end|>
{answer}
```

## Generated Datasets

| Dataset | Method | Examples | Avg Quality |
|---------|--------|----------|-------------|
| `local_data/s1K_cor_full` | Rule-based | 1000 | 0.70 |
| `local_data/s1K_cor_deepseek` | DeepSeek | 1000 | In progress |

## Quality Dimensions

1. **Consistency**: Logical coherence, no contradictions
2. **Completeness**: Coverage of necessary steps
3. **Accuracy**: Mathematical correctness
4. **Clarity**: Readability and organization

## Usage in Training

```python
from train.data_utils import DataConfig, prepare_sft_dataset

config = DataConfig(dataset_path="local_data/s1K_cor_full")
dataset = prepare_sft_dataset(config)
```
