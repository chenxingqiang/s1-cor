# Chain of Reward (CoR) Training

This directory contains the training scripts for s1 models, including the new **Chain of Reward (CoR)** framework.

## Overview

CoR extends s1's training with:
1. **Multi-dimensional intrinsic rewards** - evaluates consistency, completeness, clarity, and format
2. **Endogenous self-evaluation** - model generates self-ratings during thinking, and we reward accurate self-assessment
3. **GRPO optimization** - Group Relative Policy Optimization for efficient policy learning

## Files

| File | Description |
|------|-------------|
| `sft.py` | Original s1 SFT training script |
| `sft.sh` | Shell script for SFT training |
| `grpo.py` | **NEW**: GRPO training with CoR rewards |
| `grpo.sh` | **NEW**: Shell script for GRPO training |
| `run_cor_pipeline.sh` | **NEW**: Complete CoR training pipeline |
| `rewards/` | **NEW**: Reward calculation module |

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure TRL >= 0.14.0 for GRPO support
pip install trl>=0.14.0
```

### Option 1: Full Pipeline

Run the complete CoR training pipeline:

```bash
bash train/run_cor_pipeline.sh
```

This will:
1. Prepare data with self-ratings
2. Run SFT (creates reference model)
3. Run GRPO with CoR rewards
4. Evaluate on AIME24, MATH500, GPQA

### Option 2: Step by Step

#### Step 1: Prepare Data with Self-Ratings

```bash
python data/add_self_ratings.py \
    --input_path simplescaling/s1K_tokenized \
    --output_path local_data/s1K_rated \
    --method rule
```

#### Step 2: SFT Training

```bash
bash train/sft.sh
```

#### Step 3: GRPO Training

```bash
# Edit grpo.sh to set ref_model_path to your SFT checkpoint
bash train/grpo.sh
```

## Reward Module (`rewards/`)

The `rewards/` directory contains the CoR reward calculation:

### Components

1. **`calculator.py`** - Main `RewardCalculator` class
   - Combines external (correctness) and intrinsic (thinking quality) rewards
   - Formula: `R(c) = R_ext(c) + λ * R_int(c)`

2. **`self_rating.py`** - Endogenous self-evaluation
   - `SelfRatingExtractor`: Extracts self-ratings from thinking chains
   - `SelfRatingEvaluator`: Evaluates calibration of self-ratings

3. **`intrinsic.py`** - Multi-dimensional intrinsic rewards
   - `ConsistencyReward`: Logical coherence
   - `CompletenessReward`: Step comprehensiveness
   - `ClarityReward`: Reasoning clarity
   - `FormatReward`: Structural correctness

### Usage

```python
from train.rewards import RewardCalculator, RewardConfig

# Create calculator with custom config
config = RewardConfig(
    lambda_intrinsic=1.0,  # Weight for intrinsic rewards
    self_rating_weight=0.2,  # Weight for self-rating quality
)
calculator = RewardCalculator(config)

# Calculate reward
output = calculator.calculate_total_reward(
    thinking_chain="Step 1: ... [Self-Rating: Consistency=8/10]",
    answer="42",
    ground_truth="42"
)

print(f"Total reward: {output.total_reward}")
print(f"External: {output.external_reward}")
print(f"Intrinsic: {output.intrinsic_reward}")
```

### Self-Rating Format

During thinking, models should generate self-ratings like:

```
Step 1: Analyze the problem...
[Self-Rating: Consistency=8/10, Completeness=9/10, Accuracy=7/10, Clarity=8/10]

Step 2: Apply the formula...
[Self-Rating: Consistency=9/10, Completeness=8/10, Accuracy=9/10, Clarity=9/10]

[Overall Quality: 8.5/10]
```

## Testing

Run unit tests:

```bash
python -m pytest train/rewards/test_rewards.py -v
```

## Hyperparameters

### CoR Configuration (per theory.md and design.md)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `lambda_intrinsic` | λ | 1.0 | Weight for intrinsic rewards |
| `improvement_weight` | μ | 0.5 | Weight for improvement reward (NEW) |
| `convergence_weight` | ν | 0.1 | Weight for convergence reward (NEW) |
| `max_reflection_rounds` | K | 3 | Max reflection iterations (NEW) |
| `self_rating_weight` | w_self | 0.2 | Weight for self-rating quality reward |
| `calibration_bonus` | α | 0.2 | Bonus for high-high calibration alignment |

### GRPO Configuration

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `num_generations` | N | 8 | Candidates per input |
| `epsilon` | ε | 0.2 | Clipping range |
| `beta` | β | 0.01 | KL penalty coefficient |
| `learning_rate` | - | 1e-6 | Learning rate |

### Extended Reward Formula

```
R(c) = R_ext + λ·R_int + μ·R_improve + ν·R_converge

Where:
- R_ext: Binary correctness (0 or 1)
- R_int: Multi-dimensional quality + self-rating calibration
- R_improve: Quality improvement across reflection rounds
- R_converge: Convergence stability reward
```

## Theory

See the paper and `THEORY.md` for full mathematical derivation:

- **Total Reward**: `R(c) = R_ext(c) + λ * R_int(c)`
- **Intrinsic Reward**: `R_int(c) = Σ_d w_d * r_d(y_think) + w_self * r_self_rating_quality`
- **Calibration**: `cal_d(u, v) = 1 - |u - v|`
- **GRPO Objective**: Uses clipped surrogate with KL penalty

## MLX Fine-Tuning on Mac (Apple Silicon)

Train and verify CoR models directly on your Mac using [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms).

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 16GB+ unified memory recommended (32GB+ for 3B/7B models)

### Quick Start

```bash
# 1. Install MLX
pip install mlx-lm>=0.21.0

# 2. Prepare data + train (one command)
python train/mlx_finetune.py --prepare_data --model_size 0.5B

# Or use the shell script
bash train/mlx_finetune.sh 0.5B deepseek
```

### Step-by-Step

#### Step 1: Prepare Data

Convert CoR datasets to JSONL format for MLX:

```bash
python train/mlx_prepare_data.py --dataset deepseek --output_dir train/mlx_data
```

Options:
- `--dataset deepseek` - Use DeepSeek-generated CoR data (default)
- `--dataset full` - Use full CoR dataset
- `--dataset hf --hf_dataset xingqiang/s1K-cor-deepseek` - Load from HuggingFace Hub

#### Step 2: LoRA Fine-Tuning

```bash
# Quick start with presets
python train/mlx_finetune.py --model_size 0.5B --data train/mlx_data

# Custom configuration
python train/mlx_finetune.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --lora_layers 16 \
    --batch_size 1 \
    --iters 500 \
    --lr 5e-6

# Use YAML config
python train/mlx_finetune.py --config train/mlx_lora_config.yaml
```

Available model presets: `0.5B`, `1.5B`, `3B`, `4B` (Qwen3), `7B`

#### Step 3: Test & Evaluate

```bash
# Single prompt
python train/mlx_inference.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter_path ckpts/mlx_lora_adapters \
    --prompt "Solve: 2x + 3 = 7"

# Interactive mode
python train/mlx_inference.py --interactive

# CoR evaluation (checks for self-rating markers)
python train/mlx_inference.py --eval_cor --save_results results.json
```

#### Step 4 (Optional): Fuse Adapters

Merge LoRA adapters into the base model for faster inference:

```bash
python train/mlx_finetune.py \
    --model_size 0.5B \
    --fuse \
    --fused_model_path ckpts/mlx_fused_model
```

### MLX Files

| File | Description |
|------|-------------|
| `train/mlx_prepare_data.py` | Convert CoR datasets to MLX JSONL format |
| `train/mlx_finetune.py` | Main MLX LoRA fine-tuning script |
| `train/mlx_finetune.sh` | Shell script for quick fine-tuning |
| `train/mlx_inference.py` | Inference and evaluation for fine-tuned models |
| `train/mlx_lora_config.yaml` | Default LoRA configuration |

### Memory Requirements

| Model | Min Memory | Recommended |
|-------|-----------|-------------|
| 0.5B  | 8GB       | 16GB        |
| 1.5B  | 16GB      | 16GB        |
| 3B    | 16GB      | 32GB        |
| 4B    | 16GB      | 32GB        |
| 7B    | 32GB      | 64GB        |

## Citation

```bibtex
@article{cor2025,
  title={CoR: Chain of Reward with Endogenous Self-Evaluation for Reasoning},
  author={Chen, Xingqiang},
  year={2025}
}
```
