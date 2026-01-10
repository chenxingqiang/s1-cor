#!/bin/bash
# GRPO Training Script for Chain of Reward (CoR)
#
# This script trains a model using GRPO with CoR rewards.
# Based on s1's sft.sh with GRPO-specific modifications.
#
# Usage:
#   bash train/grpo.sh
#
# Prerequisites:
#   - Run SFT first to get reference model: bash train/sft.sh
#   - Adjust ref_model_path to point to SFT checkpoint

uid="$(date +%Y%m%d_%H%M%S)"

# Model configuration
# Use Qwen2.5-32B-Instruct as base (consistent with run_cor_pipeline.sh)
# For GRPO, ref_model should be the SFT checkpoint from previous step
base_model="Qwen/Qwen2.5-32B-Instruct"
ref_model_path="${REF_MODEL:-ckpts/cor-sft}"  # SFT checkpoint as reference policy

# Training hyperparameters
lr=1e-6
epochs=3
micro_batch_size=1
gradient_accumulation_steps=4

# GRPO specific (per paper Section 4.1)
num_generations=8  # N=8 candidates per input (paper setting)
epsilon=0.2        # δ=0.2 clipping range
beta=0.01          # β=0.01 KL penalty coefficient

# CoR specific (per theory.md Section 15)
lambda_intrinsic=1.0      # λ: intrinsic reward weight
self_rating_weight=0.2    # w_self: self-rating quality weight
improvement_weight=0.5    # μ: improvement reward weight (NEW)
convergence_weight=0.1    # ν: convergence reward weight (NEW)
max_reflection_rounds=3   # K: max reflection iterations (NEW)

# Hardware
gpu_count=$(nvidia-smi -L | wc -l)

# Output
output_dir="ckpts/cor-grpo-${uid}"

echo "Starting CoR + GRPO training..."
echo "Base model: ${base_model}"
echo "Reference model: ${ref_model_path}"
echo "Output: ${output_dir}"
echo "GPUs: ${gpu_count}"
echo ""
echo "Note: Run SFT first to create reference model:"
echo "  python train/sft_small.py --model_size 32B --push_to_hub"
echo "  or: bash train/run_cor_pipeline.sh --skip-grpo"
echo ""

torchrun --nproc-per-node ${gpu_count} --master_port 12346 \
    train/grpo.py \
    --model_name=${base_model} \
    --ref_model_name=${ref_model_path} \
    --train_file_path="local_data/s1K_cor_full" \
    --block_size=32768 \
    --num_generations=${num_generations} \
    --lambda_intrinsic=${lambda_intrinsic} \
    --self_rating_weight=${self_rating_weight} \
    --improvement_weight=${improvement_weight} \
    --convergence_weight=${convergence_weight} \
    --max_reflection_rounds=${max_reflection_rounds} \
    --enable_reflection=True \
    --per_device_train_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --learning_rate=${lr} \
    --warmup_ratio=0.1 \
    --epsilon=${epsilon} \
    --beta=${beta} \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen.json" \
    --bf16=True \
    --logging_steps=1 \
    --save_strategy="steps" \
    --save_steps=100 \
    --output_dir=${output_dir} \
    --report_to="wandb" \
    --wandb_project="cor-grpo"

echo "Training complete! Model saved to ${output_dir}"
