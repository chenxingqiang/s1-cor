#!/bin/bash
# =============================================================================
# S1-CoR 14B Model GRPO Training Script for A100 GPUs
# =============================================================================
#
# Description: GRPO (Generative Reinforcement Learning with Policy Optimization)
#              training for Chain of Reward (CoR) framework
# 
# Prerequisites:
#   1. Complete SFT training first (sft_14b_a100.sh)
#   2. SFT model saved at: /data/sa_data/data/agents/models/s1-cor-14b
#
# Hardware: 3x A100 80GB GPUs
#
# Usage:
#   bash train/grpo_14b_a100.sh
#   nohup bash train/grpo_14b_a100.sh > train_grpo_14b.log 2>&1 &
#
# =============================================================================

# Configuration
export CUDA_VISIBLE_DEVICES=4,5,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model paths - use SFT trained model
SFT_MODEL="/data/sa_data/data/agents/models/s1-cor-14b/checkpoint-123"
REF_MODEL="/data/sa_data/data/agents/models/Qwen2.5-14B-Instruct"  # Original model for KL penalty
OUTPUT_DIR="/data/sa_data/data/agents/models/s1-cor-14b-grpo"
DATASET="xingqiang/s1K-cor-deepseek"

# Training hyperparameters
BLOCK_SIZE=4096
BATCH_SIZE=1
GRAD_ACCUM=4
EPOCHS=1
LR=1e-6

# CoR reward configuration
LAMBDA_INTRINSIC=1.0
IMPROVEMENT_WEIGHT=0.5
CONVERGENCE_WEIGHT=0.1

# Distributed training
NPROC=3
MASTER_PORT=29503

# Run GRPO training
python -m torch.distributed.run \
    --nproc-per-node=$NPROC \
    --master_port=$MASTER_PORT \
    train/grpo.py \
    --model_name=$SFT_MODEL \
    --ref_model_name=$REF_MODEL \
    --train_file_path=$DATASET \
    --block_size=$BLOCK_SIZE \
    --lambda_intrinsic=$LAMBDA_INTRINSIC \
    --improvement_weight=$IMPROVEMENT_WEIGHT \
    --convergence_weight=$CONVERGENCE_WEIGHT \
    --enable_reflection=True \
    --wandb_project=cor-14b-grpo \
    --output_dir=$OUTPUT_DIR \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --num_train_epochs=$EPOCHS \
    --learning_rate=$LR \
    --warmup_ratio=0.1 \
    --bf16=True \
    --logging_steps=1 \
    --save_strategy=steps \
    --save_steps=50 \
    --save_only_model=True \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config=train/fsdp_config_qwen.json \
    --optim=adafactor \
    --push_to_hub=True \
    --hub_model_id=xingqiang/s1-cor-14b-grpo

echo "GRPO Training completed! Model saved to: $OUTPUT_DIR"
