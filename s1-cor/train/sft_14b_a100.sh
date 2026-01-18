#!/bin/bash
# =============================================================================
# S1-CoR 14B Model SFT Training Script for A100 GPUs
# =============================================================================
# 
# Description: Train Qwen2.5-14B-Instruct with Chain of Reward (CoR) framework
# Hardware: 3x A100 80GB GPUs
# Optimizer: Adafactor (memory efficient, no optimizer state saving needed)
# 
# Usage:
#   bash train/sft_14b_a100.sh           # Run in foreground
#   nohup bash train/sft_14b_a100.sh > train_14b.log 2>&1 &  # Run in background
#
# Prerequisites:
#   1. Download model: modelscope download --model Qwen/Qwen2.5-14B-Instruct --local_dir /data/sa_data/data/agents/models/Qwen2.5-14B-Instruct
#   2. Login to HuggingFace: huggingface-cli login
#   3. Check available GPUs: nvidia-smi
#
# =============================================================================

# Configuration
export CUDA_VISIBLE_DEVICES=4,5,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model paths
MODEL_NAME="/data/sa_data/data/agents/models/Qwen2.5-14B-Instruct"
OUTPUT_DIR="/data/sa_data/data/agents/models/s1-cor-14b"
DATASET="xingqiang/s1K-cor-deepseek"

# Training hyperparameters
BLOCK_SIZE=4096
BATCH_SIZE=1
GRAD_ACCUM=8
EPOCHS=3
LR=5e-7
WARMUP_RATIO=0.1

# Distributed training
NPROC=3
MASTER_PORT=29502

# Run training
python -m torch.distributed.run \
    --nproc-per-node=$NPROC \
    --master_port=$MASTER_PORT \
    train/sft.py \
    --model_name=$MODEL_NAME \
    --train_file_path=$DATASET \
    --block_size=$BLOCK_SIZE \
    --wandb_project=cor-14b-sft \
    --output_dir=$OUTPUT_DIR \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --num_train_epochs=$EPOCHS \
    --learning_rate=$LR \
    --warmup_ratio=$WARMUP_RATIO \
    --bf16=True \
    --logging_steps=1 \
    --save_strategy=steps \
    --save_steps=50 \
    --save_only_model=True \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config=train/fsdp_config_qwen.json \
    --optim=adafactor \
    --push_to_hub=True \
    --hub_model_id=xingqiang/s1-cor-14b

echo "Training completed! Model saved to: $OUTPUT_DIR"
