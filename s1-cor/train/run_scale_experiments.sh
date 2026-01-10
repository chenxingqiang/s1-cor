#!/bin/bash
# Multi-Scale CoR Validation Experiments
#
# This script runs CoR+GRPO training on different model scales
# to validate the effectiveness of endogenous self-evaluation.
#
# Qwen2.5 Model Sizes:
#   - 0.5B: Quick experiments, ~1GB VRAM
#   - 1.5B: Small scale, ~3GB VRAM  
#   - 3B:   Medium scale, ~6GB VRAM
#   - 7B:   Large scale, ~14GB VRAM
#   - 14B:  Very large, ~28GB VRAM
#   - 32B:  Full scale (paper setting)
#
# Usage:
#   bash train/run_scale_experiments.sh [model_size] [dataset]
#   
# Examples:
#   bash train/run_scale_experiments.sh 0.5B           # Quick test
#   bash train/run_scale_experiments.sh 1.5B deepseek  # Use DeepSeek dataset
#   bash train/run_scale_experiments.sh all            # Run all scales

set -e

# Configuration
MODEL_SIZE=${1:-"0.5B"}  # Default to smallest
DATASET=${2:-"full"}     # full or deepseek
EXPERIMENT_ID="$(date +%Y%m%d_%H%M%S)"

# Dataset path
if [ "$DATASET" == "deepseek" ]; then
    TRAIN_DATA="local_data/s1K_cor_deepseek"
else
    TRAIN_DATA="local_data/s1K_cor_full"
fi

# Model configurations by size
declare -A MODELS=(
    ["0.5B"]="Qwen/Qwen2.5-0.5B-Instruct"
    ["1.5B"]="Qwen/Qwen2.5-1.5B-Instruct"
    ["3B"]="Qwen/Qwen2.5-3B-Instruct"
    ["7B"]="Qwen/Qwen2.5-7B-Instruct"
    ["14B"]="Qwen/Qwen2.5-14B-Instruct"
    ["32B"]="Qwen/Qwen2.5-32B-Instruct"
)

# Hyperparameters by size (adjusted for memory)
declare -A BATCH_SIZES=(
    ["0.5B"]="4"
    ["1.5B"]="2"
    ["3B"]="2"
    ["7B"]="1"
    ["14B"]="1"
    ["32B"]="1"
)

declare -A GRAD_ACCUM=(
    ["0.5B"]="4"
    ["1.5B"]="8"
    ["3B"]="8"
    ["7B"]="16"
    ["14B"]="16"
    ["32B"]="4"
)

declare -A BLOCK_SIZES=(
    ["0.5B"]="4096"
    ["1.5B"]="8192"
    ["3B"]="8192"
    ["7B"]="16384"
    ["14B"]="16384"
    ["32B"]="32768"
)

declare -A NUM_GENS=(
    ["0.5B"]="4"
    ["1.5B"]="4"
    ["3B"]="4"
    ["7B"]="8"
    ["14B"]="8"
    ["32B"]="8"
)

# Learning rates (smaller models can use larger LR)
declare -A LEARNING_RATES=(
    ["0.5B"]="5e-6"
    ["1.5B"]="3e-6"
    ["3B"]="2e-6"
    ["7B"]="1e-6"
    ["14B"]="5e-7"
    ["32B"]="1e-6"
)

run_experiment() {
    local size=$1
    local model=${MODELS[$size]}
    local batch=${BATCH_SIZES[$size]}
    local grad_acc=${GRAD_ACCUM[$size]}
    local block=${BLOCK_SIZES[$size]}
    local num_gen=${NUM_GENS[$size]}
    local lr=${LEARNING_RATES[$size]}
    
    echo "=============================================="
    echo "Running CoR experiment: Qwen2.5-${size}"
    echo "=============================================="
    echo "Model: ${model}"
    echo "Dataset: ${TRAIN_DATA}"
    echo "Batch size: ${batch}"
    echo "Gradient accumulation: ${grad_acc}"
    echo "Block size: ${block}"
    echo "Num generations (N): ${num_gen}"
    echo "Learning rate: ${lr}"
    echo "=============================================="
    
    OUTPUT_DIR="ckpts/cor-${size}-${EXPERIMENT_ID}"
    
    # Get GPU count
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
    
    # For small models, we might not need distributed training
    if [ "$size" == "0.5B" ] || [ "$size" == "1.5B" ]; then
        # Single GPU for small models
        python train/grpo.py \
            --model_name="${model}" \
            --ref_model_name="${model}" \
            --train_file_path="${TRAIN_DATA}" \
            --block_size=${block} \
            --num_generations=${num_gen} \
            --lambda_intrinsic=1.0 \
            --self_rating_weight=0.2 \
            --per_device_train_batch_size=${batch} \
            --gradient_accumulation_steps=${grad_acc} \
            --num_train_epochs=3 \
            --learning_rate=${lr} \
            --warmup_ratio=0.1 \
            --epsilon=0.2 \
            --beta=0.01 \
            --bf16=True \
            --logging_steps=10 \
            --save_strategy="epoch" \
            --output_dir="${OUTPUT_DIR}" \
            --report_to="wandb" \
            --wandb_project="cor-scale-${size}"
    else
        # Distributed training for larger models
        torchrun --nproc-per-node ${GPU_COUNT} --master_port 12346 \
            train/grpo.py \
            --model_name="${model}" \
            --ref_model_name="${model}" \
            --train_file_path="${TRAIN_DATA}" \
            --block_size=${block} \
            --num_generations=${num_gen} \
            --lambda_intrinsic=1.0 \
            --self_rating_weight=0.2 \
            --per_device_train_batch_size=${batch} \
            --gradient_accumulation_steps=${grad_acc} \
            --num_train_epochs=3 \
            --learning_rate=${lr} \
            --warmup_ratio=0.1 \
            --epsilon=0.2 \
            --beta=0.01 \
            --fsdp="full_shard auto_wrap" \
            --fsdp_config="train/fsdp_config_qwen.json" \
            --bf16=True \
            --logging_steps=10 \
            --save_strategy="epoch" \
            --output_dir="${OUTPUT_DIR}" \
            --report_to="wandb" \
            --wandb_project="cor-scale-${size}"
    fi
    
    echo "Experiment ${size} complete! Saved to ${OUTPUT_DIR}"
    echo ""
}

# Main execution
echo "========================================"
echo "CoR Multi-Scale Validation Experiments"
echo "========================================"
echo "Experiment ID: ${EXPERIMENT_ID}"
echo "Dataset: ${TRAIN_DATA}"
echo ""

if [ "$MODEL_SIZE" == "all" ]; then
    # Run all scales sequentially (smallest to largest)
    for size in "0.5B" "1.5B" "3B" "7B"; do
        run_experiment "$size"
    done
elif [ -n "${MODELS[$MODEL_SIZE]}" ]; then
    run_experiment "$MODEL_SIZE"
else
    echo "Error: Unknown model size '${MODEL_SIZE}'"
    echo "Available sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B, all"
    exit 1
fi

echo "========================================"
echo "All experiments complete!"
echo "========================================"
