#!/bin/bash
# =============================================================================
# MLX-Tune Fine-Tuning Script for CoR Models on Mac (Apple Silicon)
#
# Uses mlx-tune (https://github.com/ARahim3/mlx-tune) which provides an
# Unsloth-compatible API for fine-tuning on Apple Silicon.
#
# This script runs the full pipeline:
#   1. Data preparation (convert CoR dataset to JSONL)
#   2. SFT fine-tuning with mlx-tune
#   3. Optional: GRPO training with CoR rewards
#   4. Test generation
#
# Requirements:
#   - macOS with Apple Silicon (M1/M2/M3/M4/M5)
#   - Python 3.10+
#   - pip install mlx-tune
#
# Usage:
#   bash train/mlx_finetune.sh                    # Default: 0.5B SFT
#   bash train/mlx_finetune.sh 1.5B               # 1.5B model
#   bash train/mlx_finetune.sh 0.5B deepseek      # Dataset choice
#   bash train/mlx_finetune.sh 0.5B deepseek grpo # SFT then GRPO
# =============================================================================

set -euo pipefail

# Parse arguments
MODEL_SIZE="${1:-0.5B}"
DATASET="${2:-deepseek}"
MODE="${3:-sft}"  # sft, grpo, or both

echo "============================================================"
echo "mlx-tune Fine-Tuning for CoR - Qwen ${MODEL_SIZE}"
echo "============================================================"
echo "Model Size: ${MODEL_SIZE}"
echo "Dataset: ${DATASET}"
echo "Mode: ${MODE}"
echo "Date: $(date)"
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "WARNING: This script is designed for macOS with Apple Silicon."
    echo "You may encounter issues on other platforms."
fi

# Check Python and mlx-tune
echo "Checking dependencies..."
python3 -c "import mlx_tune; print('mlx-tune is available')" 2>/dev/null || {
    echo "mlx-tune not found. Installing..."
    pip install mlx-tune
}

# Step 1: Prepare data
echo ""
echo "Step 1: Preparing training data..."
echo "-----------------------------------"
python3 train/mlx_prepare_data.py \
    --dataset "${DATASET}" \
    --output_dir train/mlx_data \
    --format chat

# Step 2: SFT Fine-Tuning
if [[ "${MODE}" == "sft" || "${MODE}" == "both" ]]; then
    echo ""
    echo "Step 2: Running SFT fine-tuning with mlx-tune..."
    echo "-----------------------------------"
    python3 train/mlx_finetune.py \
        --model_size "${MODEL_SIZE}" \
        --data train/mlx_data
fi

# Step 3: GRPO Training (optional)
if [[ "${MODE}" == "grpo" || "${MODE}" == "both" ]]; then
    echo ""
    echo "Step 3: Running GRPO training with CoR rewards..."
    echo "-----------------------------------"
    python3 train/mlx_grpo.py \
        --model_size "${MODEL_SIZE}" \
        --data train/mlx_data
fi

# Step 4: Test generation
echo ""
echo "Step 4: Testing fine-tuned model..."
echo "-----------------------------------"

ADAPTER_DIR="ckpts/mlx_lora_adapters"
if [[ "${MODE}" == "grpo" ]]; then
    ADAPTER_DIR="ckpts/mlx_grpo_adapters"
fi

# Get model name for the size
case "${MODEL_SIZE}" in
    "0.5B") MODEL_NAME="mlx-community/Qwen2.5-0.5B-Instruct-4bit" ;;
    "1.5B") MODEL_NAME="mlx-community/Qwen2.5-1.5B-Instruct-4bit" ;;
    "3B")   MODEL_NAME="mlx-community/Qwen2.5-3B-Instruct-4bit" ;;
    "4B")   MODEL_NAME="mlx-community/Qwen3-4B-4bit" ;;
    "7B")   MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit" ;;
    *)      MODEL_NAME="mlx-community/Qwen2.5-0.5B-Instruct-4bit" ;;
esac

echo "Running CoR evaluation..."
python3 train/mlx_inference.py \
    --model "${MODEL_NAME}" \
    --adapter_path "${ADAPTER_DIR}" \
    --eval_cor \
    || echo "Evaluation skipped (model may not be downloaded yet)"

echo ""
echo "============================================================"
echo "Fine-tuning complete!"
echo ""
echo "Adapters saved to: ${ADAPTER_DIR}/"
echo ""
echo "Usage examples:"
echo ""
echo "  # Interactive chat"
echo "  python3 train/mlx_inference.py \\"
echo "    --model ${MODEL_NAME} \\"
echo "    --adapter_path ${ADAPTER_DIR} \\"
echo "    --interactive"
echo ""
echo "  # Single prompt"
echo "  python3 train/mlx_inference.py \\"
echo "    --model ${MODEL_NAME} \\"
echo "    --adapter_path ${ADAPTER_DIR} \\"
echo "    --prompt 'Solve: 2x + 3 = 7'"
echo ""
echo "  # Run GRPO with CoR rewards"
echo "  python3 train/mlx_grpo.py \\"
echo "    --model_size ${MODEL_SIZE} \\"
echo "    --data train/mlx_data"
echo ""
echo "  # Save merged model"
echo "  python3 train/mlx_finetune.py \\"
echo "    --model_size ${MODEL_SIZE} \\"
echo "    --save_merged"
echo "============================================================"
