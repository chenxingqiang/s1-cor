#!/bin/bash
# =============================================================================
# MLX Fine-Tuning Script for CoR Models on Mac (Apple Silicon)
#
# This script runs the full MLX fine-tuning pipeline:
#   1. Data preparation (convert CoR dataset to JSONL)
#   2. LoRA fine-tuning with mlx-lm
#   3. Optional: fuse adapters and test
#
# Requirements:
#   - macOS with Apple Silicon (M1/M2/M3/M4)
#   - Python 3.10+
#   - pip install mlx-lm>=0.21.0
#
# Usage:
#   bash train/mlx_finetune.sh                    # Default: 0.5B model
#   bash train/mlx_finetune.sh 1.5B               # Specify model size
#   bash train/mlx_finetune.sh 0.5B deepseek      # Specify model + dataset
#   bash train/mlx_finetune.sh 4B hf              # Qwen3-4B with HF data
# =============================================================================

set -euo pipefail

# Parse arguments
MODEL_SIZE="${1:-0.5B}"
DATASET="${2:-deepseek}"
EXTRA_ARGS="${3:-}"

echo "============================================================"
echo "MLX LoRA Fine-Tuning for CoR - Qwen ${MODEL_SIZE}"
echo "============================================================"
echo "Model Size: ${MODEL_SIZE}"
echo "Dataset: ${DATASET}"
echo "Date: $(date)"
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "WARNING: This script is designed for macOS with Apple Silicon."
    echo "You may encounter issues on other platforms."
fi

# Check Python and MLX
echo "Checking dependencies..."
python3 -c "import mlx; print(f'MLX version: {mlx.__version__}')" 2>/dev/null || {
    echo "MLX not found. Installing mlx-lm..."
    pip install mlx-lm>=0.21.0
}

# Step 1: Prepare data
echo ""
echo "Step 1: Preparing training data..."
echo "-----------------------------------"
python3 train/mlx_prepare_data.py \
    --dataset "${DATASET}" \
    --output_dir train/mlx_data \
    --format completions

# Step 2: Run LoRA fine-tuning
echo ""
echo "Step 2: Running LoRA fine-tuning..."
echo "-----------------------------------"
python3 train/mlx_finetune.py \
    --model_size "${MODEL_SIZE}" \
    --data train/mlx_data \
    --data_format completions \
    ${EXTRA_ARGS}

# Step 3: Test generation (optional)
echo ""
echo "Step 3: Testing fine-tuned model..."
echo "-----------------------------------"

# Get model name for the size
case "${MODEL_SIZE}" in
    "0.5B") MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" ;;
    "1.5B") MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" ;;
    "3B")   MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" ;;
    "4B")   MODEL_NAME="Qwen/Qwen3-4B" ;;
    "7B")   MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" ;;
    *)      MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" ;;
esac

echo "Generating test output with adapter..."
python3 -m mlx_lm.generate \
    --model "${MODEL_NAME}" \
    --adapter-path ckpts/mlx_lora_adapters \
    --max-tokens 512 \
    --prompt "Solve step by step: What is the sum of the first 10 prime numbers?" \
    || echo "Test generation skipped (model may not be downloaded yet)"

echo ""
echo "============================================================"
echo "Fine-tuning complete!"
echo ""
echo "Adapter saved to: ckpts/mlx_lora_adapters/"
echo ""
echo "To generate with the fine-tuned model:"
echo "  python3 -m mlx_lm.generate \\"
echo "    --model ${MODEL_NAME} \\"
echo "    --adapter-path ckpts/mlx_lora_adapters \\"
echo "    --max-tokens 512 \\"
echo "    --prompt 'Your prompt here'"
echo ""
echo "To fuse adapters into base model:"
echo "  python3 -m mlx_lm.fuse \\"
echo "    --model ${MODEL_NAME} \\"
echo "    --adapter-path ckpts/mlx_lora_adapters \\"
echo "    --save-path ckpts/mlx_fused_model"
echo ""
echo "To run CoR evaluation:"
echo "  python3 train/mlx_inference.py --eval_cor \\"
echo "    --model ${MODEL_NAME} \\"
echo "    --adapter_path ckpts/mlx_lora_adapters"
echo "============================================================"
