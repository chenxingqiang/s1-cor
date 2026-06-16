#!/bin/bash
# CoR + GRPO on Qwen2.5-0.5B — minimal open-scale theory validation.
#
# ~1GB VRAM, single GPU. For Colab SFT first: bash train/colab_minimal.sh sft
#
# Usage:
#   export WANDB_DISABLED=true
#   bash train/grpo_05b.sh
#   bash train/grpo_05b.sh deepseek    # smaller local dataset
#
# Full scale runner (same hyperparams): bash train/run_scale_experiments.sh 0.5B

set -euo pipefail

DATASET="${1:-full}"
EXPERIMENT_ID="$(date +%Y%m%d_%H%M%S)"

if [ "$DATASET" = "deepseek" ]; then
  TRAIN_DATA="local_data/s1K_cor_deepseek"
else
  TRAIN_DATA="local_data/s1K_cor_full"
fi

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
REF_MODEL="${REF_MODEL:-ckpts/sft-0.5B-colab}"
if [ ! -d "$REF_MODEL" ] && [ ! -f "$REF_MODEL/config.json" ]; then
  echo "REF_MODEL not found at ${REF_MODEL}; using base instruct weights as reference."
  REF_MODEL="${MODEL}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-ckpts/cor-0.5B-${EXPERIMENT_ID}}"

USE_MATH="${USE_MATH_GRADER:-0}"
math_flag=""
if [ "$USE_MATH" = "1" ] || [ "$USE_MATH" = "true" ]; then
  math_flag="--use_math_grader=True"
fi

echo "CoR GRPO 0.5B"
echo "  model:      ${MODEL}"
echo "  ref_model:  ${REF_MODEL}"
echo "  data:       ${TRAIN_DATA}"
echo "  output:     ${OUTPUT_DIR}"
echo "  CoR:        λ=1.0 μ=0.5 ν=0.1 K=3"
echo ""

REPORT_TO="${REPORT_TO:-wandb}"
if [ "${WANDB_DISABLED:-false}" = "true" ] || [ "${WANDB_DISABLED:-false}" = "1" ]; then
  REPORT_TO="none"
fi

python train/grpo.py \
  --model_name="${MODEL}" \
  --ref_model_name="${REF_MODEL}" \
  --train_file_path="${TRAIN_DATA}" \
  --block_size=4096 \
  --num_generations=8 \
  --lambda_intrinsic=1.0 \
  --self_rating_weight=0.2 \
  --improvement_weight=0.5 \
  --convergence_weight=0.1 \
  --max_reflection_rounds=3 \
  --enable_reflection=True \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=3 \
  --learning_rate=5e-6 \
  --warmup_ratio=0.1 \
  --epsilon=0.2 \
  --beta=0.01 \
  --bf16=True \
  --logging_steps=10 \
  --save_strategy="epoch" \
  --output_dir="${OUTPUT_DIR}" \
  --report_to="${REPORT_TO}" \
  --wandb_project="cor-0.5B" \
  ${math_flag}

echo "Done. Checkpoint: ${OUTPUT_DIR}"
echo "Eval (GPU): see eval/commands.sh cor-0.5B line with pretrained=${OUTPUT_DIR}"
