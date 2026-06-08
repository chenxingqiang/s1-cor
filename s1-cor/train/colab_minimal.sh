#!/usr/bin/env bash
# Minimal CoR training on Google Colab (T4 GPU).
# Usage in Colab:
#   !bash train/colab_minimal.sh sft
#   !bash train/colab_minimal.sh verify

set -euo pipefail

MODE="${1:-sft}"

export WANDB_DISABLED=true

install_deps() {
  # Colab ships gcsfs 2025.x which requires fsspec==2025.3.0.
  # datasets 3.1.0 pulls an older fsspec; reinstall at the end to silence conflicts.
  pip install -q transformers==4.46.1 datasets==3.1.0 accelerate==1.0.1 "trl>=0.14.0"
  pip install -q "fsspec==2025.3.0"
}

case "$MODE" in
  install)
    install_deps
    ;;
  verify)
    install_deps
    python -m pytest train/rewards/test_rewards.py train/test_grpo.py -q
    python train/validate_cor_logic.py --dataset deepseek --samples 3
    ;;
  sft)
    install_deps
    python train/sft_small.py \
      --model_size 0.5B \
      --dataset deepseek \
      --epochs 1 \
      --output_dir ckpts/sft-0.5B-colab
    ;;
  *)
    echo "Usage: bash train/colab_minimal.sh [install|verify|sft]"
    exit 1
    ;;
esac
