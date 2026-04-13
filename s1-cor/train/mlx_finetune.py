#!/usr/bin/env python3
"""
MLX Fine-Tuning Script for CoR Models on Mac (Apple Silicon).

Uses mlx-lm's LoRA/QLoRA to fine-tune language models on CoR data,
enabling fast on-device training and verification on M1/M2/M3/M4 Macs.

Usage:
    # Quick start with defaults (Qwen2.5-0.5B, LoRA)
    python train/mlx_finetune.py

    # Custom model and config
    python train/mlx_finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --iters 500

    # Use YAML config
    python train/mlx_finetune.py --config train/mlx_lora_config.yaml

    # Full pipeline: prepare data + train + test
    python train/mlx_finetune.py --prepare_data --test_after_training
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default model configurations for different sizes
MODEL_CONFIGS = {
    "0.5B": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora_layers": 16,
        "lora_rank": 8,
        "batch_size": 2,
        "max_seq_length": 4096,
        "lr": 1e-5,
        "iters": 1000,
    },
    "1.5B": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_layers": 16,
        "lora_rank": 8,
        "batch_size": 1,
        "max_seq_length": 4096,
        "lr": 5e-6,
        "iters": 800,
    },
    "3B": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "lora_layers": 16,
        "lora_rank": 8,
        "batch_size": 1,
        "max_seq_length": 2048,
        "lr": 2e-6,
        "iters": 600,
    },
    "4B": {
        "model": "Qwen/Qwen3-4B",
        "lora_layers": 16,
        "lora_rank": 8,
        "batch_size": 1,
        "max_seq_length": 2048,
        "lr": 2e-6,
        "iters": 600,
    },
    "7B": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "lora_layers": 8,
        "lora_rank": 4,
        "batch_size": 1,
        "max_seq_length": 2048,
        "lr": 1e-6,
        "iters": 500,
    },
}


def check_mlx_available():
    """Check if MLX and mlx-lm are installed."""
    try:
        import mlx  # noqa: F401
        import mlx_lm  # noqa: F401

        logger.info(f"MLX version: {mlx.__version__}")
        return True
    except ImportError:
        return False


def install_mlx():
    """Install MLX dependencies."""
    logger.info("Installing MLX dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "mlx-lm>=0.21.0"],
    )
    logger.info("MLX dependencies installed successfully.")


def prepare_data(args):
    """Prepare training data in MLX format."""
    logger.info("Preparing training data for MLX...")
    cmd = [
        sys.executable,
        "train/mlx_prepare_data.py",
        "--dataset",
        args.dataset,
        "--output_dir",
        args.data,
        "--format",
        args.data_format,
    ]
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if args.hf_dataset:
        cmd.extend(["--hf_dataset", args.hf_dataset])

    subprocess.check_call(cmd)
    logger.info("Data preparation complete.")


def run_lora_training(args):
    """Run LoRA fine-tuning using mlx-lm."""
    # Build the mlx_lm.lora command
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        args.model,
        "--data",
        args.data,
        "--train",
        "--batch-size",
        str(args.batch_size),
        "--lora-layers",
        str(args.lora_layers),
        "--iters",
        str(args.iters),
        "--learning-rate",
        str(args.lr),
        "--steps-per-report",
        str(args.steps_per_report),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--val-batches",
        str(args.val_batches),
        "--save-every",
        str(args.save_every),
        "--adapter-path",
        args.adapter_path,
        "--max-seq-length",
        str(args.max_seq_length),
        "--seed",
        str(args.seed),
    ]

    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")

    if args.resume_adapter_file:
        cmd.extend(["--resume-adapter-file", args.resume_adapter_file])

    if args.config:
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.lora",
            "--config",
            args.config,
        ]

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    logger.info(f"Training complete! Adapters saved to: {args.adapter_path}")


def run_fuse(args):
    """Fuse LoRA adapters into the base model."""
    fused_path = args.fused_model_path or f"ckpts/mlx_fused_{Path(args.model).name}"

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.fuse",
        "--model",
        args.model,
        "--adapter-path",
        args.adapter_path,
        "--save-path",
        fused_path,
    ]

    logger.info(f"Fusing adapters into model...")
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    logger.info(f"Fused model saved to: {fused_path}")
    return fused_path


def run_test_generation(args, model_path=None):
    """Test the fine-tuned model with sample prompts."""
    model = model_path or args.model

    test_prompts = [
        "Solve the equation: 3x + 5 = 20",
        "What is the integral of x^2 from 0 to 1?",
        "Prove that the square root of 2 is irrational.",
    ]

    logger.info("=" * 60)
    logger.info("Testing fine-tuned model with sample prompts")
    logger.info("=" * 60)

    for prompt in test_prompts:
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.generate",
            "--model",
            model,
            "--max-tokens",
            "512",
            "--prompt",
            prompt,
        ]

        # Add adapter if not using fused model
        if not model_path and os.path.exists(args.adapter_path):
            cmd.extend(["--adapter-path", args.adapter_path])

        logger.info(f"\nPrompt: {prompt}")
        logger.info("-" * 40)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info(f"Response:\n{result.stdout}")
            else:
                logger.error(f"Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("Generation timed out (120s limit)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLX LoRA Fine-Tuning for CoR Models on Mac",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start (prepare data + train)
  python train/mlx_finetune.py --prepare_data

  # Train with a specific model size
  python train/mlx_finetune.py --model_size 1.5B --prepare_data

  # Use YAML config file
  python train/mlx_finetune.py --config train/mlx_lora_config.yaml

  # Full pipeline: prepare + train + fuse + test
  python train/mlx_finetune.py --prepare_data --fuse --test_after_training

  # Resume training from saved adapter
  python train/mlx_finetune.py --resume_adapter_file ckpts/mlx_lora_adapters/adapters.safetensors
        """,
    )

    # Model selection
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model_size",
        type=str,
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Preset model size (sets model and hyperparams automatically)",
    )
    model_group.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name or path",
    )
    model_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (overrides all other args when set)",
    )

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--prepare_data",
        action="store_true",
        help="Prepare data before training",
    )
    data_group.add_argument(
        "--dataset",
        type=str,
        default="deepseek",
        help="Dataset source: deepseek, full, original, or hf",
    )
    data_group.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (when --dataset=hf)",
    )
    data_group.add_argument(
        "--data",
        type=str,
        default="train/mlx_data",
        help="Path to JSONL data directory",
    )
    data_group.add_argument(
        "--data_format",
        type=str,
        default="completions",
        choices=["chat", "completions"],
        help="Data format for MLX",
    )
    data_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples",
    )

    # LoRA parameters
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora_layers", type=int, default=16)
    lora_group.add_argument("--lora_rank", type=int, default=8)

    # Training parameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch_size", type=int, default=1)
    train_group.add_argument("--iters", type=int, default=1000)
    train_group.add_argument("--lr", type=float, default=1e-5)
    train_group.add_argument("--max_seq_length", type=int, default=4096)
    train_group.add_argument("--steps_per_report", type=int, default=10)
    train_group.add_argument("--steps_per_eval", type=int, default=100)
    train_group.add_argument("--val_batches", type=int, default=25)
    train_group.add_argument("--save_every", type=int, default=200)
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument(
        "--grad_checkpoint",
        action="store_true",
        help="Enable gradient checkpointing (saves memory)",
    )
    train_group.add_argument(
        "--resume_adapter_file",
        type=str,
        default=None,
        help="Resume from a saved adapter file",
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--adapter_path",
        type=str,
        default="ckpts/mlx_lora_adapters",
        help="Path to save LoRA adapters",
    )

    # Post-training
    post_group = parser.add_argument_group("Post-training")
    post_group.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse LoRA adapters into base model after training",
    )
    post_group.add_argument(
        "--fused_model_path",
        type=str,
        default=None,
        help="Path for fused model output",
    )
    post_group.add_argument(
        "--test_after_training",
        action="store_true",
        help="Run test generation after training",
    )
    post_group.add_argument(
        "--install_deps",
        action="store_true",
        help="Install MLX dependencies before training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MLX LoRA Fine-Tuning for CoR Models")
    logger.info("=" * 60)

    # Apply model size preset
    if args.model_size:
        config = MODEL_CONFIGS[args.model_size]
        args.model = config["model"]
        args.lora_layers = config["lora_layers"]
        args.lora_rank = config.get("lora_rank", 8)
        args.batch_size = config["batch_size"]
        args.max_seq_length = config["max_seq_length"]
        args.lr = config["lr"]
        args.iters = config["iters"]
        logger.info(f"Using preset config for {args.model_size}: {config}")

    # Install deps if requested
    if args.install_deps:
        install_mlx()

    # Check MLX availability
    if not check_mlx_available():
        logger.error(
            "MLX is not installed. This script requires Apple Silicon Mac.\n"
            "Install with: pip install mlx-lm>=0.21.0\n"
            "Or run with: python train/mlx_finetune.py --install_deps"
        )
        sys.exit(1)

    # Prepare data if requested
    if args.prepare_data:
        prepare_data(args)

    # Check data exists
    train_file = os.path.join(args.data, "train.jsonl")
    if not os.path.exists(train_file):
        logger.error(
            f"Training data not found at {train_file}.\n"
            "Run with --prepare_data to generate it first:\n"
            f"  python train/mlx_finetune.py --prepare_data --dataset {args.dataset}"
        )
        sys.exit(1)

    # Log configuration
    if not args.config:
        logger.info(f"Model: {args.model}")
        logger.info(f"LoRA layers: {args.lora_layers}, rank: {args.lora_rank}")
        logger.info(f"Batch size: {args.batch_size}, LR: {args.lr}")
        logger.info(f"Iterations: {args.iters}")
        logger.info(f"Max seq length: {args.max_seq_length}")
        logger.info(f"Adapter path: {args.adapter_path}")
    else:
        logger.info(f"Using config file: {args.config}")

    # Run training
    run_lora_training(args)

    # Fuse if requested
    fused_path = None
    if args.fuse:
        fused_path = run_fuse(args)

    # Test if requested
    if args.test_after_training:
        run_test_generation(args, model_path=fused_path)

    logger.info("=" * 60)
    logger.info("Done! Summary:")
    logger.info(f"  Base model: {args.model}")
    logger.info(f"  Adapters: {args.adapter_path}")
    if fused_path:
        logger.info(f"  Fused model: {fused_path}")
    logger.info("")
    logger.info("To generate with the fine-tuned model:")
    logger.info(
        f"  python -m mlx_lm.generate --model {args.model} "
        f"--adapter-path {args.adapter_path} "
        '--prompt "Your prompt here"'
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
