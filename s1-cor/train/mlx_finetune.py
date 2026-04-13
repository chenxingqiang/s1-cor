#!/usr/bin/env python3
"""
MLX-Tune SFT Fine-Tuning Script for CoR Models on Mac (Apple Silicon).

Uses mlx-tune (https://github.com/ARahim3/mlx-tune) which provides an
Unsloth-compatible API for LoRA fine-tuning on Apple Silicon via MLX.

Key features:
  - FastLanguageModel API (same as Unsloth — portable to CUDA)
  - SFTTrainer with SFTConfig (same as TRL)
  - LoRA/QLoRA with target module selection
  - Save as HuggingFace format, GGUF, or push to Hub

Usage:
    # Quick start with defaults (Qwen2.5-0.5B, LoRA)
    python train/mlx_finetune.py

    # Custom model and config
    python train/mlx_finetune.py --model_size 1.5B --max_steps 500

    # Full pipeline: prepare data + train + test
    python train/mlx_finetune.py --prepare_data --test_after_training

    # With 4-bit quantization (reduces memory usage)
    python train/mlx_finetune.py --model_size 3B --load_in_4bit
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
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "lora_rank": 16,
        "batch_size": 2,
        "max_seq_length": 4096,
        "lr": 2e-4,
        "max_steps": 200,
        "load_in_4bit": True,
    },
    "1.5B": {
        "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "lora_rank": 16,
        "batch_size": 1,
        "max_seq_length": 4096,
        "lr": 2e-4,
        "max_steps": 200,
        "load_in_4bit": True,
    },
    "3B": {
        "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "lora_rank": 16,
        "batch_size": 1,
        "max_seq_length": 2048,
        "lr": 1e-4,
        "max_steps": 150,
        "load_in_4bit": True,
    },
    "4B": {
        "model": "mlx-community/Qwen3-4B-4bit",
        "lora_rank": 16,
        "batch_size": 1,
        "max_seq_length": 2048,
        "lr": 1e-4,
        "max_steps": 150,
        "load_in_4bit": True,
    },
    "7B": {
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "lora_rank": 8,
        "batch_size": 1,
        "max_seq_length": 2048,
        "lr": 5e-5,
        "max_steps": 100,
        "load_in_4bit": True,
    },
}


def check_mlx_tune_available():
    """Check if mlx-tune is installed."""
    try:
        import mlx_tune  # noqa: F401

        logger.info("mlx-tune is available")
        return True
    except ImportError:
        return False


def install_mlx_tune():
    """Install mlx-tune dependencies."""
    logger.info("Installing mlx-tune...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "mlx-tune"],
    )
    logger.info("mlx-tune installed successfully.")


def prepare_data(args):
    """Prepare training data in chat JSONL format for mlx-tune."""
    logger.info("Preparing training data for mlx-tune...")
    cmd = [
        sys.executable,
        "train/mlx_prepare_data.py",
        "--dataset",
        args.dataset,
        "--output_dir",
        args.data,
        "--format",
        "chat",  # mlx-tune uses chat format (messages JSONL)
    ]
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if args.hf_dataset:
        cmd.extend(["--hf_dataset", args.hf_dataset])

    subprocess.check_call(cmd)
    logger.info("Data preparation complete.")


def load_training_data(data_dir: str):
    """Load training data from JSONL files."""
    train_file = os.path.join(data_dir, "train.jsonl")
    train_data = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_data.append(json.loads(line))
    logger.info(f"Loaded {len(train_data)} training examples from {train_file}")
    return train_data


def run_sft_training(args):
    """Run SFT fine-tuning using mlx-tune's FastLanguageModel + SFTTrainer."""
    from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig

    # Step 1: Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    logger.info("Model loaded successfully")

    # Step 2: Apply LoRA adapters
    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    logger.info(f"LoRA configured: rank={args.lora_rank}, alpha={args.lora_alpha}")

    # Step 3: Load training data
    train_data = load_training_data(args.data)

    # Step 4: Configure training
    training_config = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
    )

    # Step 5: Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        tokenizer=tokenizer,
        args=training_config,
        max_seq_length=args.max_seq_length,
    )

    # Step 6: Train
    logger.info("Starting SFT training...")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Output dir: {args.output_dir}")

    trainer.train()
    logger.info("Training complete!")

    # Step 7: Save model
    logger.info(f"Saving LoRA adapters to {args.output_dir}")
    model.save_pretrained(args.output_dir)

    if args.save_merged:
        merged_path = args.merged_model_path or f"ckpts/mlx_merged_{Path(args.model).name}"
        logger.info(f"Saving merged model to {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer)
        logger.info(f"Merged model saved to: {merged_path}")

    if args.save_gguf:
        gguf_path = args.gguf_model_path or f"ckpts/mlx_gguf_{Path(args.model).name}"
        logger.info(f"Exporting to GGUF: {gguf_path}")
        model.save_pretrained_gguf(gguf_path, tokenizer)
        logger.info(f"GGUF model saved to: {gguf_path}")

    return model, tokenizer


def run_test_generation(model, tokenizer, args):
    """Test the fine-tuned model with sample prompts."""
    from mlx_tune import FastLanguageModel
    from mlx_lm import generate

    test_prompts = [
        "Solve the equation: 3x + 5 = 20. Show your reasoning step by step.",
        "What is the integral of x^2 from 0 to 1?",
        "Find all prime numbers between 20 and 40.",
    ]

    logger.info("=" * 60)
    logger.info("Testing fine-tuned model with sample prompts")
    logger.info("=" * 60)

    FastLanguageModel.for_inference(model)

    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logger.info(f"\nPrompt: {prompt}")
        logger.info("-" * 40)
        try:
            response = generate(
                model.model,
                tokenizer,
                prompt=formatted,
                max_tokens=512,
                verbose=False,
            )
            logger.info(f"Response:\n{response}")
        except Exception as e:
            logger.error(f"Generation error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="mlx-tune SFT Fine-Tuning for CoR Models on Mac",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start (prepare data + train)
  python train/mlx_finetune.py --prepare_data

  # Train with a specific model size
  python train/mlx_finetune.py --model_size 1.5B --prepare_data

  # Full pipeline: prepare + train + merge + test
  python train/mlx_finetune.py --prepare_data --save_merged --test_after_training

  # With custom model
  python train/mlx_finetune.py --model mlx-community/Llama-3.2-1B-Instruct-4bit

  # Export to GGUF for Ollama
  python train/mlx_finetune.py --model_size 0.5B --save_gguf
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
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model name or path (use mlx-community/ models for pre-quantized)",
    )
    model_group.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization (default: True)",
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
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples",
    )

    # LoRA parameters
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora_rank", type=int, default=16,
                            help="LoRA rank (default: 16)")
    lora_group.add_argument("--lora_alpha", type=int, default=16,
                            help="LoRA alpha scaling (default: 16)")

    # Training parameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch_size", type=int, default=2,
                             help="Per-device batch size")
    train_group.add_argument("--grad_accum", type=int, default=4,
                             help="Gradient accumulation steps")
    train_group.add_argument("--max_steps", type=int, default=200,
                             help="Maximum training steps")
    train_group.add_argument("--lr", type=float, default=2e-4,
                             help="Learning rate")
    train_group.add_argument("--max_seq_length", type=int, default=2048,
                             help="Maximum sequence length")
    train_group.add_argument("--warmup_steps", type=int, default=5,
                             help="Number of warmup steps")
    train_group.add_argument("--logging_steps", type=int, default=1,
                             help="Log every N steps")
    train_group.add_argument("--seed", type=int, default=42)

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output_dir",
        type=str,
        default="ckpts/mlx_lora_adapters",
        help="Output directory for LoRA adapters",
    )

    # Post-training
    post_group = parser.add_argument_group("Post-training")
    post_group.add_argument(
        "--save_merged",
        action="store_true",
        help="Save merged model (base + adapters) after training",
    )
    post_group.add_argument(
        "--merged_model_path",
        type=str,
        default=None,
        help="Path for merged model output",
    )
    post_group.add_argument(
        "--save_gguf",
        action="store_true",
        help="Export to GGUF format after training",
    )
    post_group.add_argument(
        "--gguf_model_path",
        type=str,
        default=None,
        help="Path for GGUF model output",
    )
    post_group.add_argument(
        "--test_after_training",
        action="store_true",
        help="Run test generation after training",
    )
    post_group.add_argument(
        "--install_deps",
        action="store_true",
        help="Install mlx-tune before training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("mlx-tune SFT Fine-Tuning for CoR Models")
    logger.info("=" * 60)

    # Apply model size preset
    if args.model_size:
        config = MODEL_CONFIGS[args.model_size]
        args.model = config["model"]
        args.lora_rank = config.get("lora_rank", 16)
        args.batch_size = config["batch_size"]
        args.max_seq_length = config["max_seq_length"]
        args.lr = config["lr"]
        args.max_steps = config["max_steps"]
        args.load_in_4bit = config.get("load_in_4bit", True)
        logger.info(f"Using preset config for {args.model_size}: {config}")

    # Install deps if requested
    if args.install_deps:
        install_mlx_tune()

    # Check mlx-tune availability
    if not check_mlx_tune_available():
        logger.error(
            "mlx-tune is not installed. This script requires Apple Silicon Mac.\n"
            "Install with: pip install mlx-tune\n"
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
    logger.info(f"Model: {args.model}")
    logger.info(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    logger.info(f"Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Max seq length: {args.max_seq_length}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"4-bit quantization: {args.load_in_4bit}")

    # Run training
    model, tokenizer = run_sft_training(args)

    # Test if requested
    if args.test_after_training:
        run_test_generation(model, tokenizer, args)

    logger.info("=" * 60)
    logger.info("Done! Summary:")
    logger.info(f"  Base model: {args.model}")
    logger.info(f"  LoRA adapters: {args.output_dir}")
    logger.info("")
    logger.info("To run inference with the fine-tuned model:")
    logger.info(f"  python train/mlx_inference.py --model {args.model} "
                f"--adapter_path {args.output_dir} --interactive")
    logger.info("")
    logger.info("To run GRPO training with CoR rewards:")
    logger.info(f"  python train/mlx_grpo.py --model {args.model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
