#!/usr/bin/env python3
"""
Prepare CoR dataset for mlx-tune fine-tuning.

Converts HuggingFace Arrow datasets (local_data/*) to JSONL chat format
that mlx-tune's SFTTrainer/GRPOTrainer expects (messages format).

See: https://github.com/ARahim3/mlx-tune

Usage:
    python train/mlx_prepare_data.py --dataset deepseek --output_dir train/mlx_data
    python train/mlx_prepare_data.py --dataset full --output_dir train/mlx_data
    python train/mlx_prepare_data.py --dataset hf --hf_dataset xingqiang/s1K-cor-deepseek
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dataset path mapping
DATASET_PATHS = {
    "deepseek": "local_data/s1K_cor_deepseek",
    "full": "local_data/s1K_cor_full",
    "original": "local_data/s1K_cor",
}


def load_dataset_from_source(dataset_name: str, hf_dataset: str = None):
    """Load dataset from local disk or HuggingFace Hub."""
    from datasets import load_dataset

    from data_utils import load_cor_dataset_from_disk

    if dataset_name == "hf":
        if not hf_dataset:
            hf_dataset = "xingqiang/s1K-cor-deepseek"
        logger.info(f"Loading from HuggingFace Hub: {hf_dataset}")
        return load_dataset(hf_dataset, split="train")

    dataset_path = DATASET_PATHS.get(dataset_name, dataset_name)
    logger.info(f"Loading from disk: {dataset_path}")
    return load_cor_dataset_from_disk(dataset_path)


def convert_to_chat_format(example: dict) -> dict:
    """Convert a CoR dataset example to MLX chat format.

    MLX expects JSONL with a 'messages' field containing a list of
    role/content dicts (OpenAI chat format).
    """
    question = example.get("question", "")
    thinking = example.get("thinking_rated", "")
    answer = example.get("attempt", "") or example.get("solution", "")

    # Fall back to thinking_trajectories if no rated thinking
    if not thinking:
        trajectories = example.get("thinking_trajectories", [])
        thinking = trajectories[0] if trajectories else ""

    # Build assistant content with thinking block (Qwen format)
    assistant_content = ""
    if thinking:
        assistant_content = f"<|im_start|>think\n{thinking}<|im_end|>\n{answer}"
    else:
        assistant_content = answer

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
    ]

    return {"messages": messages}


def convert_to_completions_format(example: dict) -> dict:
    """Convert a CoR dataset example to MLX completions format.

    Uses the pre-formatted text_cor field directly as prompt+completion.
    """
    text = example.get("text_cor", "")
    if not text:
        # Build text manually
        return convert_to_chat_format(example)

    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Prepare CoR data for MLX fine-tuning")
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepseek",
        choices=list(DATASET_PATHS.keys()) + ["hf"],
        help="Dataset source to use",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="xingqiang/s1K-cor-deepseek",
        help="HuggingFace dataset name (when --dataset=hf)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train/mlx_data",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="chat",
        choices=["chat", "completions"],
        help="Output format: 'chat' (messages) or 'completions' (raw text)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of data to use for validation (default: 0.05)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset_from_source(args.dataset, args.hf_dataset)
    logger.info(f"Loaded {len(dataset)} examples")

    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        logger.info(f"Truncated to {len(dataset)} examples")

    # Convert format
    converter = (
        convert_to_chat_format
        if args.format == "chat"
        else convert_to_completions_format
    )

    converted = []
    skipped = 0
    for i, example in enumerate(dataset):
        try:
            item = converter(dict(example))
            converted.append(item)
        except Exception as e:
            logger.warning(f"Skipping example {i}: {e}")
            skipped += 1

    logger.info(f"Converted {len(converted)} examples ({skipped} skipped)")

    # Train/val split
    import random

    random.seed(args.seed)
    indices = list(range(len(converted)))
    random.shuffle(indices)

    val_size = max(1, int(len(converted) * args.val_ratio))
    val_indices = set(indices[:val_size])
    train_data = [converted[i] for i in range(len(converted)) if i not in val_indices]
    val_data = [converted[i] for i in val_indices]

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "valid.jsonl")

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(data)} examples to {path}")

    # Write a small test set (first 10 from val)
    test_path = os.path.join(args.output_dir, "test.jsonl")
    test_data = val_data[: min(10, len(val_data))]
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(test_data)} examples to {test_path}")

    logger.info(f"Data preparation complete! Files saved to {args.output_dir}/")
    logger.info("Next: Run MLX fine-tuning with:")
    logger.info(f"  python train/mlx_finetune.py --data {args.output_dir}")


if __name__ == "__main__":
    main()
