#!/usr/bin/env python3
"""
SFT Training Script for Small Qwen Models with CoR Data

This script performs supervised fine-tuning on small Qwen models
using the CoR dataset (with self-ratings in thinking chains).

Usage:
    python train/sft_small.py --model_size 0.5B
    python train/sft_small.py --model_size 1.5B --dataset deepseek
"""

import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# Setup logging with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
for handler in logging.root.handlers:
    handler.flush = lambda: sys.stdout.flush()
logger = logging.getLogger(__name__)
logger.handlers = logging.root.handlers


# Model configurations
QWEN_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}

# Hyperparameters by size
MODEL_CONFIGS = {
    "0.5B": {
        "batch_size": 4,
        "grad_accum": 4,
        "max_length": 4096,
        "lr": 2e-5,
    },
    "1.5B": {
        "batch_size": 2,
        "grad_accum": 8,
        "max_length": 8192,
        "lr": 1e-5,
    },
    "3B": {
        "batch_size": 1,
        "grad_accum": 16,
        "max_length": 8192,
        "lr": 5e-6,
    },
    "7B": {
        "batch_size": 1,
        "grad_accum": 16,
        "max_length": 16384,
        "lr": 2e-6,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="SFT for small Qwen models")
    parser.add_argument(
        "--model_size",
        type=str,
        default="0.5B",
        choices=list(QWEN_MODELS.keys()),
        help="Model size to train"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="full",
        choices=["full", "deepseek", "hf"],
        help="Dataset to use (full/deepseek for local, hf for HuggingFace)"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="xingqiang/s1K-cor-deepseek",
        help="HuggingFace dataset name (used when --dataset=hf)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ckpts/sft-{size})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases logging"
    )
    return parser.parse_args()


def prepare_dataset(tokenizer, dataset_name: str, max_length: int, hf_dataset: str = None):
    """Load and prepare the CoR dataset for SFT."""
    
    if dataset_name == "hf":
        # Load from HuggingFace Hub
        logger.info(f"Loading dataset from HuggingFace: {hf_dataset}")
        dataset = load_dataset(hf_dataset, split="train")
    elif dataset_name == "deepseek":
        dataset_path = "local_data/s1K_cor_deepseek"
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        dataset_path = "local_data/s1K_cor_full"
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)
    
    def tokenize_function(examples):
        """Tokenize the text_cor field which contains the full training text."""
        # Use text_cor which has the CoR format with self-ratings
        texts = examples.get("text_cor", examples.get("text", []))
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    logger.info(f"Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset


def main():
    args = parse_args()
    
    # Configuration
    model_name = QWEN_MODELS[args.model_size]
    config = MODEL_CONFIGS[args.model_size]
    output_dir = args.output_dir or f"ckpts/sft-{args.model_size}"
    
    logger.info(f"=" * 50)
    logger.info(f"CoR SFT Training - Qwen2.5-{args.model_size}")
    logger.info(f"=" * 50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Config: {config}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        device_map = "auto"
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16  # MPS doesn't support bfloat16 well
        device_map = None  # MPS doesn't support device_map
    else:
        device = "cpu"
        dtype = torch.float32
        device_map = None
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if device_map is None:
        model = model.to(device)
    model.config.use_cache = False
    
    # Prepare dataset
    train_dataset = prepare_dataset(
        tokenizer,
        args.dataset,
        config["max_length"],
        hf_dataset=args.hf_dataset,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(device == "cuda"),
        fp16=(device == "mps"),
        gradient_checkpointing=(device != "mps"),  # MPS has issues with gradient checkpointing
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"cor-sft-{args.model_size}-{args.dataset}",
        use_mps_device=(device == "mps"),
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
