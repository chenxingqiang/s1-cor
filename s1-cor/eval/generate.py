#!/usr/bin/env python3
"""
CoR Model Evaluation Script

Supports evaluation on AIME24, MATH500, GPQA benchmarks.
Compatible with both CoR-trained and baseline Qwen models.

Usage:
    python eval/generate.py --model_path ckpts/cor-grpo --benchmark aime24
    python eval/generate.py --model_path Qwen/Qwen2.5-32B-Instruct --benchmark math500
"""

import os
import sys
import json
import argparse
from typing import Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="CoR Model Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpts/cor-grpo",
        help="Path to model checkpoint or HuggingFace model name"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="aime24",
        choices=["aime24", "math500", "gpqa_diamond", "all"],
        help="Benchmark to evaluate on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/eval",
        help="Output directory for results"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--budget_forcing",
        action="store_true",
        help="Enable budget forcing (thinking budget control)"
    )
    parser.add_argument(
        "--max_thinking_tokens",
        type=int,
        default=None,
        help="Maximum thinking tokens (for budget forcing)"
    )
    return parser.parse_args()


def load_benchmark(benchmark: str):
    """Load benchmark dataset.
    
    Note: This is a placeholder. In practice, you should use
    lm-evaluation-harness for proper benchmark loading.
    """
    # Placeholder - actual implementation should use lm-eval-harness
    if benchmark == "aime24":
        return [
            {"prompt": "Solve the following AIME problem: ...", "answer": "..."}
        ]
    elif benchmark == "math500":
        return [
            {"prompt": "Solve: ...", "answer": "..."}
        ]
    elif benchmark == "gpqa_diamond":
        return [
            {"prompt": "Answer: ...", "answer": "..."}
        ]
    return []


def format_prompt(question: str, model_type: str = "qwen") -> str:
    """Format prompt for model input."""
    if model_type == "qwen":
        return (
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return question


def run_evaluation(args):
    """Run evaluation on specified benchmark."""
    print(f"=" * 60)
    print(f"CoR Model Evaluation")
    print(f"=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Output: {args.output_dir}")
    print(f"=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = LLM(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    # Setup sampling params
    stop_token_ids = tokenizer("<|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=args.temperature,
    )
    
    # Load benchmark
    benchmarks = [args.benchmark] if args.benchmark != "all" else ["aime24", "math500", "gpqa_diamond"]
    
    for benchmark in benchmarks:
        print(f"\nEvaluating on {benchmark}...")
        
        # For proper evaluation, use lm-evaluation-harness
        print(f"\nNote: For full benchmark evaluation, use lm-evaluation-harness:")
        print(f"  cd eval/lm-evaluation-harness")
        print(f"  lm_eval --model vllm \\")
        print(f"    --model_args pretrained={args.model_path} \\")
        print(f"    --tasks {benchmark}_nofigures \\")
        print(f"    --batch_size auto \\")
        print(f"    --apply_chat_template \\")
        print(f"    --output_path {args.output_dir}/{benchmark}")
        print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, "eval_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
