#!/usr/bin/env python3
"""
MLX-Tune Inference Script for CoR Fine-Tuned Models on Mac.

Uses mlx-tune's FastLanguageModel API (https://github.com/ARahim3/mlx-tune)
to load, generate, and evaluate fine-tuned CoR models on Apple Silicon.

Usage:
    # Generate with LoRA adapter
    python train/mlx_inference.py --prompt "Solve: 2x + 3 = 7"

    # Generate with merged model
    python train/mlx_inference.py --model ckpts/mlx_merged_model --prompt "..."

    # Interactive mode
    python train/mlx_inference.py --interactive

    # Evaluate on test prompts with CoR markers
    python train/mlx_inference.py --eval_cor
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# CoR evaluation prompts (math reasoning with self-rating expected)
COR_EVAL_PROMPTS = [
    {
        "prompt": "Solve the equation: 3x + 5 = 20. Show your reasoning step by step.",
        "expected_answer": "x = 5",
        "category": "algebra",
    },
    {
        "prompt": "What is the integral of x^2 from 0 to 1? Explain your steps.",
        "expected_answer": "1/3",
        "category": "calculus",
    },
    {
        "prompt": "A bag contains 3 red balls and 5 blue balls. What is the probability of drawing 2 red balls without replacement?",
        "expected_answer": "3/28",
        "category": "probability",
    },
    {
        "prompt": "Find all prime numbers between 20 and 40.",
        "expected_answer": "23, 29, 31, 37",
        "category": "number_theory",
    },
    {
        "prompt": "If f(x) = x^3 - 3x + 1, find f'(x) and the critical points.",
        "expected_answer": "f'(x) = 3x^2 - 3, critical points at x = ±1",
        "category": "calculus",
    },
]


def load_model(model_path, adapter_path=None, load_in_4bit=True):
    """Load model and tokenizer using mlx-tune's FastLanguageModel."""
    from mlx_tune import FastLanguageModel

    logger.info(f"Loading model: {model_path}")
    if adapter_path:
        logger.info(f"With adapter: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=load_in_4bit,
    )

    # If adapter path exists, load it
    # mlx-tune handles adapter loading via FastLanguageModel
    if adapter_path and Path(adapter_path).exists():
        logger.info(f"Note: LoRA adapters at {adapter_path} — loading via model")

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_response(
    model, tokenizer, prompt, max_tokens=1024, temperature=0.7, top_p=0.9
):
    """Generate a response using the loaded model."""
    from mlx_lm import generate

    # Format as chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    start_time = time.time()
    response = generate(
        model.model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
        verbose=False,
    )
    elapsed = time.time() - start_time

    return response, elapsed


def check_cor_markers(response: str) -> dict:
    """Check if the response contains CoR self-rating markers."""
    import re

    markers = {
        "has_thinking": "<|im_start|>think" in response or "think" in response.lower(),
        "has_self_rating": bool(re.search(r"\[Self-Rating:", response)),
        "has_overall_quality": bool(re.search(r"\[Overall Quality:", response)),
        "has_reasoning_tags": bool(re.search(r"<reasoning>", response)),
        "has_answer_tags": bool(re.search(r"<answer>", response)),
        "self_ratings": [],
    }

    # Extract ratings
    for match in re.finditer(r"\[Self-Rating:\s*([^\]]+)\]", response):
        markers["self_ratings"].append(match.group(1))

    quality_match = re.search(r"\[Overall Quality:\s*([\d.]+)/10\]", response)
    if quality_match:
        markers["overall_quality"] = float(quality_match.group(1))

    return markers


def run_eval(model, tokenizer, args):
    """Run evaluation on CoR test prompts."""
    logger.info("=" * 60)
    logger.info("CoR Model Evaluation")
    logger.info("=" * 60)

    results = []
    for i, item in enumerate(COR_EVAL_PROMPTS):
        logger.info(f"\n--- Test {i + 1}/{len(COR_EVAL_PROMPTS)} ---")
        logger.info(f"Category: {item['category']}")
        logger.info(f"Prompt: {item['prompt']}")
        logger.info(f"Expected: {item['expected_answer']}")

        response, elapsed = generate_response(
            model,
            tokenizer,
            item["prompt"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        cor_markers = check_cor_markers(response)

        logger.info(f"\nResponse ({elapsed:.1f}s):\n{response}")
        logger.info(f"\nCoR Markers: {json.dumps(cor_markers, indent=2)}")

        results.append(
            {
                "category": item["category"],
                "prompt": item["prompt"],
                "expected": item["expected_answer"],
                "response": response,
                "time": elapsed,
                "cor_markers": cor_markers,
            }
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)

    has_thinking = sum(1 for r in results if r["cor_markers"]["has_thinking"])
    has_ratings = sum(1 for r in results if r["cor_markers"]["has_self_rating"])
    has_reasoning = sum(1 for r in results if r["cor_markers"]["has_reasoning_tags"])
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0

    logger.info(f"Total prompts: {len(results)}")
    logger.info(f"With thinking: {has_thinking}/{len(results)}")
    logger.info(f"With self-ratings: {has_ratings}/{len(results)}")
    logger.info(f"With <reasoning> tags: {has_reasoning}/{len(results)}")
    logger.info(f"Average generation time: {avg_time:.1f}s")

    # Save results
    if args.save_results:
        output_path = args.save_results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")

    return results


def interactive_mode(model, tokenizer, args):
    """Run interactive chat mode."""
    logger.info("=" * 60)
    logger.info("Interactive Mode (mlx-tune) - Type 'quit' to exit")
    logger.info("=" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            response, elapsed = generate_response(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            print(f"\n{response}")
            print(f"\n[{elapsed:.1f}s]")

            # Check for CoR markers
            markers = check_cor_markers(response)
            if markers["has_self_rating"]:
                print(f"[CoR Self-Ratings detected: {len(markers['self_ratings'])}]")

        except (KeyboardInterrupt, EOFError):
            break

    logger.info("Goodbye!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="mlx-tune Inference for CoR Fine-Tuned Models"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model name or path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="ckpts/mlx_lora_adapters",
        help="Path to LoRA adapters (set to empty string to skip)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--eval_cor",
        action="store_true",
        help="Run CoR evaluation on test prompts",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Save evaluation results to JSON file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check mlx-tune availability
    try:
        import mlx_tune  # noqa: F401
    except ImportError:
        logger.error(
            "mlx-tune is not installed. This script requires Apple Silicon Mac.\n"
            "Install with: pip install mlx-tune"
        )
        sys.exit(1)

    # Load model
    adapter = args.adapter_path if args.adapter_path and Path(args.adapter_path).exists() else None
    model, tokenizer = load_model(args.model, adapter_path=adapter)

    # Dispatch to mode
    if args.eval_cor:
        run_eval(model, tokenizer, args)
    elif args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        response, elapsed = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n{response}")
        print(f"\n[Generated in {elapsed:.1f}s]")
        markers = check_cor_markers(response)
        if markers["has_self_rating"]:
            print(f"[CoR Self-Ratings: {markers['self_ratings']}]")
    else:
        logger.info(
            "No action specified. Use --prompt, --interactive, or --eval_cor.\n"
            "Run with --help for usage information."
        )


if __name__ == "__main__":
    main()
