#!/usr/bin/env python3
"""
MLX-Tune GRPO Training Script for CoR Models on Mac (Apple Silicon).

Uses mlx-tune's GRPOTrainer (https://github.com/ARahim3/mlx-tune) to train
reasoning models with Chain of Reward (CoR) — DeepSeek R1 style.

GRPO generates multiple completions per prompt, scores them with custom
CoR reward functions, and uses group-normalized advantages for policy updates.

The CoR reward combines:
  - External correctness reward (binary: correct/incorrect)
  - Intrinsic quality reward (self-rating calibration)
  - Format reward (reasoning structure)

Usage:
    # Quick start with defaults
    python train/mlx_grpo.py

    # With specific model
    python train/mlx_grpo.py --model_size 1.5B

    # With custom reward weights
    python train/mlx_grpo.py --lambda_intrinsic 1.0 --format_weight 0.2

    # Prepare data + train
    python train/mlx_grpo.py --prepare_data --model_size 0.5B
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# CoR System Prompt — encourages self-rating during reasoning
COR_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step.
Always show your detailed reasoning before giving the final answer.
After each major reasoning step, evaluate your own work with a self-rating:
[Self-Rating: Consistency=X/10, Completeness=X/10, Accuracy=X/10, Clarity=X/10]
At the end, provide an overall quality score:
[Overall Quality: X/10]

Format your response as:
<reasoning>
[Your step-by-step work with self-ratings here]
</reasoning>
<answer>
[Your final answer here]
</answer>"""

# Model configurations
MODEL_CONFIGS = {
    "0.5B": {
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "lora_rank": 16,
        "max_seq_length": 1024,
        "max_steps": 20,
    },
    "1.5B": {
        "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "lora_rank": 16,
        "max_seq_length": 1024,
        "max_steps": 20,
    },
    "3B": {
        "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "lora_rank": 16,
        "max_seq_length": 1024,
        "max_steps": 15,
    },
    "4B": {
        "model": "mlx-community/Qwen3-4B-4bit",
        "lora_rank": 16,
        "max_seq_length": 1024,
        "max_steps": 15,
    },
    "7B": {
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "lora_rank": 8,
        "max_seq_length": 1024,
        "max_steps": 10,
    },
}


# =============================================================================
# CoR Reward Functions
# =============================================================================

def correctness_reward(response: str, ground_truth: str) -> float:
    """
    External correctness reward (R_ext).
    Binary: 1.0 if correct, 0.0 otherwise. Partial credit for close answers.
    """
    # Extract from <answer> tags
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        extracted = response.strip()

    # Numeric comparison
    response_nums = re.findall(r'-?\d+\.?\d*', extracted)
    truth_nums = re.findall(r'-?\d+\.?\d*', ground_truth)

    if response_nums and truth_nums:
        try:
            import math
            if math.isclose(float(response_nums[-1]), float(truth_nums[-1]),
                            rel_tol=1e-6):
                return 1.0
        except ValueError:
            pass

    # String containment (partial credit)
    if ground_truth.strip().lower() in response.lower():
        return 0.5

    return 0.0


def format_reward(response: str, ground_truth: str) -> float:
    """
    Format reward — checks for proper CoR structure.
    Rewards <reasoning>/<answer> tags and self-rating markers.
    """
    score = 0.0

    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))

    if has_reasoning:
        score += 0.3
    if has_answer:
        score += 0.2

    return score


def self_rating_reward(response: str, ground_truth: str) -> float:
    """
    Intrinsic self-rating reward (R_int).
    Rewards the presence and quality of self-ratings in the thinking chain.
    """
    score = 0.0

    # Check for self-rating markers
    ratings = re.findall(r'\[Self-Rating:\s*([^\]]+)\]', response)
    if ratings:
        score += min(len(ratings) * 0.15, 0.5)  # Up to 0.5 for having ratings

    # Check for overall quality
    overall = re.search(r'\[Overall Quality:\s*([\d.]+)/10\]', response)
    if overall:
        score += 0.2

    # Check for dimension completeness in ratings
    for rating_text in ratings:
        dims_found = 0
        for dim in ['Consistency', 'Completeness', 'Accuracy', 'Clarity']:
            if re.search(rf'{dim}=\d+/10', rating_text):
                dims_found += 1
        if dims_found == 4:
            score += 0.1  # Bonus for complete ratings

    return min(score, 1.0)


def cor_combined_reward(response: str, ground_truth: str,
                        lambda_intrinsic: float = 0.5,
                        format_weight: float = 0.2) -> float:
    """
    Combined CoR reward: R = R_ext + λ·R_int + w_fmt·R_fmt

    This is the main reward function that combines:
    - External correctness (primary objective)
    - Intrinsic self-rating quality (CoR innovation)
    - Format compliance (structured output)
    """
    r_ext = correctness_reward(response, ground_truth)
    r_int = self_rating_reward(response, ground_truth)
    r_fmt = format_reward(response, ground_truth)

    total = r_ext + lambda_intrinsic * r_int + format_weight * r_fmt
    return total


def make_cor_reward_fn(lambda_intrinsic: float = 0.5,
                       format_weight: float = 0.2):
    """Create a CoR reward function with configurable weights."""
    def reward_fn(response: str, ground_truth: str) -> float:
        return cor_combined_reward(
            response, ground_truth,
            lambda_intrinsic=lambda_intrinsic,
            format_weight=format_weight,
        )
    return reward_fn


# =============================================================================
# Training Data
# =============================================================================

def load_grpo_data(data_path: str):
    """Load GRPO training data from JSONL or build from CoR dataset."""
    if os.path.exists(data_path):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Convert from messages format to prompt/answer format
                    if "messages" in item:
                        messages = item["messages"]
                        question = ""
                        answer = ""
                        for msg in messages:
                            if msg["role"] == "user":
                                question = msg["content"]
                            elif msg["role"] == "assistant":
                                answer = msg["content"]
                        if question:
                            data.append({
                                "prompt": f"{COR_SYSTEM_PROMPT}\n\n{question}",
                                "answer": answer,
                            })
                    elif "prompt" in item:
                        data.append(item)
        logger.info(f"Loaded {len(data)} GRPO examples from {data_path}")
        return data
    else:
        logger.warning(f"Data file not found: {data_path}")
        return get_default_math_data()


def get_default_math_data():
    """Default math reasoning dataset for GRPO demo."""
    problems = [
        ("What is 15 + 27?", "42"),
        ("What is 8 * 7?", "56"),
        ("What is 100 - 37?", "63"),
        ("What is 144 / 12?", "12"),
        ("If a train travels at 60 mph for 2 hours, how far does it go?", "120"),
        ("What is 25% of 80?", "20"),
        ("A store has 45 apples. If 18 are sold, how many remain?", "27"),
        ("What is 2^5?", "32"),
        ("If x + 7 = 15, what is x?", "8"),
        ("What is the sum of 11, 22, and 33?", "66"),
        ("Solve: 3x - 9 = 0", "3"),
        ("What is the area of a circle with radius 5? (use pi=3.14)", "78.5"),
    ]
    return [
        {"prompt": f"{COR_SYSTEM_PROMPT}\n\n{q}", "answer": a}
        for q, a in problems
    ]


# =============================================================================
# Main Training
# =============================================================================

def run_grpo_training(args):
    """Run GRPO training with CoR rewards using mlx-tune."""
    from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig

    # Step 1: Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    logger.info("Model loaded successfully")

    # Step 2: Apply LoRA
    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
    )

    # Step 3: Load data
    data_file = os.path.join(args.data, "train.jsonl")
    train_data = load_grpo_data(data_file)

    # Step 4: Create CoR reward function
    reward_fn = make_cor_reward_fn(
        lambda_intrinsic=args.lambda_intrinsic,
        format_weight=args.format_weight,
    )
    logger.info(f"CoR reward: R = R_ext + {args.lambda_intrinsic}·R_int + {args.format_weight}·R_fmt")

    # Step 5: Configure GRPO
    config = GRPOConfig(
        loss_type=args.loss_type,
        beta=args.beta,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=1,
        output_dir=args.output_dir,
    )

    # Step 6: Create trainer
    trainer = GRPOTrainer(
        model=model,
        train_dataset=train_data,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        args=config,
    )

    logger.info(f"GRPO Config:")
    logger.info(f"  Loss type: {args.loss_type}")
    logger.info(f"  Num generations: {args.num_generations}")
    logger.info(f"  Beta (KL): {args.beta}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max steps: {args.max_steps}")

    # Step 7: Train
    logger.info("Starting GRPO training...")
    result = trainer.train()

    logger.info(f"Training result: {result.get('status', 'completed')}")
    if 'adapter_path' in result:
        logger.info(f"Adapters saved to: {result['adapter_path']}")

    # Save adapters
    model.save_pretrained(args.output_dir)
    logger.info(f"LoRA adapters saved to: {args.output_dir}")

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="mlx-tune GRPO Training with CoR Rewards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with math data
  python train/mlx_grpo.py --model_size 0.5B

  # With CoR dataset
  python train/mlx_grpo.py --model_size 0.5B --prepare_data

  # Custom reward weights
  python train/mlx_grpo.py --lambda_intrinsic 1.0 --format_weight 0.3

  # Different GRPO variant
  python train/mlx_grpo.py --loss_type dr_grpo
        """,
    )

    # Model
    parser.add_argument("--model_size", type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Preset model size")
    parser.add_argument("--model", type=str,
                        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                        help="Model name or path")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=1024)

    # Data
    parser.add_argument("--prepare_data", action="store_true",
                        help="Prepare data from CoR dataset before training")
    parser.add_argument("--dataset", type=str, default="deepseek")
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--data", type=str, default="train/mlx_data",
                        help="Path to JSONL data directory")
    parser.add_argument("--max_samples", type=int, default=None)

    # GRPO parameters
    parser.add_argument("--loss_type", type=str, default="grpo",
                        choices=["grpo", "dr_grpo", "dapo", "bnpo"],
                        help="GRPO loss type variant")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient")
    parser.add_argument("--num_generations", type=int, default=2,
                        help="Number of completions per prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_completion_length", type=int, default=256,
                        help="Max tokens per completion")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Maximum training steps")

    # CoR reward weights
    parser.add_argument("--lambda_intrinsic", type=float, default=0.5,
                        help="Weight for intrinsic self-rating reward (λ)")
    parser.add_argument("--format_weight", type=float, default=0.2,
                        help="Weight for format compliance reward")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="ckpts/mlx_grpo_adapters",
                        help="Output directory")

    # Misc
    parser.add_argument("--install_deps", action="store_true",
                        help="Install mlx-tune before training")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("mlx-tune GRPO Training with CoR Rewards")
    logger.info("(DeepSeek R1 style — on Apple Silicon)")
    logger.info("=" * 60)

    # Apply preset
    if args.model_size:
        config = MODEL_CONFIGS[args.model_size]
        args.model = config["model"]
        args.lora_rank = config["lora_rank"]
        args.max_seq_length = config["max_seq_length"]
        args.max_steps = config["max_steps"]
        logger.info(f"Using preset: {args.model_size}")

    # Install deps
    if args.install_deps:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "mlx-tune"],
        )

    # Check availability
    try:
        import mlx_tune  # noqa: F401
    except ImportError:
        logger.error(
            "mlx-tune is not installed.\n"
            "Install with: pip install mlx-tune\n"
            "Or run with: python train/mlx_grpo.py --install_deps"
        )
        sys.exit(1)

    # Prepare data
    if args.prepare_data:
        cmd = [
            sys.executable, "train/mlx_prepare_data.py",
            "--dataset", args.dataset,
            "--output_dir", args.data,
            "--format", "chat",
        ]
        if args.max_samples:
            cmd.extend(["--max_samples", str(args.max_samples)])
        subprocess.check_call(cmd)

    # Run training
    model, tokenizer = run_grpo_training(args)

    logger.info("=" * 60)
    logger.info("GRPO Training Complete!")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Adapters: {args.output_dir}")
    logger.info(f"  CoR Reward: R_ext + {args.lambda_intrinsic}·R_int + {args.format_weight}·R_fmt")
    logger.info("")
    logger.info("To test the model:")
    logger.info(f"  python train/mlx_inference.py --model {args.model} "
                f"--adapter_path {args.output_dir} --eval_cor")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
