"""
GRPO Training Script for Chain of Reward (CoR).

Implements CoR-GRPO dual coupling with self-reflection support.
Based on DESIGN.md and THEORY.md (Sections 9-17).

Extended reward formula:
R = R_ext + λ·R_int + μ·R_improve + ν·R_converge

Usage:
    python train/grpo.py --model_name Qwen/Qwen2.5-32B-Instruct \
                         --train_file_path simplescaling/s1K_tokenized \
                         --output_dir ckpts/cor-grpo
"""

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import trl
from trl import GRPOTrainer, GRPOConfig

from rewards import RewardCalculator, RewardConfig
from rewards.training_logger import CoRTrainingLogger, log_cor_reward

# Global training logger
_training_logger: CoRTrainingLogger = None
_global_step: int = 0


def get_training_logger(log_every_n: int = 10, verbose: bool = True) -> CoRTrainingLogger:
    """Get or create training logger."""
    global _training_logger
    if _training_logger is None:
        _training_logger = CoRTrainingLogger(
            log_every_n=log_every_n,
            verbose=verbose,
            log_file="cor_training_log.jsonl"
        )
    return _training_logger


@dataclass
class CoRTrainingConfig:
    """Configuration for CoR + GRPO training with self-reflection."""
    
    # Model
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    ref_model_name: Optional[str] = field(default=None)  # If None, uses model_name
    
    # Data
    train_file_path: str = field(default='simplescaling/s1K_tokenized')
    block_size: int = field(default=32768)
    
    # GRPO specific (per paper: N=8 candidates)
    num_generations: int = field(default=8)  # N candidates per input
    
    # CoR reward configuration
    lambda_intrinsic: float = field(default=1.0)
    self_rating_weight: float = field(default=0.2)
    calibration_bonus: float = field(default=0.2)
    
    # NEW: Self-reflection configuration (theory.md Section 14-15)
    improvement_weight: float = field(default=0.5)  # μ: R_improve weight
    convergence_weight: float = field(default=0.1)  # ν: R_converge weight
    max_reflection_rounds: int = field(default=3)   # K: max iterations
    enable_reflection: bool = field(default=True)   # Enable multi-round reflection
    
    # W&B
    wandb_project: Optional[str] = field(default="cor-grpo")
    wandb_entity: Optional[str] = field(default=None)
    
    def __post_init__(self):
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity


def create_reward_fn(config: CoRTrainingConfig, enable_logging: bool = True):
    """Create reward function for GRPO trainer.
    
    Supports both standard CoR and self-reflection modes.
    Returns a callable compatible with TRL's GRPOTrainer.
    
    Args:
        config: Training configuration
        enable_logging: Whether to enable detailed CoR logging
    """
    global _global_step
    
    reward_config = RewardConfig(
        lambda_intrinsic=config.lambda_intrinsic,
        self_rating_weight=config.self_rating_weight,
        calibration_bonus=config.calibration_bonus,
        improvement_weight=config.improvement_weight,
        convergence_weight=config.convergence_weight,
        max_reflection_rounds=config.max_reflection_rounds,
    )
    
    calculator = RewardCalculator(reward_config)
    logger = get_training_logger(log_every_n=10, verbose=True) if enable_logging else None
    
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        """Compute rewards for completions.
        
        Supports multi-round reflection if enabled.
        Logs detailed reward breakdowns for validation.
        
        Args:
            completions: List of model-generated completions.
            **kwargs: May include 'prompts', 'ground_truths', etc.
            
        Returns:
            List of reward values.
        """
        global _global_step
        rewards = []
        
        ground_truths = kwargs.get('ground_truths', [None] * len(completions))
        
        for i, completion in enumerate(completions):
            _global_step += 1
            gt = ground_truths[i] if i < len(ground_truths) else None
            
            # Check if completion has multiple reflection rounds
            chain_sequence = extract_reflection_rounds(completion)
            
            if len(chain_sequence) > 1 and config.enable_reflection:
                # Multi-round reflection: use extended reward
                thinking, answer = parse_qwen_completion(chain_sequence[-1])
                
                if gt is None:
                    # No ground truth - use intrinsic + improvement rewards
                    intrinsic, dim_scores = calculator.calculate_intrinsic_reward(
                        chain_sequence[-1],
                        include_self_rating=True,
                        final_answer_correct=False
                    )
                    # Add improvement reward
                    improvement = calculator.improvement_calculator.compute_cumulative_improvement(
                        chain_sequence
                    )
                    total = intrinsic + config.improvement_weight * improvement
                    rewards.append(total)
                    
                    # Log
                    if logger:
                        self_ratings = calculator.self_rating_extractor.extract(chain_sequence[-1])
                        logger.log_reward(
                            step=_global_step,
                            sample_id=f"sample_{i}",
                            r_external=0.0,
                            r_intrinsic=intrinsic,
                            r_total=total,
                            dim_scores=dim_scores,
                            answer_correct=False,
                            thinking_chain=thinking,
                            r_improve=improvement,
                            reflection_rounds=len(chain_sequence),
                            self_ratings=self_ratings,
                        )
                else:
                    output = calculator.calculate_reflection_reward(
                        chain_sequence, answer, gt
                    )
                    rewards.append(output.total_reward)
                    
                    # Log
                    if logger:
                        self_ratings = calculator.self_rating_extractor.extract(chain_sequence[-1])
                        logger.log_reward(
                            step=_global_step,
                            sample_id=f"sample_{i}",
                            r_external=output.external_reward,
                            r_intrinsic=output.intrinsic_reward,
                            r_total=output.total_reward,
                            dim_scores=output.dimension_scores,
                            answer_correct=output.external_reward > 0.5,
                            thinking_chain=thinking,
                            r_improve=output.improvement_reward,
                            r_converge=output.convergence_reward,
                            reflection_rounds=len(chain_sequence),
                            self_ratings=self_ratings,
                        )
            else:
                # Single-round: standard CoR reward
                thinking, answer = parse_qwen_completion(completion)
                
                if gt is None:
                    intrinsic, dim_scores = calculator.calculate_intrinsic_reward(
                        thinking,
                        include_self_rating=True,
                        final_answer_correct=False
                    )
                    rewards.append(intrinsic)
                    
                    # Log
                    if logger:
                        self_ratings = calculator.self_rating_extractor.extract(thinking)
                        logger.log_reward(
                            step=_global_step,
                            sample_id=f"sample_{i}",
                            r_external=0.0,
                            r_intrinsic=intrinsic,
                            r_total=intrinsic,
                            dim_scores=dim_scores,
                            answer_correct=False,
                            thinking_chain=thinking,
                            self_ratings=self_ratings,
                        )
                else:
                    output = calculator.calculate_total_reward(
                        thinking, answer, gt
                    )
                    rewards.append(output.total_reward)
                    
                    # Log
                    if logger:
                        self_ratings = calculator.self_rating_extractor.extract(thinking)
                        logger.log_reward(
                            step=_global_step,
                            sample_id=f"sample_{i}",
                            r_external=output.external_reward,
                            r_intrinsic=output.intrinsic_reward,
                            r_total=output.total_reward,
                            dim_scores=output.dimension_scores,
                            answer_correct=output.external_reward > 0.5,
                            thinking_chain=thinking,
                            self_ratings=self_ratings,
                        )
        
        # Log batch summary
        if logger and len(rewards) > 0:
            logger.log_batch(
                step=_global_step,
                batch_rewards=rewards,
                batch_correct=[r > 0.5 for r in rewards],
                mean_reward=sum(rewards) / len(rewards),
            )
        
        return rewards
    
    return reward_fn


def extract_reflection_rounds(completion: str) -> List[str]:
    """Extract individual reflection rounds from a completion.
    
    Parses format:
    [Round 1]
    ...thinking...
    [Self-Rating: ...]
    
    [Reflection]
    ...
    
    [Round 2]
    ...thinking...
    
    Returns list of thinking chains [c_0, c_1, ..., c_K]
    """
    # Check for round markers
    round_pattern = r'\[Round (\d+)\]'
    rounds = list(re.finditer(round_pattern, completion))
    
    if len(rounds) < 2:
        # No multiple rounds, return entire completion as single chain
        return [completion]
    
    chain_sequence = []
    
    for i, match in enumerate(rounds):
        start = match.end()
        
        # End is either next round or end of string
        if i + 1 < len(rounds):
            end = rounds[i + 1].start()
        else:
            end = len(completion)
        
        # Extract this round's content
        round_content = completion[start:end].strip()
        
        # Remove reflection section if present (belongs to previous round's analysis)
        reflection_match = re.search(r'\[Reflection\].*$', round_content, re.DOTALL)
        if reflection_match:
            round_content = round_content[:reflection_match.start()].strip()
        
        if round_content:
            chain_sequence.append(round_content)
    
    return chain_sequence if chain_sequence else [completion]


def parse_qwen_completion(completion: str) -> tuple:
    """Parse Qwen-format completion into thinking and answer.
    
    Format: <|im_start|>assistant\n{thinking}<|im_end|>\n{answer}
    
    Returns:
        Tuple of (thinking, answer).
    """
    import re
    
    # Handle Qwen format
    if '<|im_end|>' in completion:
        parts = completion.split('<|im_end|>')
        thinking = parts[0].replace('<|im_start|>assistant\n', '').strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return thinking, answer
    
    # Handle generic format - look for answer markers
    answer_patterns = [
        r'\n[Aa]nswer:\s*(.+?)$',
        r'[Tt]he (?:final )?answer is[:\s]+(.+?)$',
        r'\n[Ff]inal [Aa]nswer:\s*(.+?)$',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            thinking = completion[:match.start()].strip()
            return thinking, answer
    
    # Fallback: last line is answer
    lines = completion.strip().split('\n')
    if len(lines) > 1:
        return '\n'.join(lines[:-1]), lines[-1]
    
    return completion, completion


def prepare_dataset(dataset_path: str, tokenizer) -> Dataset:
    """Prepare dataset for GRPO training.
    
    Expects dataset with 'text' field containing prompts.
    """
    dataset = load_dataset(dataset_path)
    
    if 'train' in dataset:
        dataset = dataset['train']
    
    # Extract prompts from text field
    def extract_prompt(example):
        text = example.get('text', '')
        
        # Extract user prompt from Qwen format
        if '<|im_start|>user' in text:
            import re
            match = re.search(
                r'<\|im_start\|>user\n(.+?)<\|im_end\|>',
                text,
                re.DOTALL
            )
            if match:
                return {'prompt': match.group(1).strip()}
        
        # Fallback
        return {'prompt': text}
    
    return dataset.map(extract_prompt)


def train():
    """Main training function."""
    
    # Parse arguments
    parser = transformers.HfArgumentParser((CoRTrainingConfig, GRPOConfig))
    config, grpo_args = parser.parse_args_into_dataclasses()
    
    log_config = {**asdict(config), **asdict(grpo_args)}
    logging.info(f"Training config: {log_config}")
    
    # Load model
    logging.info(f"Loading model: {config.model_name}")
    
    model_kwargs = {}
    if "70B" in config.model_name or "32B" in config.model_name:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
        }
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # Load reference model (for KL penalty)
    ref_model_name = config.ref_model_name or config.model_name
    logging.info(f"Loading reference model: {ref_model_name}")
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        **model_kwargs
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    if "Qwen" in config.model_name:
        tokenizer.pad_token = "<|fim_pad|>"
    elif "Llama" in config.model_name:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    
    # Prepare dataset
    logging.info(f"Loading dataset: {config.train_file_path}")
    dataset = prepare_dataset(config.train_file_path, tokenizer)
    logging.info(f"Dataset size: {len(dataset)}")
    
    # Create reward function
    logging.info("Creating reward function...")
    reward_fn = create_reward_fn(config)
    
    # Configure GRPO
    grpo_args.max_completion_length = config.block_size
    
    # Initialize trainer
    logging.info("Initializing GRPOTrainer...")
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    
    # Train
    logging.info("Starting training...")
    trainer.train()
    
    # Save
    logging.info(f"Saving model to {grpo_args.output_dir}")
    trainer.save_model(grpo_args.output_dir)
    tokenizer.save_pretrained(grpo_args.output_dir)
    
    logging.info("Training complete!")


if __name__ == "__main__":
    train()
