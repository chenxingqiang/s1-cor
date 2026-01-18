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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    
    # Note: num_generations is defined in GRPOConfig, not here to avoid conflict
    
    # CoR reward configuration
    lambda_intrinsic: float = field(default=1.0)
    self_rating_weight: float = field(default=0.2)
    calibration_bonus: float = field(default=0.2)
    
    # NEW: Self-reflection configuration (theory.md Section 14-15)
    improvement_weight: float = field(default=0.5)  # μ: R_improve weight
    convergence_weight: float = field(default=0.1)  # ν: R_converge weight
    max_reflection_rounds: int = field(default=3)   # K: max iterations
    enable_reflection: bool = field(default=True)   # Enable multi-round reflection
    
    # PEFT/LoRA configuration for memory-efficient training
    use_peft: bool = field(default=False)  # Enable PEFT/LoRA
    lora_r: int = field(default=16)  # LoRA rank
    lora_alpha: int = field(default=32)  # LoRA alpha
    lora_dropout: float = field(default=0.05)  # LoRA dropout
    lora_target_modules: Optional[str] = field(default=None)  # Comma-separated target modules
    
    # W&B
    wandb_project: Optional[str] = field(default="cor-grpo")
    wandb_entity: Optional[str] = field(default=None)
    
    def __post_init__(self):
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity


def create_reward_fn(config: CoRTrainingConfig, dataset: Dataset = None, enable_logging: bool = True):
    """Create reward function for GRPO trainer.
    
    Supports both standard CoR and self-reflection modes.
    Returns a callable compatible with TRL's GRPOTrainer.
    
    Args:
        config: Training configuration
        dataset: Training dataset with ground_truth field for external rewards
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
    
    # Build prompt-to-ground_truth mapping if dataset is provided
    prompt_to_gt = {}
    if dataset is not None:
        for item in dataset:
            prompt = item.get('prompt', '')
            gt = item.get('ground_truth', '')
            if prompt and gt:
                # Use first 500 chars as key to handle variations
                key = prompt[:500].strip()
                prompt_to_gt[key] = gt
        logging.info(f"Built prompt-to-ground_truth mapping with {len(prompt_to_gt)} entries")
    
    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """Compute rewards for completions.
        
        Supports multi-round reflection if enabled.
        Logs detailed reward breakdowns for validation.
        
        Args:
            completions: List of model-generated completions.
            prompts: List of input prompts (used to look up ground truths).
            **kwargs: May include 'ground_truths', etc.
            
        Returns:
            List of reward values.
        """
        global _global_step
        rewards = []
        
        # Get ground truths from kwargs or look up from dataset
        ground_truths = kwargs.get('ground_truths', [])
        
        if not ground_truths and prompts is not None:
            # Look up ground truths from prompt mapping
            ground_truths = []
            for prompt in prompts:
                key = prompt[:500].strip() if prompt else ''
                gt = prompt_to_gt.get(key)
                ground_truths.append(gt)
        
        if not ground_truths:
            ground_truths = [None] * len(completions)
        
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
        
        # Debug logging for first few batches
        if _global_step <= 100:
            logging.info(f"[Reward Debug] Step {_global_step}: rewards={rewards[:5]}, "
                        f"mean={sum(rewards)/len(rewards) if rewards else 0:.4f}, "
                        f"std={torch.tensor(rewards).std().item() if len(rewards) > 1 else 0:.4f}, "
                        f"has_gt={sum(1 for gt in ground_truths if gt)}/{len(ground_truths)}")
        
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
    
    Extracts prompts and ground truth answers from the dataset.
    The ground truth is used for computing external rewards.
    """
    dataset = load_dataset(dataset_path)
    
    if 'train' in dataset:
        dataset = dataset['train']
    
    # Extract prompts and answers from text field
    def extract_prompt_and_answer(example):
        text = example.get('text', '')
        solution = example.get('solution', '')
        
        prompt = text
        ground_truth = None
        
        # Extract user prompt from Qwen format
        if '<|im_start|>user' in text:
            import re
            match = re.search(
                r'<\|im_start\|>user\n(.+?)<\|im_end\|>',
                text,
                re.DOTALL
            )
            if match:
                prompt = match.group(1).strip()
        
        # Extract ground truth answer
        # Try to get from solution field first
        if solution:
            # Extract boxed answer if present
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
            if boxed_match:
                ground_truth = boxed_match.group(1).strip()
            else:
                # Use last line or "The answer is" pattern
                answer_match = re.search(r'(?:answer is|Answer:)\s*(.+?)(?:\.|$)', solution, re.IGNORECASE)
                if answer_match:
                    ground_truth = answer_match.group(1).strip()
        
        # If no solution field, try to extract from text (assistant response)
        if ground_truth is None and '<|im_start|>assistant' in text:
            assistant_match = re.search(
                r'<\|im_start\|>assistant\n(.+?)(?:<\|im_end\|>|$)',
                text,
                re.DOTALL
            )
            if assistant_match:
                assistant_text = assistant_match.group(1)
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', assistant_text)
                if boxed_match:
                    ground_truth = boxed_match.group(1).strip()
        
        return {
            'prompt': prompt,
            'ground_truth': ground_truth if ground_truth else '',
        }
    
    return dataset.map(extract_prompt_and_answer)


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
    if any(size in config.model_name for size in ["14B", "32B", "70B"]):
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",  # Use SDPA for broader compatibility
            "use_cache": False,
            "low_cpu_mem_usage": True,
        }
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # Apply LoRA if enabled
    if config.use_peft:
        logging.info(f"Applying LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
        
        # Determine target modules
        if config.lora_target_modules:
            target_modules = config.lora_target_modules.split(",")
        else:
            # Default target modules for Qwen models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Enable gradient checkpointing before applying LoRA
        model.gradient_checkpointing_enable()
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Note: In newer TRL versions, ref_model is handled internally by GRPOTrainer
    # GRPOTrainer will create its own ref_model copy, which for PEFT models
    # should share the base weights and only differ in LoRA parameters
    
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
    
    # Check for ground truth availability
    gt_count = sum(1 for item in dataset if item.get('ground_truth'))
    logging.info(f"Samples with ground truth: {gt_count}/{len(dataset)}")
    
    # Create reward function with dataset for ground truth lookup
    logging.info("Creating reward function...")
    reward_fn = create_reward_fn(config, dataset=dataset)
    
    # Configure GRPO
    grpo_args.max_completion_length = config.block_size
    
    # Initialize trainer
    logging.info("Initializing GRPOTrainer...")
    
    trainer = GRPOTrainer(
        model=model,
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
