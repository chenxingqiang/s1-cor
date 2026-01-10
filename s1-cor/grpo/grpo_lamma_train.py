# -*- coding: utf-8 -*-
"""
Llama 3.1 8B GRPO Training with Chain of Reward (CoR)

This script implements GRPO training with CoR rewards on Llama models.
CoR extends standard reward with multi-dimensional intrinsic evaluation.

Based on Unsloth's GRPO notebook, extended with CoR:
- 5-dimension intrinsic rewards (Consistency, Completeness, Accuracy, Clarity, Format)
- Self-rating extraction and calibration
- Improvement and convergence rewards for reflection

Usage:
    python grpo/grpo_lamma_train.py

References:
    - CoR: theory.md, design.md
    - Unsloth: https://github.com/unslothai/unsloth
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from datasets import load_dataset, Dataset

# ============================================
# CoR Configuration
# ============================================

@dataclass
class CoRConfig:
    """Chain of Reward configuration.
    
    Parameters per theory.md:
    - λ (lambda_intrinsic): 1.0 - intrinsic reward weight
    - μ (improvement_weight): 0.5 - improvement reward weight  
    - ν (convergence_weight): 0.1 - convergence reward weight
    - K (max_reflection_rounds): 3 - max reflection iterations
    """
    lambda_intrinsic: float = 1.0
    improvement_weight: float = 0.5
    convergence_weight: float = 0.1
    max_reflection_rounds: int = 3
    
    # Dimension weights (w_d = 0.2 each for 5 dimensions)
    dimension_weights: Dict[str, float] = None
    self_rating_weight: float = 0.2
    calibration_bonus: float = 0.2
    
    def __post_init__(self):
        if self.dimension_weights is None:
            self.dimension_weights = {
                "consistency": 0.2,
                "completeness": 0.2,
                "accuracy": 0.2,
                "clarity": 0.2,
                "format": 0.2,
            }


# Global CoR config
COR_CONFIG = CoRConfig()

# ============================================
# CoR System Prompt with Self-Rating
# ============================================

COR_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step.

For each problem:
1. Think through the problem carefully
2. Show your reasoning step by step
3. Evaluate your own reasoning quality
4. Provide the final answer

At the end of your thinking, provide a self-rating in this format:
[Self-Rating: Consistency=X/10, Completeness=X/10, Accuracy=X/10, Clarity=X/10, Format=X/10]

Where:
- Consistency: How logically coherent is your reasoning? (1-10)
- Completeness: Did you cover all necessary steps? (1-10)
- Accuracy: How confident are you in the correctness? (1-10)
- Clarity: How clear and understandable is your explanation? (1-10)
- Format: Is your answer properly structured? (1-10)

Then provide your final answer."""

# Alternative: Math-focused system prompt
MATH_SYSTEM_PROMPT = """You are a mathematical reasoning assistant. Solve problems step by step.

For each problem:
1. Analyze what is given and what is asked
2. Plan your solution approach
3. Execute the solution with clear steps
4. Verify your answer
5. Rate your solution quality

After solving, rate yourself:
[Self-Rating: Consistency=X/10, Completeness=X/10, Accuracy=X/10, Clarity=X/10, Format=X/10]

Final Answer: [your answer]"""

# ============================================
# Self-Rating Extraction
# ============================================

class SelfRatingExtractor:
    """Extract self-ratings from model output."""
    
    DIMENSIONS = ["consistency", "completeness", "accuracy", "clarity", "format"]
    
    def __init__(self):
        # Pattern: [Self-Rating: Consistency=8/10, Completeness=9/10, ...]
        self.pattern = re.compile(
            r'\[Self-Rating:\s*'
            r'Consistency=(\d+)/10,?\s*'
            r'Completeness=(\d+)/10,?\s*'
            r'Accuracy=(\d+)/10,?\s*'
            r'Clarity=(\d+)/10,?\s*'
            r'(?:Format=(\d+)/10)?\s*\]',
            re.IGNORECASE
        )
    
    def extract(self, text: str) -> Dict[str, float]:
        """Extract self-ratings from text.
        
        Returns:
            Dict mapping dimension names to normalized scores (0-1).
        """
        matches = self.pattern.findall(text)
        if not matches:
            return {}
        
        last_match = matches[-1]
        result = {
            "consistency": float(last_match[0]) / 10,
            "completeness": float(last_match[1]) / 10,
            "accuracy": float(last_match[2]) / 10,
            "clarity": float(last_match[3]) / 10,
        }
        
        # Format is optional
        if len(last_match) > 4 and last_match[4]:
            result["format"] = float(last_match[4]) / 10
        else:
            result["format"] = 0.5  # Default neutral
        
        return result
    
    def has_self_rating(self, text: str) -> bool:
        """Check if text contains a self-rating."""
        return bool(self.pattern.search(text))


# Global extractor
SELF_RATING_EXTRACTOR = SelfRatingExtractor()

# ============================================
# CoR Intrinsic Reward Functions
# ============================================

def compute_consistency_score(text: str) -> float:
    """Evaluate logical consistency of reasoning.
    
    From theory.md: r_consistency(c) ∈ [0, 1]
    """
    score = 0.5  # Neutral baseline
    
    # Positive: logical flow markers
    flow_markers = ['therefore', 'thus', 'hence', 'so', 'because', 'since', 'given']
    for marker in flow_markers:
        if marker in text.lower():
            score += 0.1
    
    # Negative: contradiction indicators
    contradiction_patterns = [
        r'(?:wait|actually)[,\s]+(?:that\'s|this is)\s+(?:wrong|incorrect)',
        r'I made (?:a|an) (?:mistake|error)',
        r'(?:let me|I\'ll)\s+(?:reconsider|redo)',
    ]
    for pattern in contradiction_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score -= 0.1
    
    return max(0.0, min(1.0, score))


def compute_completeness_score(text: str) -> float:
    """Evaluate step comprehensiveness.
    
    From theory.md: r_completeness(c) ∈ [0, 1]
    """
    score = 0.0
    
    # Count step markers
    step_patterns = [
        r'step\s*\d+',
        r'first|second|third|finally',
        r'^\s*\d+[\.\)]\s+',
    ]
    
    step_count = 0
    for pattern in step_patterns:
        step_count += len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
    
    # 3+ steps = good, 5+ = excellent
    if step_count >= 3:
        score += 0.4
    if step_count >= 5:
        score += 0.2
    
    # Check for problem analysis
    if re.search(r'(?:given|known|we have|let)', text, re.IGNORECASE):
        score += 0.2
    
    # Check for verification
    if re.search(r'(?:verify|check|confirm|validate)', text, re.IGNORECASE):
        score += 0.2
    
    return min(1.0, score)


def compute_accuracy_score(text: str) -> float:
    """Evaluate mathematical/factual accuracy indicators.
    
    From theory.md: r_accuracy(c) ∈ [0, 1]
    """
    score = 0.5  # Neutral baseline
    
    # Check for mathematical expressions
    if re.search(r'\d+\s*[\+\-\*\/\=]\s*\d+', text):
        score += 0.15
    
    # Check for verification
    if re.search(r'(?:verify|check|substitut)', text, re.IGNORECASE):
        score += 0.15
    
    # Check for formula references
    if re.search(r'(?:formula|equation|theorem)', text, re.IGNORECASE):
        score += 0.1
    
    # Check for self-correction (awareness of errors)
    if re.search(r'(?:wait|actually)', text, re.IGNORECASE):
        score += 0.1
    
    return min(1.0, score)


def compute_clarity_score(text: str) -> float:
    """Evaluate reasoning clarity.
    
    From theory.md: r_clarity(c) ∈ [0, 1]
    """
    score = 0.5
    
    # Check for definitions
    if re.search(r'(?:let\s+\w+\s*=|define|denote)', text, re.IGNORECASE):
        score += 0.15
    
    # Check for structure (paragraphs)
    paragraphs = text.split('\n\n')
    if len(paragraphs) >= 2:
        score += 0.15
    
    # Check for clear section markers
    if re.search(r'(?:step\s*\d+|first|second|finally)', text, re.IGNORECASE):
        score += 0.1
    
    # Penalize very long unbroken text
    for para in paragraphs:
        if len(para) > 1000:
            score -= 0.1
    
    return max(0.0, min(1.0, score))


def compute_format_score(text: str, has_answer: bool = True) -> float:
    """Evaluate structural correctness.
    
    From theory.md: r_format(c) ∈ [0, 1]
    """
    score = 0.0
    
    # Check for self-rating presence (important for CoR)
    if SELF_RATING_EXTRACTOR.has_self_rating(text):
        score += 0.4
    
    # Check for answer marker
    if re.search(r'(?:final answer|answer is|therefore)', text, re.IGNORECASE):
        score += 0.3
    
    # Check for proper structure
    if '\n' in text:  # Has line breaks
        score += 0.15
    
    # Check for boxed answer
    if re.search(r'\\boxed\{[^}]+\}', text):
        score += 0.15
    
    return min(1.0, score)


def compute_calibration_score(
    self_ratings: Dict[str, float],
    actual_scores: Dict[str, float],
    config: CoRConfig = COR_CONFIG
) -> float:
    """Compute self-rating calibration quality.
    
    From theory.md:
    cal_d(u, v) = 1 - |u - v|
    r_self = (1/D) Σ cal_d(self_rating_d/10, actual_d)
    """
    if not self_ratings:
        return 0.0
    
    calibration_scores = []
    for dim in SelfRatingExtractor.DIMENSIONS:
        if dim in self_ratings and dim in actual_scores:
            u = self_ratings[dim]
            v = actual_scores[dim]
            cal = 1.0 - abs(u - v)
            
            # High-high alignment bonus
            if u > 0.8 and v > 0.8:
                cal += config.calibration_bonus
            
            calibration_scores.append(cal)
    
    if not calibration_scores:
        return 0.0
    
    return sum(calibration_scores) / len(calibration_scores)


# ============================================
# CoR Reward Functions for GRPO
# ============================================

def cor_intrinsic_reward_func(completions, **kwargs) -> List[float]:
    """Compute CoR intrinsic reward.
    
    From theory.md:
    R_int(c) = Σ_d w_d × r_d(y_think) + w_self × r_self_rating_quality
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        dim_scores = {
            "consistency": compute_consistency_score(response),
            "completeness": compute_completeness_score(response),
            "accuracy": compute_accuracy_score(response),
            "clarity": compute_clarity_score(response),
            "format": compute_format_score(response),
        }
        
        # Weighted intrinsic reward
        intrinsic = sum(
            COR_CONFIG.dimension_weights.get(dim, 0.2) * score
            for dim, score in dim_scores.items()
        )
        
        # Self-rating calibration bonus
        self_ratings = SELF_RATING_EXTRACTOR.extract(response)
        if self_ratings:
            calibration = compute_calibration_score(self_ratings, dim_scores)
            intrinsic += COR_CONFIG.self_rating_weight * calibration
        
        rewards.append(intrinsic)
    
    return rewards


def cor_self_rating_reward_func(completions, **kwargs) -> List[float]:
    """Reward for including self-ratings.
    
    Encourages the model to generate [Self-Rating: ...] tags.
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        if SELF_RATING_EXTRACTOR.has_self_rating(response):
            self_ratings = SELF_RATING_EXTRACTOR.extract(response)
            # Reward for having ratings, bonus for all 5 dimensions
            reward = 0.3
            if len(self_ratings) >= 5:
                reward += 0.2
            # Penalize extreme ratings (all 10/10 or all 1/10)
            if self_ratings:
                avg = sum(self_ratings.values()) / len(self_ratings)
                if 0.3 <= avg <= 0.9:
                    reward += 0.1
            rewards.append(reward)
        else:
            rewards.append(0.0)
    
    return rewards


def cor_external_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """Compute external (correctness) reward.
    
    From theory.md:
    R_ext(c) = I[y_answer == y_gt]
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for i, response in enumerate(responses):
        gt = answer[i] if i < len(answer) else None
        
        if gt is None:
            rewards.append(0.0)
            continue
        
        # Try to extract answer from response
        extracted = extract_final_answer(response)
        
        # Compare with ground truth
        if extracted and gt:
            if str(extracted).strip().lower() == str(gt).strip().lower():
                rewards.append(1.0)
            elif str(gt).strip() in response:
                rewards.append(0.8)  # Partial credit
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    
    return rewards


def cor_total_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """Compute total CoR reward.
    
    From theory.md:
    R(c) = R_ext(c) + λ × R_int(c)
    """
    r_ext = cor_external_reward_func(prompts, completions, answer, **kwargs)
    r_int = cor_intrinsic_reward_func(completions, **kwargs)
    
    total_rewards = []
    for ext, intr in zip(r_ext, r_int):
        total = ext + COR_CONFIG.lambda_intrinsic * intr
        total_rewards.append(total)
    
    # Log for debugging
    if kwargs.get("verbose", False):
        print(f"R_ext: {r_ext}")
        print(f"R_int: {r_int}")
        print(f"R_total: {total_rewards}")
    
    return total_rewards


# ============================================
# Helper Functions
# ============================================

def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer from response."""
    # Try boxed format
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Try "Final Answer:" format
    final_match = re.search(r'Final Answer[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()
    
    # Try "The answer is" format
    answer_match = re.search(r'(?:the )?answer is[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Try "####" format (GSM8K)
    if "####" in text:
        return text.split("####")[-1].strip()
    
    return None


def extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from GSM8K format (#### answer)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# ============================================
# Dataset Preparation
# ============================================

def get_gsm8k_dataset(split: str = "train", use_cor_prompt: bool = True) -> Dataset:
    """Load GSM8K dataset with CoR prompts."""
    data = load_dataset('openai/gsm8k', 'main')[split]
    
    system_prompt = COR_SYSTEM_PROMPT if use_cor_prompt else ""
    
    def format_example(x):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': x['question']})
        
        return {
            'prompt': messages,
            'answer': extract_hash_answer(x['answer'])
        }
    
    return data.map(format_example)


def get_math_dataset(split: str = "train", use_cor_prompt: bool = True) -> Dataset:
    """Load MATH dataset with CoR prompts."""
    # Note: MATH dataset may require different loading
    try:
        data = load_dataset('hendrycks/competition_math')[split]
        
        system_prompt = MATH_SYSTEM_PROMPT if use_cor_prompt else ""
        
        def format_example(x):
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': x['problem']})
            
            return {
                'prompt': messages,
                'answer': x.get('solution', '')
            }
        
        return data.map(format_example)
    except Exception as e:
        print(f"Error loading MATH dataset: {e}")
        return get_gsm8k_dataset(split, use_cor_prompt)


# ============================================
# Main Training Script
# ============================================

def main():
    """Main training function with CoR rewards."""
    
    # Check for Unsloth
    try:
        from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
        PatchFastRL("GRPO", FastLanguageModel)
    except ImportError:
        print("Unsloth not installed. Install with: pip install unsloth")
        print("Falling back to standard TRL...")
        return train_with_trl()
    
    # Model configuration
    max_seq_length = 2048  # Increase for longer reasoning
    lora_rank = 32
    
    print("=" * 60)
    print("CoR GRPO Training - Llama 3.1 8B")
    print("=" * 60)
    print(f"CoR Config: λ={COR_CONFIG.lambda_intrinsic}, μ={COR_CONFIG.improvement_weight}, ν={COR_CONFIG.convergence_weight}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = get_gsm8k_dataset(use_cor_prompt=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Training configuration
    from trl import GRPOConfig, GRPOTrainer
    
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,  # N=8 per theory.md
        max_prompt_length=512,
        max_completion_length=1024,
        max_steps=500,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="outputs/cor-grpo-llama",
    )
    
    # Initialize wandb
    try:
        import wandb
        wandb.init(project="CoR-GRPO-Llama", config={
            "lambda_intrinsic": COR_CONFIG.lambda_intrinsic,
            "improvement_weight": COR_CONFIG.improvement_weight,
            "convergence_weight": COR_CONFIG.convergence_weight,
            "model": "meta-llama/meta-Llama-3.1-8B-Instruct",
        })
    except:
        print("Wandb not available, skipping logging")
    
    # Create trainer with CoR rewards
    print("\nStarting CoR GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            cor_total_reward_func,      # Main CoR reward
            cor_self_rating_reward_func, # Bonus for self-ratings
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    output_dir = "outputs/cor-grpo-llama-final"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to Hub
    hub_model_id = os.environ.get("HF_MODEL_ID", "xingqiang/CoR-Llama3.1-8B")
    try:
        model.push_to_hub_merged(
            hub_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Model pushed to: https://huggingface.co/{hub_model_id}")
    except Exception as e:
        print(f"Failed to push to hub: {e}")
    
    print("\nTraining complete!")
    
    # Finish wandb
    try:
        wandb.finish()
    except:
        pass


def train_with_trl():
    """Fallback training with standard TRL (no Unsloth)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    
    print("Using standard TRL training...")
    
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = get_gsm8k_dataset(use_cor_prompt=True)
    
    training_args = GRPOConfig(
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_steps=100,
        output_dir="outputs/cor-grpo-llama-trl",
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[cor_total_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
