"""
Data utilities for CoR training.

Handles loading and formatting datasets for SFT and GRPO training.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from datasets import load_dataset, load_from_disk, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration for CoR training."""
    dataset_path: str = "local_data/s1K_cor_deepseek"
    use_hf_hub: bool = False
    max_length: int = 8192
    text_field: str = "text_cor"  # Field with formatted text
    thinking_field: str = "thinking_rated"  # Field with rated thinking
    question_field: str = "question"
    answer_field: str = "attempt"
    
    # For GRPO
    num_generations: int = 4
    prompt_template: str = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_cor_dataset(config: DataConfig) -> Dataset:
    """Load CoR dataset from local or HuggingFace Hub."""
    if config.use_hf_hub:
        logger.info(f"Loading from HuggingFace Hub: {config.dataset_path}")
        dataset = load_dataset(config.dataset_path, split='train')
    else:
        if os.path.isdir(config.dataset_path):
            logger.info(f"Loading from disk: {config.dataset_path}")
            dataset = load_from_disk(config.dataset_path)
        else:
            logger.info(f"Loading from HuggingFace: {config.dataset_path}")
            dataset = load_dataset(config.dataset_path, split='train')
    
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def extract_self_ratings(thinking: str) -> List[Dict[str, float]]:
    """Extract self-ratings from thinking text.
    
    Returns:
        List of dicts with dimension scores
    """
    ratings = []
    pattern = r'\[Self-Rating:\s*([^\]]+)\]'
    
    for match in re.finditer(pattern, thinking):
        rating_text = match.group(1)
        dims = {}
        
        for dim in ['Consistency', 'Completeness', 'Accuracy', 'Clarity']:
            dim_match = re.search(rf'{dim}=(\d+)/10', rating_text)
            if dim_match:
                dims[dim.lower()] = int(dim_match.group(1)) / 10
        
        if dims:
            ratings.append(dims)
    
    return ratings


def extract_overall_quality(thinking: str) -> Optional[float]:
    """Extract overall quality score from thinking text."""
    match = re.search(r'\[Overall Quality:\s*([\d.]+)/10\]', thinking)
    if match:
        return float(match.group(1)) / 10
    return None


def format_for_sft(
    example: Dict[str, Any],
    config: DataConfig
) -> Dict[str, str]:
    """Format example for SFT training.
    
    Returns dict with 'text' field ready for training.
    """
    # Use pre-formatted text if available
    if config.text_field in example and example[config.text_field]:
        return {"text": example[config.text_field]}
    
    # Otherwise format manually
    question = example.get(config.question_field, "")
    thinking = example.get(config.thinking_field, "")
    answer = example.get(config.answer_field, "") or example.get("solution", "")
    
    if not thinking:
        # Fall back to original thinking
        thinking_list = example.get("thinking_trajectories", [])
        thinking = thinking_list[0] if thinking_list else ""
    
    text = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<|im_start|>think\n{thinking}<|im_end|>\n"
        f"{answer}"
    )
    
    return {"text": text}


def format_for_grpo(
    example: Dict[str, Any],
    config: DataConfig
) -> Dict[str, Any]:
    """Format example for GRPO training.
    
    Returns dict with:
        - prompt: The formatted prompt
        - question: Original question
        - reference_answer: Ground truth answer
    """
    question = example.get(config.question_field, "")
    answer = example.get(config.answer_field, "") or example.get("solution", "")
    
    prompt = config.prompt_template.format(question=question)
    
    return {
        "prompt": prompt,
        "question": question,
        "reference_answer": answer,
        "metadata": example.get("metadata", {}),
        "source_type": example.get("source_type", ""),
    }


def prepare_sft_dataset(
    config: DataConfig,
    tokenizer=None,
    max_samples: Optional[int] = None
) -> Dataset:
    """Prepare dataset for SFT training.
    
    Args:
        config: Data configuration
        tokenizer: Optional tokenizer for length filtering
        max_samples: Optional max number of samples
    
    Returns:
        Dataset ready for SFTTrainer
    """
    dataset = load_cor_dataset(config)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Format for SFT
    dataset = dataset.map(
        lambda x: format_for_sft(x, config),
        remove_columns=[c for c in dataset.column_names if c != "text"]
    )
    
    # Filter by length if tokenizer provided
    if tokenizer:
        def filter_by_length(example):
            tokens = tokenizer.encode(example["text"])
            return len(tokens) <= config.max_length
        
        original_len = len(dataset)
        dataset = dataset.filter(filter_by_length)
        logger.info(f"Filtered {original_len} -> {len(dataset)} by length")
    
    return dataset


def prepare_grpo_dataset(
    config: DataConfig,
    max_samples: Optional[int] = None
) -> Dataset:
    """Prepare dataset for GRPO training.
    
    Returns dataset with prompts for generation.
    """
    dataset = load_cor_dataset(config)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Format for GRPO
    dataset = dataset.map(
        lambda x: format_for_grpo(x, config),
        remove_columns=dataset.column_names
    )
    
    return dataset


def get_quality_statistics(dataset: Dataset, thinking_field: str = "thinking_rated") -> Dict:
    """Compute quality statistics for dataset."""
    scores = []
    dim_scores = {
        'consistency': [],
        'completeness': [],
        'accuracy': [],
        'clarity': []
    }
    
    for example in dataset:
        thinking = example.get(thinking_field, "")
        
        # Overall quality
        overall = extract_overall_quality(thinking)
        if overall is not None:
            scores.append(overall)
        
        # Per-dimension
        ratings = extract_self_ratings(thinking)
        for rating in ratings:
            for dim, val in rating.items():
                if dim in dim_scores:
                    dim_scores[dim].append(val)
    
    stats = {}
    if scores:
        stats['overall'] = {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'count': len(scores)
        }
    
    for dim, values in dim_scores.items():
        if values:
            stats[dim] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    return stats


if __name__ == "__main__":
    # Test
    config = DataConfig(dataset_path="local_data/s1K_cor_deepseek_test2")
    
    try:
        dataset = load_cor_dataset(config)
        print(f"Loaded {len(dataset)} examples")
        
        # Test formatting
        if len(dataset) > 0:
            example = dict(dataset[0])
            sft_formatted = format_for_sft(example, config)
            grpo_formatted = format_for_grpo(example, config)
            
            print("\nSFT text (first 500 chars):")
            print(sft_formatted["text"][:500])
            
            print("\nGRPO prompt:")
            print(grpo_formatted["prompt"])
            
            # Quality stats
            stats = get_quality_statistics(dataset)
            print("\nQuality statistics:")
            for key, val in stats.items():
                print(f"  {key}: {val}")
    
    except Exception as e:
        print(f"Error: {e}")
