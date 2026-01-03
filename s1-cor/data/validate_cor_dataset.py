"""
Validate CoR Dataset for Training.

Checks:
1. Self-rating format correctness
2. Quality score distribution
3. Token length statistics
4. Missing data detection
"""

import os
import sys
import re
import json
import logging
import argparse
from typing import Dict, List
from collections import Counter
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_from_disk, load_dataset


@dataclass
class ValidationResult:
    """Validation results."""
    total_examples: int = 0
    has_thinking: int = 0
    has_ratings: int = 0
    has_overall: int = 0
    valid_format: int = 0
    
    avg_quality: float = 0.0
    min_quality: float = 1.0
    max_quality: float = 0.0
    
    avg_thinking_length: float = 0.0
    avg_rating_count: float = 0.0
    
    dimension_stats: Dict = None
    errors: List[str] = None
    
    def __post_init__(self):
        self.dimension_stats = {}
        self.errors = []
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "CoR DATASET VALIDATION REPORT",
            "=" * 50,
            f"Total examples: {self.total_examples}",
            f"Has thinking: {self.has_thinking} ({100*self.has_thinking/max(1,self.total_examples):.1f}%)",
            f"Has self-ratings: {self.has_ratings} ({100*self.has_ratings/max(1,self.total_examples):.1f}%)",
            f"Has overall score: {self.has_overall} ({100*self.has_overall/max(1,self.total_examples):.1f}%)",
            f"Valid format: {self.valid_format} ({100*self.valid_format/max(1,self.total_examples):.1f}%)",
            "",
            "Quality Statistics:",
            f"  Average: {self.avg_quality:.2f}",
            f"  Range: [{self.min_quality:.2f}, {self.max_quality:.2f}]",
            "",
            f"Average thinking length: {self.avg_thinking_length:.0f} chars",
            f"Average ratings per example: {self.avg_rating_count:.1f}",
            "",
        ]
        
        if self.dimension_stats:
            lines.append("Dimension Statistics (average):")
            for dim, avg in self.dimension_stats.items():
                lines.append(f"  {dim}: {avg:.2f}/10")
        
        if self.errors:
            lines.append("")
            lines.append(f"Errors found ({len(self.errors)}):")
            for err in self.errors[:5]:
                lines.append(f"  - {err}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors)-5} more")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def validate_example(example: Dict, idx: int) -> Dict:
    """Validate a single example."""
    issues = []
    stats = {
        'has_thinking': False,
        'has_ratings': False,
        'has_overall': False,
        'valid_format': False,
        'quality_score': None,
        'thinking_length': 0,
        'rating_count': 0,
        'dimensions': {}
    }
    
    # Check thinking
    thinking = example.get('thinking_rated', '')
    if not thinking:
        thinking_list = example.get('thinking_trajectories', [])
        thinking = thinking_list[0] if thinking_list else ''
    
    if thinking:
        stats['has_thinking'] = True
        stats['thinking_length'] = len(thinking)
    else:
        issues.append(f"Example {idx}: No thinking content")
        return stats, issues
    
    # Check self-ratings
    ratings = re.findall(r'\[Self-Rating:([^\]]+)\]', thinking)
    stats['rating_count'] = len(ratings)
    
    if ratings:
        stats['has_ratings'] = True
        
        # Parse dimensions
        all_dims = {
            'Consistency': [],
            'Completeness': [],
            'Accuracy': [],
            'Clarity': []
        }
        
        for rating in ratings:
            for dim in all_dims:
                match = re.search(rf'{dim}=(\d+)/10', rating)
                if match:
                    all_dims[dim].append(int(match.group(1)))
        
        # Compute averages
        for dim, values in all_dims.items():
            if values:
                stats['dimensions'][dim] = sum(values) / len(values)
        
        stats['valid_format'] = len(stats['dimensions']) == 4
    else:
        issues.append(f"Example {idx}: No self-ratings found")
    
    # Check overall score
    overall_match = re.search(r'\[Overall Quality:\s*([\d.]+)/10\]', thinking)
    if overall_match:
        stats['has_overall'] = True
        stats['quality_score'] = float(overall_match.group(1)) / 10
    elif 'overall_quality_score' in example:
        stats['has_overall'] = True
        stats['quality_score'] = example['overall_quality_score']
    
    return stats, issues


def validate_dataset(dataset_path: str) -> ValidationResult:
    """Validate a CoR dataset."""
    
    # Load dataset
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path, split='train')
    
    result = ValidationResult()
    result.total_examples = len(dataset)
    
    quality_scores = []
    thinking_lengths = []
    rating_counts = []
    dimension_values = {
        'Consistency': [],
        'Completeness': [],
        'Accuracy': [],
        'Clarity': []
    }
    
    for idx, example in enumerate(dataset):
        stats, issues = validate_example(dict(example), idx)
        
        if stats['has_thinking']:
            result.has_thinking += 1
            thinking_lengths.append(stats['thinking_length'])
        
        if stats['has_ratings']:
            result.has_ratings += 1
            rating_counts.append(stats['rating_count'])
        
        if stats['has_overall']:
            result.has_overall += 1
        
        if stats['valid_format']:
            result.valid_format += 1
        
        if stats['quality_score'] is not None:
            quality_scores.append(stats['quality_score'])
        
        for dim, val in stats['dimensions'].items():
            dimension_values[dim].append(val)
        
        result.errors.extend(issues)
    
    # Compute aggregates
    if quality_scores:
        result.avg_quality = sum(quality_scores) / len(quality_scores)
        result.min_quality = min(quality_scores)
        result.max_quality = max(quality_scores)
    
    if thinking_lengths:
        result.avg_thinking_length = sum(thinking_lengths) / len(thinking_lengths)
    
    if rating_counts:
        result.avg_rating_count = sum(rating_counts) / len(rating_counts)
    
    for dim, values in dimension_values.items():
        if values:
            result.dimension_stats[dim] = sum(values) / len(values)
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="local_data/s1K_cor_full",
        help="Path to CoR dataset"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional JSON output path"
    )
    
    args = parser.parse_args()
    
    logging.info(f"Validating dataset: {args.dataset_path}")
    result = validate_dataset(args.dataset_path)
    
    print(result.summary())
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({
                'total_examples': result.total_examples,
                'has_ratings': result.has_ratings,
                'valid_format': result.valid_format,
                'avg_quality': result.avg_quality,
                'dimension_stats': result.dimension_stats,
                'error_count': len(result.errors)
            }, f, indent=2)
        logging.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
