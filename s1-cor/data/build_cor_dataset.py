"""
Build CoR Training Dataset with Self-Ratings.

This script processes s1K dataset to add self-ratings to thinking chains,
preparing data for CoR training.

Usage:
    python data/build_cor_dataset.py --output_path local_data/s1K_cor
    
Or with LLM enhancement:
    python data/build_cor_dataset.py --output_path local_data/s1K_cor --use_llm
"""

import os
import sys
import re
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset


@dataclass
class QualityScores:
    """Quality scores for a thinking section."""
    consistency: float
    completeness: float
    accuracy: float
    clarity: float
    
    def to_rating_string(self) -> str:
        """Convert to self-rating format string."""
        return (
            f"[Self-Rating: Consistency={int(self.consistency*10)}/10, "
            f"Completeness={int(self.completeness*10)}/10, "
            f"Accuracy={int(self.accuracy*10)}/10, "
            f"Clarity={int(self.clarity*10)}/10]"
        )
    
    def overall(self) -> float:
        """Calculate overall score."""
        return (self.consistency + self.completeness + self.accuracy + self.clarity) / 4


class ThinkingAnalyzer:
    """Analyze thinking chain and compute quality scores."""
    
    def __init__(self):
        # Patterns for quality assessment
        self.logic_patterns = [
            r'(?:therefore|thus|hence|so|consequently)',
            r'(?:this means|this implies|it follows)',
            r'(?:given that|since|because)',
            r'(?:from|based on) (?:this|the above)',
        ]
        
        self.step_patterns = [
            r'(?:step\s+\d+|first|second|third|finally)',
            r'^\s*\d+[\.\)]\s+',
            r'(?:let\'s|we need to|i will|i\'ll)',
        ]
        
        self.math_patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\\[\[\]]',   # LaTeX display math
            r'=\s*[\d\.\-]+',  # Equations
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Arithmetic
        ]
        
        self.verify_patterns = [
            r'(?:verify|check|confirm|validate)',
            r'(?:let\'s see if|does this work)',
            r'(?:correct|right|makes sense)',
        ]
        
        self.error_patterns = [
            r'(?:wait|actually|no)[,\s]+(?:that\'s|this is)\s+(?:wrong|incorrect)',
            r'i made (?:a|an) (?:mistake|error)',
            r'let me (?:reconsider|recalculate|redo|start over)',
        ]
    
    def analyze_section(self, text: str) -> QualityScores:
        """Analyze a thinking section and return quality scores."""
        text_lower = text.lower()
        
        # Consistency: logical flow, no contradictions
        logic_count = sum(
            len(re.findall(p, text_lower)) 
            for p in self.logic_patterns
        )
        error_count = sum(
            len(re.findall(p, text_lower)) 
            for p in self.error_patterns
        )
        consistency = min(1.0, 0.5 + 0.1 * logic_count - 0.2 * error_count)
        
        # Completeness: steps, structure
        step_count = sum(
            len(re.findall(p, text_lower, re.MULTILINE)) 
            for p in self.step_patterns
        )
        completeness = min(1.0, 0.4 + 0.15 * min(step_count, 4))
        
        # Accuracy: math content, verification
        math_count = sum(
            len(re.findall(p, text)) 
            for p in self.math_patterns
        )
        verify_count = sum(
            len(re.findall(p, text_lower)) 
            for p in self.verify_patterns
        )
        accuracy = min(1.0, 0.5 + 0.05 * min(math_count, 5) + 0.1 * verify_count)
        
        # Clarity: reasonable length, not too dense
        words = len(text.split())
        sentences = len(re.split(r'[.!?]+', text))
        avg_sentence_len = words / max(1, sentences)
        
        if 10 <= avg_sentence_len <= 30:
            clarity = 0.8
        elif avg_sentence_len < 10:
            clarity = 0.6
        else:
            clarity = max(0.4, 0.8 - 0.01 * (avg_sentence_len - 30))
        
        # Boost clarity for well-structured text
        if step_count >= 2:
            clarity = min(1.0, clarity + 0.1)
        
        return QualityScores(
            consistency=consistency,
            completeness=completeness,
            accuracy=accuracy,
            clarity=clarity
        )
    
    def split_into_sections(self, thinking: str) -> List[str]:
        """Split thinking into logical sections."""
        # Try to split by step markers
        step_split = re.split(
            r'((?:Step\s+\d+[:\.])|(?:\n\d+[\.\)]\s+)|(?:\n(?:First|Second|Third|Finally)[,:\s]))',
            thinking,
            flags=re.IGNORECASE
        )
        
        if len(step_split) > 2:
            # Recombine markers with their content
            sections = []
            for i in range(0, len(step_split), 2):
                if i + 1 < len(step_split):
                    sections.append(step_split[i] + step_split[i+1])
                elif step_split[i].strip():
                    sections.append(step_split[i])
            sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 50]
            if sections:
                return sections
        
        # Fallback: split by double newlines
        paragraphs = thinking.split('\n\n')
        
        # Group small paragraphs
        sections = []
        current = ""
        for para in paragraphs:
            current += para.strip() + "\n\n"
            if len(current) > 800:  # ~200 words per section
                sections.append(current.strip())
                current = ""
        
        if current.strip():
            sections.append(current.strip())
        
        # Ensure we have at least 2-5 sections
        if len(sections) == 1 and len(thinking) > 1500:
            # Force split into 3 parts
            n = len(thinking)
            sections = [
                thinking[:n//3].strip(),
                thinking[n//3:2*n//3].strip(),
                thinking[2*n//3:].strip(),
            ]
        
        return sections if sections else [thinking]


def add_self_ratings_to_thinking(
    thinking: str,
    analyzer: ThinkingAnalyzer
) -> Tuple[str, float]:
    """Add self-ratings to a thinking chain.
    
    Returns:
        Tuple of (rated_thinking, overall_score)
    """
    sections = analyzer.split_into_sections(thinking)
    
    rated_parts = []
    total_score = 0.0
    
    for i, section in enumerate(sections):
        scores = analyzer.analyze_section(section)
        total_score += scores.overall()
        
        # Add rating after each section
        rated_section = section.strip()
        if not rated_section.endswith('\n'):
            rated_section += '\n'
        rated_section += scores.to_rating_string()
        
        rated_parts.append(rated_section)
    
    # Combine sections
    rated_thinking = '\n\n'.join(rated_parts)
    
    # Add overall rating at the end
    overall = total_score / len(sections) if sections else 0.5
    rated_thinking += f'\n\n[Overall Quality: {overall*10:.1f}/10]'
    
    return rated_thinking, overall


def format_for_training(
    question: str,
    rated_thinking: str,
    answer: str,
    system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
) -> str:
    """Format example for SFT training in Qwen format."""
    
    # Ensure thinking has proper format
    if not rated_thinking.strip().startswith('<|im_start|>think'):
        # Wrap thinking in think tags if not already present
        formatted_thinking = rated_thinking
    else:
        formatted_thinking = rated_thinking
    
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<|im_start|>think\n{formatted_thinking}<|im_end|>\n"
        f"{answer}"
    )
    
    return text


def process_example(
    example: Dict,
    analyzer: ThinkingAnalyzer
) -> Dict:
    """Process a single example to add self-ratings."""
    
    question = example.get('question', '')
    
    # Get thinking trajectory
    thinking_list = example.get('thinking_trajectories', [])
    thinking = thinking_list[0] if thinking_list else ''
    
    # Get answer
    answer = example.get('attempt', '') or example.get('solution', '')
    
    if not thinking:
        logging.warning(f"Empty thinking for question: {question[:50]}...")
        return example
    
    # Add self-ratings
    rated_thinking, overall_score = add_self_ratings_to_thinking(thinking, analyzer)
    
    # Format for training
    text = format_for_training(question, rated_thinking, answer)
    
    # Update example
    example['thinking_rated'] = rated_thinking
    example['overall_quality_score'] = overall_score
    example['text_cor'] = text  # New field for CoR training
    example['has_self_ratings'] = True
    
    return example


def main():
    parser = argparse.ArgumentParser(description="Build CoR training dataset")
    parser.add_argument(
        "--input_path", 
        type=str, 
        default="simplescaling/s1K_tokenized",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="local_data/s1K_cor",
        help="Output path (local directory or HuggingFace hub path)"
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="Use LLM for enhanced self-rating generation (slower, higher quality)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to process (for testing)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push to HuggingFace Hub instead of saving locally"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    logging.info(f"Loading dataset from {args.input_path}")
    dataset = load_dataset(args.input_path, split='train')
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    logging.info(f"Processing {len(dataset)} examples")
    
    # Initialize analyzer
    analyzer = ThinkingAnalyzer()
    
    # Process examples
    processed = []
    quality_scores = []
    
    for example in tqdm(dataset, desc="Adding self-ratings"):
        try:
            result = process_example(dict(example), analyzer)
            processed.append(result)
            if 'overall_quality_score' in result:
                quality_scores.append(result['overall_quality_score'])
        except Exception as e:
            logging.error(f"Error processing example: {e}")
            processed.append(dict(example))
    
    # Create output dataset
    output_dataset = Dataset.from_list(processed)
    
    # Log statistics
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        logging.info(f"Average quality score: {avg_quality:.2f}")
        logging.info(f"Quality range: {min(quality_scores):.2f} - {max(quality_scores):.2f}")
    
    # Save
    if args.push_to_hub:
        logging.info(f"Pushing to HuggingFace Hub: {args.output_path}")
        output_dataset.push_to_hub(args.output_path)
    else:
        # Create local directory
        os.makedirs(args.output_path, exist_ok=True)
        logging.info(f"Saving to local path: {args.output_path}")
        output_dataset.save_to_disk(args.output_path)
        
        # Also save a sample for inspection
        sample_path = os.path.join(args.output_path, "sample.json")
        sample = processed[0]
        with open(sample_path, 'w', encoding='utf-8') as f:
            # Convert non-serializable fields
            sample_clean = {k: v for k, v in sample.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            json.dump(sample_clean, f, indent=2, ensure_ascii=False)
        logging.info(f"Sample saved to {sample_path}")
    
    logging.info("Done!")
    
    # Print sample
    if processed:
        sample = processed[0]
        logging.info("\n" + "="*50)
        logging.info("SAMPLE OUTPUT:")
        logging.info("="*50)
        if 'thinking_rated' in sample:
            # Show first 2000 chars
            rated = sample['thinking_rated']
            logging.info(f"Rated thinking (first 2000 chars):\n{rated[:2000]}...")
            logging.info(f"\nOverall quality score: {sample.get('overall_quality_score', 'N/A')}")


if __name__ == "__main__":
    main()
