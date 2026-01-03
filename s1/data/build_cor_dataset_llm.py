"""
Build CoR Training Dataset with LLM-Enhanced Self-Ratings.

This script uses an LLM to generate high-quality self-ratings for thinking chains,
providing more accurate quality assessments for CoR training.

Usage:
    # Using Gemini (default)
    python data/build_cor_dataset_llm.py --output_path local_data/s1K_cor_llm
    
    # Using OpenAI
    OPENAI_API_KEY=xxx python data/build_cor_dataset_llm.py --llm_provider openai
"""

import os
import sys
import re
import json
import logging
import argparse
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset


SELF_RATING_PROMPT = """You are an expert evaluator of mathematical reasoning. Given a segment of thinking/reasoning, provide a quality assessment.

## THINKING SEGMENT:
{thinking_segment}

## QUESTION CONTEXT:
{question}

## EVALUATION TASK:
Rate this thinking segment on the following dimensions (0-10 scale):

1. **Consistency** (0-10): Is the reasoning logically consistent? Are there contradictions or errors?
2. **Completeness** (0-10): Does the segment cover the necessary steps? Are there gaps?
3. **Accuracy** (0-10): Are the mathematical operations and conclusions correct?
4. **Clarity** (0-10): Is the reasoning clearly expressed and easy to follow?

## OUTPUT FORMAT (strictly follow this format):
[Self-Rating: Consistency=X/10, Completeness=X/10, Accuracy=X/10, Clarity=X/10]

IMPORTANT: Output ONLY the rating in the exact format above, nothing else."""


@dataclass  
class LLMConfig:
    """LLM configuration."""
    provider: str = "gemini"
    model: str = None
    api_key: str = None
    max_retries: int = 3
    timeout: float = 30.0
    
    def __post_init__(self):
        if self.provider == "gemini":
            self.model = self.model or "gemini-2.0-flash-exp"
            self.api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        elif self.provider == "openai":
            self.model = self.model or "gpt-4o-mini"
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")


class LLMRater:
    """LLM-based quality rater."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._init_client()
    
    def _init_client(self):
        if self.config.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self.model = genai.GenerativeModel(self.config.model)
        elif self.config.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
    
    def rate_segment(
        self, 
        thinking_segment: str, 
        question: str
    ) -> Optional[str]:
        """Rate a thinking segment using LLM."""
        prompt = SELF_RATING_PROMPT.format(
            thinking_segment=thinking_segment[:2000],  # Limit length
            question=question[:500]
        )
        
        for attempt in range(self.config.max_retries):
            try:
                if self.config.provider == "gemini":
                    response = self.model.generate_content(prompt)
                    result = response.text.strip()
                elif self.config.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100,
                        temperature=0.3
                    )
                    result = response.choices[0].message.content.strip()
                
                # Validate format
                if re.match(r'\[Self-Rating:.*?\]', result):
                    return result
                else:
                    # Try to extract from response
                    match = re.search(r'\[Self-Rating:.*?\]', result)
                    if match:
                        return match.group(0)
                    logging.warning(f"Invalid rating format: {result[:100]}")
                    
            except Exception as e:
                logging.warning(f"LLM rating attempt {attempt+1} failed: {e}")
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return None


def split_thinking(thinking: str) -> List[str]:
    """Split thinking into segments for rating."""
    # Try step-based split
    step_pattern = r'((?:Step\s+\d+[:\.])|(?:\n\d+[\.\)]\s+)|(?:\n(?:First|Second|Third|Finally)[,:\s]))'
    parts = re.split(step_pattern, thinking, flags=re.IGNORECASE)
    
    if len(parts) > 2:
        segments = []
        for i in range(0, len(parts), 2):
            seg = parts[i] if i+1 >= len(parts) else parts[i] + parts[i+1]
            if seg.strip() and len(seg.strip()) > 100:
                segments.append(seg.strip())
        if segments:
            return segments
    
    # Paragraph-based split
    paragraphs = thinking.split('\n\n')
    segments = []
    current = ""
    
    for para in paragraphs:
        current += para.strip() + "\n\n"
        if len(current) > 1000:
            segments.append(current.strip())
            current = ""
    
    if current.strip():
        segments.append(current.strip())
    
    # Ensure reasonable number of segments
    if len(segments) == 1 and len(thinking) > 2000:
        n = len(thinking)
        segments = [
            thinking[:n//3].strip(),
            thinking[n//3:2*n//3].strip(),
            thinking[2*n//3:].strip()
        ]
    
    return segments if segments else [thinking]


def add_ratings_with_llm(
    thinking: str,
    question: str,
    rater: LLMRater,
    use_fallback: bool = True
) -> Tuple[str, float]:
    """Add LLM-generated ratings to thinking."""
    from build_cor_dataset import ThinkingAnalyzer
    
    segments = split_thinking(thinking)
    rated_segments = []
    scores = []
    
    fallback_analyzer = ThinkingAnalyzer() if use_fallback else None
    
    for segment in segments:
        rating = rater.rate_segment(segment, question)
        
        if rating:
            rated_segments.append(f"{segment}\n{rating}")
            # Parse score
            match = re.search(r'Consistency=(\d+)', rating)
            if match:
                scores.append(float(match.group(1)) / 10)
        else:
            # Fallback to rule-based
            if fallback_analyzer:
                fb_scores = fallback_analyzer.analyze_section(segment)
                rating = fb_scores.to_rating_string()
                rated_segments.append(f"{segment}\n{rating}")
                scores.append(fb_scores.overall())
            else:
                rated_segments.append(segment)
    
    rated_thinking = '\n\n'.join(rated_segments)
    overall = sum(scores) / len(scores) if scores else 0.5
    rated_thinking += f'\n\n[Overall Quality: {overall*10:.1f}/10]'
    
    return rated_thinking, overall


def process_batch(
    examples: List[Dict],
    rater: LLMRater,
    batch_size: int = 5
) -> List[Dict]:
    """Process a batch of examples."""
    results = []
    
    for example in examples:
        question = example.get('question', '')
        thinking_list = example.get('thinking_trajectories', [])
        thinking = thinking_list[0] if thinking_list else ''
        
        if not thinking:
            results.append(example)
            continue
        
        try:
            rated_thinking, score = add_ratings_with_llm(
                thinking, question, rater
            )
            example['thinking_rated'] = rated_thinking
            example['overall_quality_score'] = score
            example['has_self_ratings'] = True
            example['rating_method'] = 'llm'
        except Exception as e:
            logging.error(f"Error: {e}")
        
        results.append(example)
        time.sleep(0.1)  # Rate limiting
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="simplescaling/s1K_tokenized")
    parser.add_argument("--output_path", default="local_data/s1K_cor_llm")
    parser.add_argument("--llm_provider", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Load dataset
    logging.info(f"Loading from {args.input_path}")
    dataset = load_dataset(args.input_path, split='train')
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    # Init LLM rater
    config = LLMConfig(provider=args.llm_provider)
    if not config.api_key:
        logging.error(f"No API key found for {args.llm_provider}")
        logging.info("Falling back to rule-based rating")
        # Fall back to rule-based
        os.system(f"python data/build_cor_dataset.py --output_path {args.output_path}")
        return
    
    rater = LLMRater(config)
    
    # Process
    logging.info(f"Processing {len(dataset)} examples with LLM rating")
    
    all_examples = [dict(ex) for ex in dataset]
    processed = []
    
    for example in tqdm(all_examples, desc="LLM Rating"):
        try:
            question = example.get('question', '')
            thinking_list = example.get('thinking_trajectories', [])
            thinking = thinking_list[0] if thinking_list else ''
            
            if thinking:
                rated, score = add_ratings_with_llm(thinking, question, rater)
                example['thinking_rated'] = rated
                example['overall_quality_score'] = score
                example['has_self_ratings'] = True
                example['rating_method'] = 'llm'
            
            processed.append(example)
        except Exception as e:
            logging.error(f"Error: {e}")
            processed.append(example)
    
    # Save
    output_dataset = Dataset.from_list(processed)
    os.makedirs(args.output_path, exist_ok=True)
    output_dataset.save_to_disk(args.output_path)
    logging.info(f"Saved to {args.output_path}")
    
    # Stats
    scores = [e['overall_quality_score'] for e in processed if 'overall_quality_score' in e]
    if scores:
        logging.info(f"Average quality: {sum(scores)/len(scores):.2f}")


if __name__ == "__main__":
    main()
