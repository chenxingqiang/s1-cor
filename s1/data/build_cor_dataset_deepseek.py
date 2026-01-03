"""
Build CoR Training Dataset with DeepSeek Reasoning Model.

Uses DeepSeek-R1 or DeepSeek-Chat for high-quality self-rating generation.

Usage:
    python data/build_cor_dataset_deepseek.py --output_path local_data/s1K_cor_deepseek
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset
from openai import OpenAI


# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


SELF_RATING_PROMPT = """你是一位数学推理质量评估专家。请评估以下推理片段的质量。

## 推理片段:
{thinking_segment}

## 问题背景:
{question}

## 评估任务:
请从以下四个维度评分（0-10分）：

1. **Consistency（一致性）**: 推理是否逻辑一致？有无矛盾或错误？
2. **Completeness（完整性）**: 是否覆盖了必要的步骤？有无遗漏？
3. **Accuracy（准确性）**: 数学运算和结论是否正确？
4. **Clarity（清晰度）**: 推理表达是否清晰易懂？

## 输出格式（严格遵循）:
[Self-Rating: Consistency=X/10, Completeness=X/10, Accuracy=X/10, Clarity=X/10]

重要：只输出上述格式的评分，不要输出其他内容。"""


class DeepSeekRater:
    """DeepSeek-based quality rater."""
    
    def __init__(
        self, 
        model: str = "deepseek-reasoner",  # or "deepseek-chat"
        max_retries: int = 3
    ):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.model = model
        self.max_retries = max_retries
    
    def rate_segment(
        self, 
        thinking_segment: str, 
        question: str
    ) -> Optional[str]:
        """Rate a thinking segment using DeepSeek."""
        prompt = SELF_RATING_PROMPT.format(
            thinking_segment=thinking_segment[:3000],
            question=question[:500]
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.3
                )
                
                result = response.choices[0].message.content.strip()
                
                # Extract rating
                match = re.search(r'\[Self-Rating:[^\]]+\]', result)
                if match:
                    return match.group(0)
                
                # Try to parse even if format is slightly different
                if 'Consistency=' in result and 'Accuracy=' in result:
                    # Reconstruct format
                    dims = {}
                    for dim in ['Consistency', 'Completeness', 'Accuracy', 'Clarity']:
                        m = re.search(rf'{dim}\s*[=:]\s*(\d+)', result)
                        if m:
                            dims[dim] = m.group(1)
                    
                    if len(dims) == 4:
                        return (f"[Self-Rating: Consistency={dims['Consistency']}/10, "
                               f"Completeness={dims['Completeness']}/10, "
                               f"Accuracy={dims['Accuracy']}/10, "
                               f"Clarity={dims['Clarity']}/10]")
                
                logging.warning(f"Invalid format: {result[:100]}")
                
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        
        return None


def split_thinking(thinking: str, max_segments: int = 5) -> List[str]:
    """Split thinking into segments."""
    # Step-based split
    step_pattern = r'((?:Step\s+\d+[:\.])|(?:\n\d+[\.\)]\s+))'
    parts = re.split(step_pattern, thinking, flags=re.IGNORECASE)
    
    if len(parts) > 2:
        segments = []
        for i in range(0, len(parts), 2):
            seg = parts[i] if i+1 >= len(parts) else parts[i] + parts[i+1]
            if seg.strip() and len(seg.strip()) > 100:
                segments.append(seg.strip())
        if segments:
            return segments[:max_segments]
    
    # Paragraph split
    paragraphs = thinking.split('\n\n')
    segments = []
    current = ""
    
    for para in paragraphs:
        current += para.strip() + "\n\n"
        if len(current) > 1200:
            segments.append(current.strip())
            current = ""
    
    if current.strip():
        segments.append(current.strip())
    
    # Limit segments
    if len(segments) > max_segments:
        # Merge some segments
        merged = []
        chunk_size = len(segments) // max_segments + 1
        for i in range(0, len(segments), chunk_size):
            merged.append('\n\n'.join(segments[i:i+chunk_size]))
        segments = merged
    
    if not segments:
        # Force split
        n = len(thinking)
        segments = [
            thinking[:n//3].strip(),
            thinking[n//3:2*n//3].strip(),
            thinking[2*n//3:].strip()
        ]
    
    return segments


def add_ratings_with_deepseek(
    thinking: str,
    question: str,
    rater: DeepSeekRater
) -> Tuple[str, float]:
    """Add DeepSeek-generated ratings to thinking."""
    segments = split_thinking(thinking)
    rated_segments = []
    scores = []
    
    for segment in segments:
        rating = rater.rate_segment(segment, question)
        
        if rating:
            rated_segments.append(f"{segment}\n{rating}")
            # Parse scores
            dims = {}
            for dim in ['Consistency', 'Completeness', 'Accuracy', 'Clarity']:
                m = re.search(rf'{dim}=(\d+)', rating)
                if m:
                    dims[dim] = int(m.group(1))
            if dims:
                scores.append(sum(dims.values()) / len(dims) / 10)
        else:
            # Fallback: simple heuristic
            rated_segments.append(segment)
            logging.warning("Using fallback for segment")
    
    rated_thinking = '\n\n'.join(rated_segments)
    overall = sum(scores) / len(scores) if scores else 0.5
    rated_thinking += f'\n\n[Overall Quality: {overall*10:.1f}/10]'
    
    return rated_thinking, overall


def format_for_training(
    question: str,
    rated_thinking: str,
    answer: str,
    system_prompt: str = "You are a helpful assistant."
) -> str:
    """Format for SFT training."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<|im_start|>think\n{rated_thinking}<|im_end|>\n"
        f"{answer}"
    )


def process_example(
    example: Dict,
    rater: DeepSeekRater
) -> Dict:
    """Process a single example."""
    question = example.get('question', '')
    thinking_list = example.get('thinking_trajectories', [])
    thinking = thinking_list[0] if thinking_list else ''
    answer = example.get('attempt', '') or example.get('solution', '')
    
    if not thinking:
        return example
    
    try:
        rated_thinking, score = add_ratings_with_deepseek(thinking, question, rater)
        
        example['thinking_rated'] = rated_thinking
        example['overall_quality_score'] = score
        example['text_cor'] = format_for_training(question, rated_thinking, answer)
        example['has_self_ratings'] = True
        example['rating_method'] = 'deepseek'
        
    except Exception as e:
        logging.error(f"Error processing: {e}")
    
    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="simplescaling/s1K_tokenized")
    parser.add_argument("--output_path", default="local_data/s1K_cor_deepseek")
    parser.add_argument("--model", default="deepseek-chat", 
                       help="deepseek-chat (recommended) or deepseek-reasoner")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--rate_limit", type=float, default=0.5,
                       help="Seconds between API calls")
    
    args = parser.parse_args()
    
    if not DEEPSEEK_API_KEY:
        logging.error("DEEPSEEK_API_KEY not found in environment")
        sys.exit(1)
    
    # Load dataset
    logging.info(f"Loading from {args.input_path}")
    dataset = load_dataset(args.input_path, split='train')
    
    # Select range
    end_idx = args.start_idx + args.max_samples if args.max_samples else len(dataset)
    dataset = dataset.select(range(args.start_idx, min(end_idx, len(dataset))))
    
    logging.info(f"Processing {len(dataset)} examples with DeepSeek {args.model}")
    
    # Init rater
    rater = DeepSeekRater(model=args.model)
    
    # Process
    processed = []
    quality_scores = []
    
    for example in tqdm(dataset, desc="DeepSeek Rating"):
        result = process_example(dict(example), rater)
        processed.append(result)
        
        if 'overall_quality_score' in result:
            quality_scores.append(result['overall_quality_score'])
        
        time.sleep(args.rate_limit)  # Rate limiting
    
    # Save
    output_dataset = Dataset.from_list(processed)
    os.makedirs(args.output_path, exist_ok=True)
    output_dataset.save_to_disk(args.output_path)
    
    # Stats
    if quality_scores:
        logging.info(f"Average quality: {sum(quality_scores)/len(quality_scores):.2f}")
        logging.info(f"Quality range: [{min(quality_scores):.2f}, {max(quality_scores):.2f}]")
    
    # Save sample
    if processed:
        sample_path = os.path.join(args.output_path, "sample.json")
        sample = {k: v for k, v in processed[0].items() 
                  if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
