#!/usr/bin/env python3
"""
CoR Logic Validation Script.

Tests and validates the Chain of Reward implementation
with sample data to ensure correctness before training.

Usage:
    python train/validate_cor_logic.py
    python train/validate_cor_logic.py --dataset hf --samples 10
"""

import sys
import os
import argparse
import functools

# Force unbuffered output
print = functools.partial(print, flush=True)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk, load_dataset

from rewards import RewardCalculator, RewardConfig
from rewards.self_rating import SelfRatingExtractor
from rewards.intrinsic import IntrinsicRewardCalculator
from rewards.training_logger import CoRTrainingLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Validate CoR Logic")
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepseek",
        choices=["full", "deepseek", "hf"],
        help="Dataset to validate"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="xingqiang/s1K-cor-deepseek",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to validate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    return parser.parse_args()


def load_samples(args):
    """Load sample data for validation."""
    if args.dataset == "hf":
        print(f"\n📥 Loading dataset from HuggingFace: {args.hf_dataset}")
        dataset = load_dataset(args.hf_dataset, split="train")
    else:
        data_path = f"local_data/s1K_cor_{args.dataset}"
        print(f"\n📥 Loading dataset from: {data_path}")
        dataset = load_from_disk(data_path)
    
    return dataset.select(range(min(args.samples, len(dataset))))


def extract_thinking_from_text(text: str) -> str:
    """Extract thinking portion from formatted text."""
    import re
    
    # Try to find <|im_start|>think section
    match = re.search(
        r'<\|im_start\|>think\n(.*?)(?:<\|im_end\|>|$)',
        text,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    
    # Try to find thinking_rated field
    if "thinking_rated" in text:
        return text
    
    # Fallback: return full text
    return text


def validate_sample(
    sample: dict,
    calculator: RewardCalculator,
    extractor: SelfRatingExtractor,
    intrinsic_calc: IntrinsicRewardCalculator,
    logger: CoRTrainingLogger,
    sample_idx: int
):
    """Validate a single sample."""
    print("\n" + "="*80)
    print(f"🔬 VALIDATING SAMPLE {sample_idx}")
    print("="*80)
    
    # Extract thinking chain
    thinking = sample.get("thinking_rated") or sample.get("thinking_trajectories", [""])[0]
    if not thinking:
        thinking = extract_thinking_from_text(sample.get("text", "") or sample.get("text_cor", ""))
    
    question = sample.get("question", "")[:100] + "..."
    solution = sample.get("solution", "")[:100] + "..." if sample.get("solution") else "N/A"
    
    print(f"\n📝 Question: {question}")
    print(f"✅ Solution: {solution}")
    print(f"📏 Thinking length: {len(thinking)} chars")
    
    # 1. Self-Rating Extraction
    print("\n" + "-"*40)
    print("1️⃣  SELF-RATING EXTRACTION")
    print("-"*40)
    
    self_ratings = extractor.extract(thinking)
    
    if self_ratings:
        print(f"   ✅ Found {len(self_ratings)} self-ratings:")
        for dim, rating in self_ratings.items():
            print(f"      • {dim}: {rating.score}/10 (normalized: {rating.normalized:.2f})")
    else:
        print("   ⚠️  No self-ratings found in thinking chain")
        # Show a snippet to debug
        print(f"   [Debug] First 500 chars of thinking:")
        print(f"   {thinking[:500]}...")
    
    # 2. Intrinsic Dimension Scores
    print("\n" + "-"*40)
    print("2️⃣  INTRINSIC DIMENSION SCORES")
    print("-"*40)
    
    dim_scores = intrinsic_calc.compute_all_dimensions(thinking)
    
    for dim, score in dim_scores.items():
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"   {dim:12s}: [{bar}] {score:.3f}")
    
    # 3. Calibration Calculation
    print("\n" + "-"*40)
    print("3️⃣  SELF-RATING CALIBRATION")
    print("-"*40)
    
    if self_ratings:
        total_cal = 0.0
        count = 0
        for dim, rating in self_ratings.items():
            if dim in dim_scores:
                actual = dim_scores[dim]
                predicted = rating.normalized
                cal = 1.0 - abs(predicted - actual)
                
                arrow = "↗️" if predicted > actual else ("↘️" if predicted < actual else "➡️")
                print(f"   {dim}: self={predicted:.2f}, actual={actual:.2f}, cal={cal:.3f} {arrow}")
                
                total_cal += cal
                count += 1
        
        if count > 0:
            avg_cal = total_cal / count
            print(f"\n   📊 Average calibration: {avg_cal:.3f}")
            
            if avg_cal > 0.8:
                print("   🎯 Excellent calibration!")
            elif avg_cal > 0.6:
                print("   👍 Good calibration")
            elif avg_cal > 0.4:
                print("   ⚠️ Moderate calibration - needs improvement")
            else:
                print("   ❌ Poor calibration - model needs training")
    else:
        print("   ⚠️  Cannot calculate calibration without self-ratings")
    
    # 4. Total Reward Calculation
    print("\n" + "-"*40)
    print("4️⃣  TOTAL REWARD CALCULATION")
    print("-"*40)
    
    # Get ground truth answer (simplified - in real case would parse properly)
    ground_truth = sample.get("attempt", "") or sample.get("solution", "")
    answer = ground_truth  # For validation, assume correct
    
    output = calculator.calculate_total_reward(
        thinking_chain=thinking,
        answer=answer,
        ground_truth=ground_truth
    )
    
    print(f"\n   📈 REWARD BREAKDOWN:")
    print(f"      R_ext (external):     {output.external_reward:.4f}")
    print(f"      R_int (intrinsic):    {output.intrinsic_reward:.4f}")
    if output.improvement_reward != 0:
        print(f"      R_improve (reflect):  {output.improvement_reward:.4f}")
    if output.convergence_reward != 0:
        print(f"      R_converge (stable):  {output.convergence_reward:.4f}")
    print(f"      ─────────────────────────────")
    print(f"      R_total:              {output.total_reward:.4f}")
    
    # 5. Log using training logger
    logger.log_reward(
        step=sample_idx,
        sample_id=f"validation_sample_{sample_idx}",
        r_external=output.external_reward,
        r_intrinsic=output.intrinsic_reward,
        r_total=output.total_reward,
        dim_scores=output.dimension_scores,
        answer_correct=output.external_reward > 0.5,
        thinking_chain=thinking,
        self_ratings=self_ratings,
    )
    
    return {
        "has_self_rating": len(self_ratings) > 0,
        "num_dimensions": len(self_ratings),
        "r_external": output.external_reward,
        "r_intrinsic": output.intrinsic_reward,
        "r_total": output.total_reward,
        "calibration": avg_cal if self_ratings and count > 0 else 0.0,
    }


def run_validation(args):
    """Run full validation."""
    print("\n" + "="*80)
    print("🚀 CoR LOGIC VALIDATION")
    print("="*80)
    
    # Load samples
    samples = load_samples(args)
    print(f"   Loaded {len(samples)} samples for validation")
    
    # Initialize components
    config = RewardConfig(
        lambda_intrinsic=1.0,
        self_rating_weight=0.2,
        calibration_bonus=0.2,
    )
    calculator = RewardCalculator(config)
    extractor = SelfRatingExtractor()
    intrinsic_calc = IntrinsicRewardCalculator()
    logger = CoRTrainingLogger(log_every_n=1, verbose=False)
    
    # Validate each sample
    results = []
    for i, sample in enumerate(samples):
        result = validate_sample(
            sample=sample,
            calculator=calculator,
            extractor=extractor,
            intrinsic_calc=intrinsic_calc,
            logger=logger,
            sample_idx=i
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("📊 VALIDATION SUMMARY")
    print("="*80)
    
    n = len(results)
    self_rating_rate = sum(1 for r in results if r["has_self_rating"]) / n
    avg_r_total = sum(r["r_total"] for r in results) / n
    avg_r_int = sum(r["r_intrinsic"] for r in results) / n
    
    # Only calculate avg calibration for samples with ratings
    rated_samples = [r for r in results if r["has_self_rating"]]
    avg_calibration = sum(r["calibration"] for r in rated_samples) / len(rated_samples) if rated_samples else 0
    
    print(f"\n   Samples validated: {n}")
    print(f"\n   Self-Rating Detection:")
    print(f"      Rate: {self_rating_rate*100:.1f}%")
    print(f"      Avg dimensions per sample: {sum(r['num_dimensions'] for r in results)/n:.1f}")
    
    print(f"\n   Reward Statistics:")
    print(f"      Avg R_total: {avg_r_total:.4f}")
    print(f"      Avg R_intrinsic: {avg_r_int:.4f}")
    
    print(f"\n   Calibration Quality:")
    print(f"      Avg calibration: {avg_calibration:.4f}")
    
    # Validation status
    print("\n" + "-"*40)
    print("🔍 VALIDATION STATUS")
    print("-"*40)
    
    issues = []
    
    if self_rating_rate < 0.5:
        issues.append(f"⚠️ Low self-rating detection rate ({self_rating_rate*100:.1f}%)")
    
    if avg_calibration < 0.4 and rated_samples:
        issues.append(f"⚠️ Poor average calibration ({avg_calibration:.2f})")
    
    if avg_r_total < 0.3:
        issues.append(f"⚠️ Low average total reward ({avg_r_total:.2f})")
    
    if issues:
        print("\n   Issues found:")
        for issue in issues:
            print(f"      {issue}")
    else:
        print("\n   ✅ All checks passed!")
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80 + "\n")
    
    return len(issues) == 0


def main():
    args = parse_args()
    success = run_validation(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
