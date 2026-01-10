"""
Training Logger for Chain of Reward (CoR).

Provides detailed logging of intermediate results during training
for logic validation and model effect tracking.

Logs:
- Reward components (R_ext, R_int, R_improve, R_converge)
- Self-rating extraction and calibration
- Per-dimension quality scores
- Training progress metrics
"""

import sys
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import functools

# Force unbuffered output
print = functools.partial(print, flush=True)


@dataclass
class RewardLogEntry:
    """A single reward calculation log entry."""
    step: int
    sample_id: str
    
    # Reward components
    r_external: float
    r_intrinsic: float
    r_improve: float = 0.0
    r_converge: float = 0.0
    r_total: float = 0.0
    
    # Dimension scores
    consistency: float = 0.0
    completeness: float = 0.0
    accuracy: float = 0.0
    clarity: float = 0.0
    format: float = 0.0
    
    # Self-rating
    self_rating_quality: float = 0.0
    calibration_score: float = 0.0
    has_self_rating: bool = False
    
    # Reflection
    reflection_rounds: int = 1
    improvement_delta: float = 0.0
    
    # Meta
    answer_correct: bool = False
    thinking_length: int = 0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class CoRTrainingLogger:
    """Logger for CoR training process.
    
    Tracks and displays:
    1. Per-sample reward breakdowns
    2. Running statistics
    3. Self-rating calibration trends
    4. Reflection quality over time
    """
    
    def __init__(
        self,
        log_every_n: int = 10,
        verbose: bool = True,
        log_file: Optional[str] = None
    ):
        """Initialize logger.
        
        Args:
            log_every_n: Log detailed stats every N steps
            verbose: Print detailed per-sample logs
            log_file: Optional file path for JSON logs
        """
        self.log_every_n = log_every_n
        self.verbose = verbose
        self.log_file = log_file
        
        # Running statistics
        self.total_steps = 0
        self.entries: List[RewardLogEntry] = []
        
        # Aggregated metrics
        self.running_stats = {
            "r_external_sum": 0.0,
            "r_intrinsic_sum": 0.0,
            "r_total_sum": 0.0,
            "calibration_sum": 0.0,
            "correct_count": 0,
            "has_self_rating_count": 0,
            "total_count": 0,
        }
        
        # Per-dimension tracking
        self.dimension_sums = {
            "consistency": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "clarity": 0.0,
            "format": 0.0,
        }
        
        # Setup logging
        self.logger = logging.getLogger("CoRTraining")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [CoR] %(message)s',
                datefmt='%H:%M:%S'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_reward(
        self,
        step: int,
        sample_id: str,
        r_external: float,
        r_intrinsic: float,
        r_total: float,
        dim_scores: Dict[str, float],
        answer_correct: bool,
        thinking_chain: str = "",
        r_improve: float = 0.0,
        r_converge: float = 0.0,
        reflection_rounds: int = 1,
        self_ratings: Optional[Dict] = None,
        **kwargs
    ):
        """Log a single reward calculation.
        
        Args:
            step: Training step number
            sample_id: Sample identifier
            r_external: External reward
            r_intrinsic: Intrinsic reward
            r_total: Total reward
            dim_scores: Per-dimension scores
            answer_correct: Whether answer was correct
            thinking_chain: The thinking text (for length)
            r_improve: Improvement reward (for reflection)
            r_converge: Convergence reward
            reflection_rounds: Number of reflection rounds
            self_ratings: Extracted self-ratings
        """
        # Create entry
        entry = RewardLogEntry(
            step=step,
            sample_id=sample_id,
            r_external=r_external,
            r_intrinsic=r_intrinsic,
            r_improve=r_improve,
            r_converge=r_converge,
            r_total=r_total,
            consistency=dim_scores.get("consistency", 0.0),
            completeness=dim_scores.get("completeness", 0.0),
            accuracy=dim_scores.get("accuracy", 0.0),
            clarity=dim_scores.get("clarity", 0.0),
            format=dim_scores.get("format", 0.0),
            self_rating_quality=dim_scores.get("self_rating_quality", 0.0),
            calibration_score=kwargs.get("calibration_score", 0.0),
            has_self_rating=self_ratings is not None and len(self_ratings) > 0,
            reflection_rounds=reflection_rounds,
            improvement_delta=r_improve,
            answer_correct=answer_correct,
            thinking_length=len(thinking_chain),
        )
        
        self.entries.append(entry)
        self.total_steps += 1
        
        # Update running stats
        self._update_stats(entry)
        
        # Print if verbose or at interval
        if self.verbose or (step % self.log_every_n == 0):
            self._print_entry(entry, self_ratings)
        
        # Print summary at intervals
        if step > 0 and step % self.log_every_n == 0:
            self._print_summary()
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(entry)
    
    def _update_stats(self, entry: RewardLogEntry):
        """Update running statistics."""
        self.running_stats["r_external_sum"] += entry.r_external
        self.running_stats["r_intrinsic_sum"] += entry.r_intrinsic
        self.running_stats["r_total_sum"] += entry.r_total
        self.running_stats["calibration_sum"] += entry.self_rating_quality
        self.running_stats["total_count"] += 1
        
        if entry.answer_correct:
            self.running_stats["correct_count"] += 1
        if entry.has_self_rating:
            self.running_stats["has_self_rating_count"] += 1
        
        for dim in self.dimension_sums:
            self.dimension_sums[dim] += getattr(entry, dim, 0.0)
    
    def _print_entry(self, entry: RewardLogEntry, self_ratings: Optional[Dict] = None):
        """Print detailed entry log."""
        print("\n" + "="*70)
        print(f"📊 CoR Reward Log | Step {entry.step} | Sample: {entry.sample_id[:20]}...")
        print("="*70)
        
        # Reward breakdown
        print(f"\n🎯 REWARD BREAKDOWN:")
        print(f"   R_ext (external)     = {entry.r_external:.4f}  {'✅' if entry.answer_correct else '❌'}")
        print(f"   R_int (intrinsic)    = {entry.r_intrinsic:.4f}")
        if entry.r_improve != 0:
            print(f"   R_improve (reflect)  = {entry.r_improve:.4f}")
        if entry.r_converge != 0:
            print(f"   R_converge (stable)  = {entry.r_converge:.4f}")
        print(f"   ─────────────────────────────")
        print(f"   R_total              = {entry.r_total:.4f}")
        
        # Dimension scores
        print(f"\n📐 DIMENSION SCORES (5-dim):")
        dims = [
            ("Consistency", entry.consistency),
            ("Completeness", entry.completeness),
            ("Accuracy", entry.accuracy),
            ("Clarity", entry.clarity),
            ("Format", entry.format),
        ]
        for name, score in dims:
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"   {name:12s}: [{bar}] {score:.3f}")
        
        # Self-rating
        print(f"\n🔍 SELF-RATING CALIBRATION:")
        if entry.has_self_rating:
            print(f"   ✅ Self-ratings detected")
            print(f"   Calibration quality: {entry.self_rating_quality:.4f}")
            if self_ratings:
                print(f"   Extracted ratings:")
                for dim, rating in self_ratings.items():
                    if hasattr(rating, 'score'):
                        print(f"      {dim}: {rating.score}/10 (normalized: {rating.normalized:.2f})")
                    else:
                        print(f"      {dim}: {rating}")
        else:
            print(f"   ⚠️  No self-ratings found in thinking chain")
        
        # Reflection info
        if entry.reflection_rounds > 1:
            print(f"\n🔄 REFLECTION:")
            print(f"   Rounds: {entry.reflection_rounds}")
            print(f"   Improvement: {entry.improvement_delta:+.4f}")
        
        # Meta
        print(f"\n📝 META:")
        print(f"   Thinking length: {entry.thinking_length} chars")
        print(f"   Timestamp: {entry.timestamp}")
        
        sys.stdout.flush()
    
    def _print_summary(self):
        """Print running summary statistics."""
        n = self.running_stats["total_count"]
        if n == 0:
            return
        
        print("\n" + "─"*70)
        print(f"📈 RUNNING SUMMARY (last {n} samples)")
        print("─"*70)
        
        # Averages
        avg_ext = self.running_stats["r_external_sum"] / n
        avg_int = self.running_stats["r_intrinsic_sum"] / n
        avg_total = self.running_stats["r_total_sum"] / n
        avg_cal = self.running_stats["calibration_sum"] / n
        accuracy = self.running_stats["correct_count"] / n
        self_rating_rate = self.running_stats["has_self_rating_count"] / n
        
        print(f"\n   Average Rewards:")
        print(f"      R_ext:   {avg_ext:.4f}")
        print(f"      R_int:   {avg_int:.4f}")
        print(f"      R_total: {avg_total:.4f}")
        
        print(f"\n   Performance:")
        print(f"      Accuracy:         {accuracy*100:.1f}%")
        print(f"      Self-rating rate: {self_rating_rate*100:.1f}%")
        print(f"      Avg calibration:  {avg_cal:.4f}")
        
        print(f"\n   Dimension Averages:")
        for dim, total in self.dimension_sums.items():
            avg = total / n
            bar = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
            print(f"      {dim:12s}: [{bar}] {avg:.3f}")
        
        print("─"*70 + "\n")
        sys.stdout.flush()
    
    def _write_to_file(self, entry: RewardLogEntry):
        """Write entry to JSON file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
    
    def log_batch(
        self,
        step: int,
        batch_rewards: List[float],
        batch_correct: List[bool],
        mean_reward: float,
        **kwargs
    ):
        """Log batch-level statistics."""
        print(f"\n📦 Batch {step}: mean_R={mean_reward:.4f}, "
              f"correct={sum(batch_correct)}/{len(batch_correct)}, "
              f"rewards={[f'{r:.2f}' for r in batch_rewards[:5]]}...")
        sys.stdout.flush()
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        print(f"\n{'='*70}")
        print(f"🚀 EPOCH {epoch}/{total_epochs} STARTED")
        print(f"{'='*70}\n")
        sys.stdout.flush()
    
    def log_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Log epoch end with metrics."""
        print(f"\n{'='*70}")
        print(f"✅ EPOCH {epoch} COMPLETED")
        print(f"{'='*70}")
        print(f"\n   Metrics:")
        for name, value in metrics.items():
            print(f"      {name}: {value:.4f}")
        print(f"{'='*70}\n")
        sys.stdout.flush()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        n = self.running_stats["total_count"]
        if n == 0:
            return {}
        
        return {
            "total_samples": n,
            "avg_r_external": self.running_stats["r_external_sum"] / n,
            "avg_r_intrinsic": self.running_stats["r_intrinsic_sum"] / n,
            "avg_r_total": self.running_stats["r_total_sum"] / n,
            "accuracy": self.running_stats["correct_count"] / n,
            "self_rating_rate": self.running_stats["has_self_rating_count"] / n,
            "avg_calibration": self.running_stats["calibration_sum"] / n,
            "dimension_avgs": {
                dim: total / n for dim, total in self.dimension_sums.items()
            },
        }
    
    def reset_stats(self):
        """Reset running statistics (e.g., at epoch boundary)."""
        self.running_stats = {k: 0.0 if isinstance(v, float) else 0 
                             for k, v in self.running_stats.items()}
        self.dimension_sums = {k: 0.0 for k in self.dimension_sums}


# Global logger instance
_global_logger: Optional[CoRTrainingLogger] = None


def get_logger(
    log_every_n: int = 10,
    verbose: bool = True,
    log_file: Optional[str] = None
) -> CoRTrainingLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = CoRTrainingLogger(
            log_every_n=log_every_n,
            verbose=verbose,
            log_file=log_file
        )
    return _global_logger


def log_cor_reward(
    step: int,
    sample_id: str,
    reward_output,  # RewardOutput
    thinking_chain: str = "",
    self_ratings: Optional[Dict] = None,
    verbose: bool = True
):
    """Convenience function to log a RewardOutput."""
    logger = get_logger(verbose=verbose)
    
    logger.log_reward(
        step=step,
        sample_id=sample_id,
        r_external=reward_output.external_reward,
        r_intrinsic=reward_output.intrinsic_reward,
        r_total=reward_output.total_reward,
        dim_scores=reward_output.dimension_scores,
        answer_correct=reward_output.external_reward > 0.5,
        thinking_chain=thinking_chain,
        r_improve=reward_output.improvement_reward,
        r_converge=reward_output.convergence_reward,
        reflection_rounds=reward_output.reflection_rounds,
        self_ratings=self_ratings,
    )
