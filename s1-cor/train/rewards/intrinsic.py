"""
Intrinsic Reward Functions for Chain of Reward.

Multi-dimensional quality assessment of thinking chains.
Based on DESIGN.md Section 4.2 and THEORY.md Section 2.

Dimensions:
- Consistency: Logical coherence of reasoning
- Completeness: Step comprehensiveness  
- Accuracy: Factual correctness
- Clarity: Reasoning clarity
- Format: Structural correctness
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


class BaseIntrinsicReward(ABC):
    """Base class for intrinsic reward functions."""
    
    @abstractmethod
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute reward for thinking chain.
        
        Args:
            thinking_chain: The model's thinking process text.
            **kwargs: Additional context (e.g., question, answer).
            
        Returns:
            Reward score in [0, 1] range.
        """
        pass
    
    @property
    @abstractmethod
    def dimension_name(self) -> str:
        """Return the dimension name."""
        pass


class ConsistencyReward(BaseIntrinsicReward):
    """Evaluate logical consistency of reasoning.
    
    Checks:
    - No self-contradictions
    - Conclusions follow from premises
    - Intermediate results align with final answer
    """
    
    @property
    def dimension_name(self) -> str:
        return "consistency"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute consistency reward."""
        score = 1.0
        
        # Check for contradiction indicators
        contradiction_patterns = [
            r'(?:wait|actually|no)[,\s]+(?:that\'s|this is)\s+(?:wrong|incorrect)',
            r'I made (?:a|an) (?:mistake|error)',
            r'(?:Let me|I\'ll|I should)\s+(?:reconsider|recalculate|redo|start over)',
            r'(?:This|That) contradicts',
            r'(?:This|That) doesn\'t make sense',
        ]
        
        for pattern in contradiction_patterns:
            matches = re.findall(pattern, thinking_chain, re.IGNORECASE)
            # Penalize for contradictions, but not too harshly
            # (some self-correction is good, excessive is not)
            score -= 0.1 * min(len(matches), 3)
        
        # Check for logical flow markers (positive)
        flow_patterns = [
            r'(?:Therefore|Thus|Hence|So|Consequently)',
            r'(?:This means|This implies|It follows)',
            r'(?:Given that|Since|Because)',
            r'(?:From|Based on) (?:this|the above)',
        ]
        
        flow_count = 0
        for pattern in flow_patterns:
            flow_count += len(re.findall(pattern, thinking_chain, re.IGNORECASE))
        
        # Reward logical flow (up to a point)
        score += 0.05 * min(flow_count, 5)
        
        return max(0.0, min(1.0, score))


class CompletenessReward(BaseIntrinsicReward):
    """Evaluate step comprehensiveness.
    
    Checks:
    - Presence of clear reasoning steps
    - Problem decomposition
    - Verification/checking steps
    """
    
    @property
    def dimension_name(self) -> str:
        return "completeness"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute completeness reward."""
        score = 0.0
        
        # Check for numbered/bulleted steps
        step_patterns = [
            r'(?:Step\s+\d+|First|Second|Third|Finally)',
            r'^\s*\d+[\.\)]\s+',  # "1. " or "1) "
            r'^\s*[-•]\s+',  # Bullets
        ]
        
        step_count = 0
        for pattern in step_patterns:
            step_count += len(re.findall(pattern, thinking_chain, re.MULTILINE))
        
        # Reward having steps (3-10 is ideal)
        if step_count >= 3:
            score += 0.3
        if step_count >= 5:
            score += 0.2
        
        # Check for problem analysis
        analysis_patterns = [
            r'(?:Let\'s|I\'ll|We need to)\s+(?:analyze|understand|break down)',
            r'(?:The problem|This question)\s+(?:asks|requires|involves)',
            r'(?:Given|Known|We have)',
        ]
        
        for pattern in analysis_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.1
        
        # Check for verification
        verify_patterns = [
            r'(?:Let\'s|I\'ll|Let me)\s+(?:verify|check|confirm|validate)',
            r'(?:To verify|Checking|Double-check)',
            r'(?:This|The answer) (?:seems|looks|appears) (?:correct|right)',
        ]
        
        for pattern in verify_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.15
        
        # Check for conclusion
        conclusion_patterns = [
            r'(?:Therefore|Thus|Hence|So|In conclusion)',
            r'(?:The answer is|Final answer)',
            r'(?:We conclude|I conclude)',
        ]
        
        for pattern in conclusion_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.15
        
        return max(0.0, min(1.0, score))


class ClarityReward(BaseIntrinsicReward):
    """Evaluate reasoning clarity.
    
    Checks:
    - Clear structure and organization
    - Explicit variable/term definitions
    - Readable formatting
    """
    
    @property
    def dimension_name(self) -> str:
        return "clarity"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute clarity reward."""
        score = 0.5  # Start neutral
        
        # Check for definitions/explanations
        definition_patterns = [
            r'(?:Let|Define|Denote)\s+\w+\s*(?:=|as|to be)',
            r'(?:where|here)\s+\w+\s+(?:is|represents|denotes)',
        ]
        
        for pattern in definition_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.1
        
        # Check for clear section markers
        section_patterns = [
            r'(?:Step\s+\d+:|Part\s+\d+:|First,|Second,|Finally,)',
            r'(?:Analysis:|Solution:|Approach:|Method:)',
        ]
        
        section_count = 0
        for pattern in section_patterns:
            section_count += len(re.findall(pattern, thinking_chain, re.IGNORECASE))
        
        score += 0.05 * min(section_count, 5)
        
        # Penalize for very long unbroken text (wall of text)
        paragraphs = thinking_chain.split('\n\n')
        for para in paragraphs:
            if len(para) > 1000:  # Very long paragraph
                score -= 0.1
        
        # Check average sentence length (too long = unclear)
        sentences = re.split(r'[.!?]+', thinking_chain)
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len > 50:  # Very long sentences
                score -= 0.1
            elif avg_len < 30:  # Reasonable length
                score += 0.1
        
        return max(0.0, min(1.0, score))


class AccuracyReward(BaseIntrinsicReward):
    """Evaluate factual and mathematical accuracy.
    
    Checks:
    - Mathematical expressions correctness
    - Numerical consistency
    - Reference to known facts/formulas
    """
    
    @property
    def dimension_name(self) -> str:
        return "accuracy"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute accuracy reward."""
        score = 0.5  # Start neutral
        
        # Check for mathematical content
        math_patterns = [
            r'\$[^$]+\$',  # LaTeX inline
            r'\\\[[^\]]+\\\]',  # LaTeX display
            r'=\s*[\d\.\-]+',  # Equations
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Arithmetic
        ]
        
        math_count = 0
        for pattern in math_patterns:
            math_count += len(re.findall(pattern, thinking_chain))
        
        if math_count > 0:
            score += 0.1 * min(math_count / 5, 0.3)
        
        # Check for verification/checking
        verify_patterns = [
            r'(?:verify|check|confirm|validate)',
            r'(?:substitut|plug).+(?:back|in)',
            r'(?:correct|right|checks out)',
        ]
        
        for pattern in verify_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.1
        
        # Check for error/correction patterns (indicates awareness)
        correction_patterns = [
            r'(?:wait|actually),?\s+(?:that|this)\s+(?:is|should be)',
            r'I\s+(?:notice|see)\s+(?:an?\s+)?(?:error|mistake)',
        ]
        
        for pattern in correction_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.05  # Self-correction is good
        
        # Check for formula/theorem references
        reference_patterns = [
            r'(?:theorem|lemma|formula|equation)\s+(?:\d+|[A-Z])',
            r'(?:by|using|from)\s+(?:the\s+)?(?:Pythagorean|quadratic|binomial)',
            r'(?:definition|property)\s+of',
        ]
        
        for pattern in reference_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.1
        
        return max(0.0, min(1.0, score))


class FormatReward(BaseIntrinsicReward):
    """Evaluate structural correctness of output format.
    
    Checks:
    - Proper thinking/answer separation
    - Self-rating format presence
    - No malformed markers
    """
    
    @property
    def dimension_name(self) -> str:
        return "format"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute format reward."""
        score = 0.5  # Start neutral
        
        # Check for self-rating format (important for CoR)
        self_rating_patterns = [
            r'\[Self-Rating:',
            r'\[评分:',
            r'\[Rating:.*?/10\]',
        ]
        
        has_self_rating = False
        for pattern in self_rating_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                has_self_rating = True
                score += 0.3
                break
        
        # Check for proper structure markers
        if '<|im_start|>' in kwargs.get('full_text', ''):
            # Qwen format
            if '<|im_end|>' in kwargs.get('full_text', ''):
                score += 0.1
        
        # Check for markdown formatting (headers, lists)
        markdown_patterns = [
            r'^#+\s+',  # Headers
            r'^\s*[-*]\s+',  # Lists
            r'\*\*[^*]+\*\*',  # Bold
            r'`[^`]+`',  # Code
        ]
        
        markdown_count = 0
        for pattern in markdown_patterns:
            markdown_count += len(re.findall(pattern, thinking_chain, re.MULTILINE))
        
        if markdown_count > 0:
            score += 0.1
        
        # Penalize for incomplete/malformed content
        if thinking_chain.strip().endswith('...'):
            score -= 0.2
        
        if thinking_chain.count('(') != thinking_chain.count(')'):
            score -= 0.1
        
        if thinking_chain.count('[') != thinking_chain.count(']'):
            score -= 0.1
        
        return max(0.0, min(1.0, score))


class IntrinsicRewardCalculator:
    """Calculate multi-dimensional intrinsic rewards.
    
    Combines multiple reward dimensions with configurable weights.
    Based on THEORY.md Definition 5:
    R_int(c) = Σ_d w_d * r_d(y_think)
    """
    
    # Per paper: w_d = 0.2 for each of 5 dimensions
    DEFAULT_WEIGHTS = {
        "consistency": 0.2,
        "completeness": 0.2,
        "accuracy": 0.2,
        "clarity": 0.2,
        "format": 0.2,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize calculator.
        
        Args:
            weights: Weights for each dimension. Defaults to equal weights.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Initialize reward functions
        self.reward_functions: Dict[str, BaseIntrinsicReward] = {
            "consistency": ConsistencyReward(),
            "completeness": CompletenessReward(),
            "accuracy": AccuracyReward(),
            "clarity": ClarityReward(),
            "format": FormatReward(),
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def compute_all_dimensions(
        self, 
        thinking_chain: str,
        **kwargs
    ) -> Dict[str, float]:
        """Compute rewards for all dimensions.
        
        Args:
            thinking_chain: The model's thinking process.
            **kwargs: Additional context.
            
        Returns:
            Dict mapping dimension names to reward scores.
        """
        results = {}
        
        for dim, func in self.reward_functions.items():
            results[dim] = func.compute(thinking_chain, **kwargs)
        
        return results
    
    def compute_weighted_reward(
        self,
        thinking_chain: str,
        **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """Compute weighted intrinsic reward.
        
        Args:
            thinking_chain: The model's thinking process.
            **kwargs: Additional context.
            
        Returns:
            Tuple of (weighted_total, per_dimension_scores).
        """
        per_dim = self.compute_all_dimensions(thinking_chain, **kwargs)
        
        weighted_total = sum(
            self.weights.get(dim, 0.0) * score
            for dim, score in per_dim.items()
        )
        
        return weighted_total, per_dim
    
    def get_actual_qualities(self, thinking_chain: str, **kwargs) -> Dict[str, float]:
        """Get actual quality scores for self-rating calibration.
        
        These are used to evaluate how well the model's self-ratings
        match the actual quality metrics.
        """
        return self.compute_all_dimensions(thinking_chain, **kwargs)


class ReflectionReward(BaseIntrinsicReward):
    """Evaluate quality of self-reflection and improvement.
    
    Part of CoR-GRPO dual coupling: rewards iterative improvement.
    Based on theory.md Section 10-12.
    
    Checks:
    - Presence of reflection markers
    - Identification of weak dimensions
    - Actual improvement in quality
    """
    
    @property
    def dimension_name(self) -> str:
        return "reflection"
    
    def __init__(self):
        self.intrinsic_calc = IntrinsicRewardCalculator()
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute reflection quality reward."""
        score = 0.0
        
        # Check for reflection markers
        reflection_patterns = [
            r'\[Reflection\]',
            r'\[Round \d+\]',
            r'(?:Let me|I should|I need to)\s+(?:reconsider|rethink|revise)',
            r'(?:Upon|After)\s+(?:reflection|review)',
            r'(?:I notice|I see)\s+(?:that|an?\s+(?:error|issue|problem))',
        ]
        
        has_reflection = False
        for pattern in reflection_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                has_reflection = True
                score += 0.2
                break
        
        # Check for dimension identification
        dimension_patterns = [
            r'(?:consistency|completeness|accuracy|clarity)\s+is\s+(?:low|weak|poor)',
            r'(?:my|the)\s+(?:consistency|completeness|accuracy|clarity)\s+(?:score|rating)',
            r'\d+/10.*(?:low|needs improvement)',
        ]
        
        for pattern in dimension_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.15
                break
        
        # Check for revision/improvement markers
        improvement_patterns = [
            r'(?:revised|corrected|improved|fixed)',
            r'(?:better|more accurate|clearer)',
            r'\[Round 2\]',
            r'(?:Updated|New)\s+(?:approach|solution|reasoning)',
        ]
        
        for pattern in improvement_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.2
                break
        
        # Check for convergence/stopping
        convergence_patterns = [
            r'\[Convergence',
            r'(?:satisfied|confident)\s+(?:with|in)\s+(?:this|the)\s+(?:answer|solution)',
            r'Continue[=:]\s*(?:No|False)',
        ]
        
        for pattern in convergence_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.15
                break
        
        # Bonus for multiple rounds with improvement
        rounds = re.findall(r'\[Round (\d+)\]', thinking_chain)
        if len(rounds) >= 2:
            score += 0.1
        
        return max(0.0, min(1.0, score))


class ImprovementRewardCalculator:
    """Calculate improvement reward between reasoning chains.
    
    R_improve(c_k, c_{k+1}) = Q(c_{k+1}) - Q(c_k)
    
    Part of the self-reflection framework.
    Based on theory.md Section 10.
    """
    
    def __init__(self):
        self.intrinsic_calc = IntrinsicRewardCalculator()
    
    def compute_quality(self, thinking_chain: str, **kwargs) -> float:
        """Compute overall quality Q(c) of a reasoning chain."""
        weighted, _ = self.intrinsic_calc.compute_weighted_reward(thinking_chain, **kwargs)
        return weighted
    
    def compute_improvement(
        self, 
        chain_old: str, 
        chain_new: str,
        **kwargs
    ) -> float:
        """Compute improvement reward between two chains.
        
        R_improve = Q(c_new) - Q(c_old)
        
        Args:
            chain_old: Previous reasoning chain
            chain_new: Revised reasoning chain
            
        Returns:
            Improvement score (can be negative if quality decreased)
        """
        q_old = self.compute_quality(chain_old, **kwargs)
        q_new = self.compute_quality(chain_new, **kwargs)
        
        return q_new - q_old
    
    def compute_cumulative_improvement(
        self,
        chain_sequence: List[str],
        gamma: float = 0.9,
        **kwargs
    ) -> float:
        """Compute cumulative discounted improvement over multiple rounds.
        
        R_total = Σ_{k=0}^{K-1} γ^k * R_improve(c_k, c_{k+1})
        
        Args:
            chain_sequence: List of reasoning chains [c_0, c_1, ..., c_K]
            gamma: Discount factor
            
        Returns:
            Total cumulative improvement
        """
        if len(chain_sequence) < 2:
            return 0.0
        
        total = 0.0
        for k in range(len(chain_sequence) - 1):
            improvement = self.compute_improvement(
                chain_sequence[k], 
                chain_sequence[k + 1],
                **kwargs
            )
            total += (gamma ** k) * improvement
        
        return total


class ConvergenceRewardCalculator:
    """Calculate convergence reward to encourage stability.
    
    R_converge = -|c_{k+1} - c_k|
    
    Encourages the model to converge rather than oscillate.
    """
    
    def __init__(self):
        self.intrinsic_calc = IntrinsicRewardCalculator()
    
    def compute_divergence(self, chain_old: str, chain_new: str, **kwargs) -> float:
        """Compute divergence between two chains based on quality scores."""
        scores_old = self.intrinsic_calc.compute_all_dimensions(chain_old, **kwargs)
        scores_new = self.intrinsic_calc.compute_all_dimensions(chain_new, **kwargs)
        
        # L1 distance between score vectors
        divergence = sum(
            abs(scores_new.get(d, 0) - scores_old.get(d, 0))
            for d in set(scores_old.keys()) | set(scores_new.keys())
        )
        
        return divergence
    
    def compute_convergence_reward(
        self, 
        chain_old: str, 
        chain_new: str,
        **kwargs
    ) -> float:
        """Compute convergence reward (negative divergence, normalized).
        
        Returns value in [0, 1] where 1 = perfect convergence.
        """
        divergence = self.compute_divergence(chain_old, chain_new, **kwargs)
        
        # Normalize: divergence of 0 -> reward 1, divergence >= 1 -> reward 0
        return max(0.0, 1.0 - divergence)
    
    def has_converged(
        self, 
        chain_old: str, 
        chain_new: str, 
        threshold: float = 0.1,
        **kwargs
    ) -> bool:
        """Check if reflection has converged."""
        divergence = self.compute_divergence(chain_old, chain_new, **kwargs)
        return divergence < threshold
