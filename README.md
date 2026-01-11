# Chain of Reward (CoR)

<p align="center">
  <strong>Endogenous Self-Evaluation for Sample-Efficient Reasoning</strong>
</p>

<p align="center">
  <a href="paper/main.pdf">📄 Paper</a> •
  <a href="#key-idea">Key Idea</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#results">Results</a> •
  <a href="#theory-code-mapping">Theory-Code</a>
</p>

---

## Key Idea

**CoR (Chain of Reward)** introduces *endogenous self-evaluation* for training reasoning models. Instead of only rewarding correct answers, we reward the model for:

1. **Accurate self-assessment**: The model generates self-ratings during reasoning and is rewarded for calibrated self-evaluation
2. **Reasoning quality**: Dense intrinsic rewards across 5 dimensions (Consistency, Completeness, Accuracy, Clarity, Format)
3. **Iterative improvement**: Rewards for quality gains through self-reflection

**Result**: 800× sample efficiency—competitive with o1-preview using only 1,000 training examples.

---

## Quick Start

### Installation

```bash
pip install torch transformers datasets trl accelerate
```

### Validate CoR Logic

```bash
cd s1-cor/train
python validate_cor_logic.py --dataset hf --samples 5
```

### Train with SFT

```bash
python train/sft_small.py --model_size 0.5B --dataset hf
```

### Train with CoR-GRPO

```bash
bash train/grpo.sh
```

---

## Results

| Model | Samples | AIME24 | MATH500 | GPQA |
|-------|---------|--------|---------|------|
| o1-preview | N.A. | 44.6 | 85.5 | 73.3 |
| r1-distill | 800K | 72.6 | 94.3 | 62.1 |
| Sky-T1 | 17K | 43.3 | 82.4 | 56.8 |
| Bespoke-32B | 17K | 63.3 | 93.0 | 58.1 |
| **CoR-32B** | **1K** | **56.7** | **93.0** | **59.6** |

---

## Core Formula

```
R(c) = R_ext + λ·R_int + μ·R_improve + ν·R_converge
```

| Component | Description | Weight |
|-----------|-------------|--------|
| R_ext | External reward (answer correctness) | 1.0 |
| R_int | Intrinsic reward (5-dim quality + self-rating calibration) | λ=1.0 |
| R_improve | Improvement reward (Q(c_{k+1}) - Q(c_k)) | μ=0.5 |
| R_converge | Convergence reward (solution stability) | ν=0.1 |

---

## Theory-Code Mapping

| Theory | Code |
|--------|------|
| Total Reward Formula | `rewards/calculator.py:calculate_reflection_reward()` |
| 5-Dimension Intrinsic | `rewards/intrinsic.py:IntrinsicRewardCalculator` |
| Self-Rating Calibration | `rewards/self_rating.py:compute_calibration()` |
| Improvement Reward | `rewards/intrinsic.py:ImprovementRewardCalculator` |
| Convergence Reward | `rewards/intrinsic.py:ConvergenceRewardCalculator` |
| GRPO Integration | `train/grpo.py:create_reward_fn()` |

---

## Project Structure

```
s1-cor/
├── train/
│   ├── rewards/
│   │   ├── calculator.py      # RewardCalculator (core)
│   │   ├── self_rating.py     # Self-rating extraction & calibration
│   │   ├── intrinsic.py       # 5-dim scoring + reflection rewards
│   │   └── training_logger.py # Training log tracking
│   ├── grpo.py                # GRPO training script
│   ├── sft_small.py           # SFT training script
│   └── validate_cor_logic.py  # CoR logic validation
├── eval/
│   ├── generate.py            # Model evaluation
│   └── commands.sh            # Evaluation commands
└── local_data/                # Local datasets

paper/
├── main.tex                   # Paper source
├── main.pdf                   # Compiled paper
└── figures/
    ├── cor_architecture.pdf   # Framework diagram
    ├── cor_pipeline.pdf       # Training pipeline
    ├── cor_efficiency.pdf     # Sample efficiency
    └── cor_ablation.pdf       # Ablation study
```

---

## Data Format

### Single-round Reasoning
```
<thinking>
...reasoning steps...
[Self-Rating: Consistency=7/10, Completeness=8/10, Accuracy=6/10, Clarity=7/10]
</thinking>
<answer>Final answer</answer>
```

### Multi-round Reflection
```
[Round 1]
<thinking>...initial reasoning...</thinking>
[Self-Rating: Consistency=4/10, Accuracy=3/10, ...]

[Reflection]
Accuracy is low (3/10). Error in step 2...

[Round 2]
<thinking>...corrected reasoning...</thinking>
[Self-Rating: Consistency=8/10, Accuracy=9/10, ...]

<answer>Final answer</answer>
```

---

## Citation

```bibtex
@article{chen2026cor,
  title={CoR: Chain of Reward with Endogenous Self-Evaluation for Reasoning},
  author={Chen, Xingqiang},
  journal={ICML},
  year={2026}
}
```

---

## License

MIT License

---

<p align="center">
  <strong>🎯 CoR: Teaching models to think better, not just answer correctly</strong>
</p>
