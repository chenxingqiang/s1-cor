---
title: "CoR: Chain of Reward with Endogenous Self-Evaluation for Reasoning"
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: static
pinned: true
license: mit
tags:
  - reasoning
  - reinforcement-learning
  - chain-of-thought
  - self-evaluation
  - grpo
  - llm
---

# CoR: Chain of Reward with Endogenous Self-Evaluation for Reasoning

<p align="center">
  <strong>🎯 Teaching models to think better, not just answer correctly</strong>
</p>

<p align="center">
  <a href="https://github.com/chenxingqiang/s1-cor">💻 Code</a> •
  <a href="https://huggingface.co/papers/cor-2026">📄 Paper</a> •
  <a href="https://huggingface.co/datasets/xingqiang/s1K-cor-deepseek">📊 Dataset</a>
</p>

---

## Abstract

How do expert problem-solvers learn? They don't just check final answers—they continuously monitor their reasoning: "Is this step logically sound? Am I missing something?" This metacognitive self-awareness is largely absent in how we train reasoning models today.

Current reinforcement learning approaches provide only sparse, outcome-based rewards—correct or incorrect—telling the model *what* to achieve, but not *how* to think better. We propose **Chain of Reward (CoR)**, which introduces *endogenous self-evaluation*: the model generates multi-dimensional self-ratings during reasoning (e.g., "Consistency=8/10, Completeness=9/10"), and we reward it for *accurate* self-assessment. This creates a powerful metacognitive learning signal: the model learns not only to think well, but to *know when it is thinking well*—enabling genuine self-correction.

## Key Results

| Model | Samples | AIME24 | MATH500 | GPQA |
|-------|---------|--------|---------|------|
| o1-preview | N.A. | 44.6 | 85.5 | 73.3 |
| r1-distill | 800K | 72.6 | 94.3 | 62.1 |
| **CoR-32B (Ours)** | **1K** | **56.7** | **93.0** | **59.6** |

**800× sample efficiency**: Competitive with o1-preview using only 1,000 training examples.

## Core Innovation

### Endogenous Self-Evaluation

The model generates self-ratings during reasoning:
```
<thinking>
Let me solve this step by step...
[Self-Rating: Consistency=8/10, Completeness=9/10, Accuracy=7/10, Clarity=8/10]
</thinking>
```

We reward the model for **accurate** self-assessment, not just correct answers.

### Four-Component Reward

```
R(c) = R_ext + λ·R_int + μ·R_improve + ν·R_converge
```

| Component | Description | Weight |
|-----------|-------------|--------|
| R_ext | External (answer correctness) | 1.0 |
| R_int | Intrinsic (5-dim quality + calibration) | λ=1.0 |
| R_improve | Improvement across reflection | μ=0.5 |
| R_converge | Solution stability | ν=0.1 |

### Dual-Coupled Dynamics

Better policies → Higher-quality chains → Stronger reward signals → Better policies ⟳

## Framework

![CoR Framework](figures/cor_architecture.pdf)

**Left**: Unlike traditional RL with sparse final rewards, CoR distributes dense intrinsic rewards along the reasoning chain.

**Right**: Four-component reward decomposition with dual-coupled evolutionary dynamics.

## Training Pipeline

![Training Pipeline](figures/cor_pipeline.pdf)

1. **Data Curation**: Build CoR-1K from s1-1K with self-rating augmentation
2. **SFT Training**: Teach the model reasoning format
3. **CoR-GRPO**: Optimize with four-component rewards
4. **Final Model**: Achieves competitive performance with 800× sample efficiency

## Contributions

1. **Endogenous Self-Evaluation**: A novel mechanism where models generate and are rewarded for accurate self-ratings during reasoning
2. **Chain of Reward (CoR)**: A four-component reward decomposition with theoretical convergence guarantees
3. **Extreme Sample Efficiency**: Competitive performance with o1-preview using only 1,000 training examples

## Citation

```bibtex
@article{chen2026cor,
  title={CoR: Chain of Reward with Endogenous Self-Evaluation for Reasoning},
  author={Chen, Xingqiang},
  journal={ICML},
  year={2026}
}
```

## License

MIT License
