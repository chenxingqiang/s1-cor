"""Shared constants for benchmark reproduction tooling."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "eval" / "lm-evaluation-harness"
COMMANDS = REPO_ROOT / "eval" / "commands.sh"
SMOKE_RESULTS_DIR = REPO_ROOT / "results" / "eval_smoke_dummy"
FIXTURE_LM_EVAL = REPO_ROOT / "train" / "fixtures" / "lm_eval_sample_results.json"

PAPER_TARGETS = {
    "AIME24": 56.7,
    "MATH500": 93.0,
    "GPQA": 59.6,
}

DEFAULT_CHECKPOINTS = [
    "ckpts/cor-grpo",
    "ckpts/cor-sft",
    "ckpts/cor-32B",
]

SCALE_05B_CHECKPOINTS = [
    "ckpts/sft-0.5B-colab",
    "ckpts/cor-0.5B",
]

SCALE_05B_PIPELINE = [
    {
        "id": "sft_05b",
        "title": "SFT Qwen2.5-0.5B on s1K-cor",
        "command": "export WANDB_DISABLED=true && bash train/colab_minimal.sh sft",
        "artifact_glob": "ckpts/sft-0.5B-colab",
        "doc": "docs/MODEL_05B_TEST.md",
    },
    {
        "id": "grpo_05b",
        "title": "CoR GRPO on 0.5B",
        "command": "export WANDB_DISABLED=true USE_MATH_GRADER=1 REF_MODEL=ckpts/sft-0.5B-colab bash train/grpo_05b.sh deepseek",
        "artifact_glob": "ckpts/cor-0.5B*",
        "doc": "train/grpo_05b.sh",
    },
    {
        "id": "readiness",
        "title": "Benchmark eval readiness",
        "command": "python scripts/check_eval_readiness.py --json",
        "doc": "docs/EVAL_REPRODUCTION.md",
    },
    {
        "id": "eval_05b",
        "title": "lm_eval cor-0.5B (commands.sh line)",
        "command": (
            "cd eval/lm-evaluation-harness && "
            "OPENAI_API_KEY=$OPENAI_API_KEY bash ../commands.sh  # set pretrained= to GRPO output"
        ),
        "doc": "s1-cor/eval/commands.sh",
    },
    {
        "id": "compare",
        "title": "Compare lm_eval JSON (sanity, not 32B paper targets)",
        "command": "python scripts/compare_eval_to_paper.py --results-dir cor-0.5B-eval --json",
        "doc": "docs/EVAL_REPRODUCTION.md",
    },
]

REPRODUCTION_STEPS = [
    {
        "id": "train",
        "title": "Train CoR pipeline (SFT → GRPO)",
        "command": "USE_MATH_GRADER=1 bash train/run_cor_pipeline.sh",
        "doc": "docs/GPU_TRAINING.md",
    },
    {
        "id": "readiness",
        "title": "Eval readiness gate",
        "command": "python scripts/check_eval_readiness.py",
        "doc": "docs/EVAL_REPRODUCTION.md",
    },
    {
        "id": "eval",
        "title": "lm_eval benchmarks",
        "command": "cd eval/lm-evaluation-harness && bash ../commands.sh",
        "doc": "s1-cor/eval/commands.sh",
    },
    {
        "id": "compare",
        "title": "Compare to paper targets",
        "command": "python scripts/compare_eval_to_paper.py --results-dir <output>",
        "doc": "docs/EVAL_REPRODUCTION.md",
    },
]
