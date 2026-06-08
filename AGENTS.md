# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

This repository is an **ML research codebase** for **Chain of Reward (CoR)** / **s1** reasoning-model training. It is not a web app. Workflows are Python scripts and shell launchers.

### Environment

- **Python**: 3.12+ with a project virtualenv at `/workspace/.venv`
- **Activate**: `source /workspace/.venv/bin/activate`
- **Package manager**: `pip` (`s1-cor/requirements.txt`)
- **GPU**: Training, vLLM inference, and full benchmark eval require CUDA. Cloud Agent VMs are CPU-only by default; use the CPU smoke tests below.

Core deps are installed with PyTorch CPU wheels. **vLLM**, **unsloth**, and **bitsandbytes** from `requirements.txt` are GPU-oriented and are not installed in the default Cloud update script.

### Local datasets

Load CoR snapshots with `load_cor_dataset_from_disk()` from `s1-cor/train/data_utils.py` (used by validation, SFT, GRPO, and MLX prep scripts). `datasets.load_from_disk` works for the bundled `local_data/s1K_cor_*` shards after the schema/metadata fixes in this repo.

### Quick verification (no GPU)

From repo root with venv activated:

```bash
# Unit tests (23 tests)
cd s1-cor && python -m pytest train/rewards/test_rewards.py -v

# CoR logic on bundled local sample (when HF dataset / load_from_disk unavailable)
cd s1-cor && python3 -c "
import json, sys; sys.path.insert(0,'train')
from validate_cor_logic import validate_sample
from rewards import RewardCalculator, RewardConfig
from rewards.self_rating import SelfRatingExtractor
from rewards.intrinsic import IntrinsicRewardCalculator
from rewards.training_logger import CoRTrainingLogger
sample = json.load(open('local_data/s1K_cor_deepseek/sample.json'))
cfg = RewardConfig(lambda_intrinsic=1.0, self_rating_weight=0.2, calibration_bonus=0.2)
validate_sample(sample, RewardCalculator(cfg), SelfRatingExtractor(),
    IntrinsicRewardCalculator(), CoRTrainingLogger(1, False), 0)
"
```

Official README smoke test (needs HuggingFace access or working `local_data`):

```bash
cd s1-cor && python train/validate_cor_logic.py --dataset hf --samples 5
```

### Training & evaluation (GPU hosts)

See `README.md` and `s1-cor/README.md`:

| Step | Command |
|------|---------|
| Full pipeline | `bash s1-cor/train/run_cor_pipeline.sh` |
| SFT | `python s1-cor/train/sft_small.py --model_size 0.5B --dataset hf` |
| GRPO | `bash s1-cor/train/grpo.sh` |
| lm-eval setup | `cd s1-cor/eval/lm-evaluation-harness && pip install -e .[math,vllm]` |
| Benchmarks | `bash s1-cor/eval/commands.sh` |

Set `WANDB_DISABLED=true` to skip Weights & Biases during training.

### Optional services

| Service | When needed |
|---------|-------------|
| Hugging Face Hub | Model/dataset download |
| OpenAI API | MATH/GPQA grading in `eval/commands.sh` |
| Weights & Biases | Training logs (disable with `WANDB_DISABLED=true`) |
| vLLM + CUDA | Local inference and benchmark eval |

### Lint / test

- **Tests**: `pytest s1-cor/train/rewards/test_rewards.py`
- **No project-level linter** configured at repo root; vendored `lm-evaluation-harness` has its own pre-commit config.
