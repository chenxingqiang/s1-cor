# GPU 训练与论文数字闭环

CPU Cloud Agent 无法跑完整 pipeline；本文档供 **GPU 主机** 按 Loop R7+ 串联训练与评测。

## 快速路径

```bash
cd s1-cor
export WANDB_DISABLED=true          # 可选：跳过 W&B
export USE_MATH_GRADER=1            # 推荐：eval 对齐的 R_ext
bash train/run_cor_pipeline.sh
```

分步：

| 步骤 | 命令 |
|------|------|
| SFT | `python train/sft_small.py` 或 pipeline Step 2 |
| GRPO | `USE_MATH_GRADER=1 bash train/grpo.sh` |
| 闸门 | `python scripts/check_eval_readiness.py` |
| 评测 | `cd eval/lm-evaluation-harness && bash ../commands.sh` |
| 对比 | `python scripts/compare_eval_to_paper.py --results-dir <out>` |

## R_ext：string vs math grader

| 模式 | 启用方式 | 说明 |
|------|----------|------|
| 默认 | （无） | 字符串规范化匹配 |
| Eval 对齐 | `USE_MATH_GRADER=1` 或 `--use_math_grader=True` | `answer_grading.py`：boxed + sympy |

CPU 预检（无需 GPU）：

```bash
cd s1-cor
make loop-grpo-smoke      # reward_fn 接线
make loop-r-ext-align     # attempt vs solution 分歧
```

## R_int：五维权重 $w_d$

默认各维 $w_d=0.2$（`IntrinsicRewardCalculator.DEFAULT_WEIGHTS`）。GPU 可覆盖：

```bash
export DIMENSION_WEIGHTS_JSON='{"accuracy":0.4,"format":0.2,"consistency":0.1,"completeness":0.1,"clarity":0.2}'
bash train/grpo.sh
```

CPU 预检：

```bash
make loop-intrinsic-ablation   # emphasize/drop 敏感度
make loop-intrinsic-scale      # suggested λ_intrinsic
```

详见 [FIVE_DIM_INTRINSIC.md](FIVE_DIM_INTRINSIC.md)。

## 环境要求

- CUDA + 多卡（32B 用 FSDP，见 `train/fsdp_config_qwen.json`）
- `sympy`（math grader；Cloud update script 已含）
- 评测：`pip install -e eval/lm-evaluation-harness[math,vllm]` + `OPENAI_API_KEY`

## Colab / 小 GPU

不产出论文表数字，仅烟雾：

```bash
bash train/colab_minimal.sh sft
```

## 相关文档

- [EVAL_REPRODUCTION.md](EVAL_REPRODUCTION.md) — 评测复现
- [LOOPS.md](LOOPS.md) — 双层 Loop
- [AGENTS.md](../AGENTS.md) — 无限优化闭环
