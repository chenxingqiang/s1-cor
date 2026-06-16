# Qwen2.5-0.5B：最小开源尺度 CoR 理论验证

用 **Qwen/Qwen2.5-0.5B-Instruct** 在最低 VRAM 下验证 CoR 框架的**理论阶梯**（非复现论文 32B AIME 数字）。

## 两层验证

| 层 | 环境 | 测什么 | 入口 |
|----|------|--------|------|
| **CPU 理论代理** | Cloud / 无 GPU | design.md §9 三阶段奖励阶梯 + 反思深度 K | `make loop-05b-theory` |
| **GPU 全链路** | CUDA ~1GB+ | SFT → GRPO → lm_eval | `grpo_05b.sh`、`run_scale_experiments.sh 0.5B` |

CPU 代理在** bundled `chain_sequence`** 上算 `RewardCalculator`，不加载 0.5B 权重，因此速度快、适合 Loop 闸门。

## CPU：理论阶梯（Loop R19）

```bash
cd s1-cor
source /workspace/.venv/bin/activate
make loop-05b-theory
# 或
python scripts/run_05b_theory_verify.py --json --strict
```

**期望（奖励代理，非 benchmark）**：

1. `cor_self_rating` 的 `mean_total` ≥ `sft_baseline`（λ=0 → λ=1 启用 R_int）
2. `cor_reflection` 的 `mean_total` ≥ `cor_self_rating`（启用 μ·R_improve）
3. K 增大时 `mean_total` 不下降（多轮反思项生效）

这些与 `run_reflection_k_ablation.py` 一致，但 `run_05b_theory_verify.py` 额外绑定 0.5B 训练超参与 `theory_checks` JSON。

## GPU：训练与评测

### 1. SFT（Colab / 单卡）

```bash
export WANDB_DISABLED=true
bash train/colab_minimal.sh sft
# 或
python train/sft_small.py --model_size 0.5B --dataset deepseek --colab
```

### 2. GRPO + CoR

```bash
export WANDB_DISABLED=true
export REF_MODEL=ckpts/sft-0.5B-colab   # 可选；缺省回退 base 权重
bash train/grpo_05b.sh deepseek
# 等价长入口
bash train/run_scale_experiments.sh 0.5B deepseek
```

对齐超参：`N=8`，`λ=1.0`，`μ=0.5`，`ν=0.1`，`K=3`，`block_size=4096`。

### 3. Benchmark（需 vLLM + OpenAI grader）

`eval/commands.sh` 中 `cor-0.5B` 行；将 `pretrained=` 指向 GRPO 输出目录。

```bash
export USE_MATH_GRADER=1   # 训练侧与 eval 对齐时建议在 GRPO 前设置
python scripts/check_eval_readiness.py --json
```

## 与论文叙述的关系

- **0.5B** 用于验证「奖励链 + 反思环 + GRPO 接线」在最小尺度可跑通，并观察代理指标是否沿 SFT → +CoR → +Reflection 上升。
- **论文主表数字**（如 AIME 56.7%）仍依赖 32B + 全量 eval；见 [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) P0。
- Token-level CoR、φ 双耦合仍为 **deferred**；见 [DEFERRED_CLAIMS.md](DEFERRED_CLAIMS.md)。

## Loop 索引

- Meta：`make loop-perceive` → `make loop-verify`
- 产品：`make loop-product-verify`（含 mini `05b_theory_verify`）
- 本专题：`make loop-05b-theory`

详见 [LOOPS.md](LOOPS.md) R19。
