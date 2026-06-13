# Five-Dimension Intrinsic Reward（R_int 契约）

`theory.md` §2 / `target.md` §3.1 定义：

$$R_{\text{int}}(c) = \sum_{d=1}^{D} w_d \, r_d(y_{\text{think}}) + w_{\text{self}} \, r_{\text{self}}$$

本仓库 **CPU 可审计** 的实现与论文主张差距如下。

## 实现状态（matrix: `five_dim_intrinsic` → **partial**）

| 主张 | 代码 | 状态 |
|------|------|------|
| 五维加权 $w_d$ | `IntrinsicRewardCalculator` + `RewardConfig.dimension_weights` | ✅ 可配置，默认各 0.2 |
| 启发式 $r_d$ | `rewards/intrinsic.py` 规则打分 | ⚠️ heuristic，非学习 $Q_\phi$ |
| GRPO 接线 | `grpo.py` → `dimension_weights_json` / `DIMENSION_WEIGHTS_JSON` | ✅ Loop R13 |
| token-level CoR | — | ❌ deferred（链级标量） |

## CPU 证据链

```bash
cd s1-cor
make loop-intrinsic-ablation   # w_d emphasize/drop 敏感度
make loop-intrinsic-scale      # R_ext vs R_int 尺度 + suggested λ
```

`run_intrinsic_dim_ablation.py` 输出：

- `emphasis_delta_vs_uniform` — 单维强调相对 uniform 的 ΔR_int
- `drop_delta_vs_uniform` — 单维置零相对 uniform 的 ΔR_int
- `most_sensitive_dimension` — |Δ| 最大维度

## GPU 训练

```bash
export USE_MATH_GRADER=1
# 可选：强调 format 维
export DIMENSION_WEIGHTS_JSON='{"format":0.4,"accuracy":0.3,"consistency":0.1,"completeness":0.1,"clarity":0.1}'
bash train/grpo.sh
```

或 CLI：`--dimension_weights_json='{"accuracy":1.0}'`

解析规则见 `train/intrinsic_weights.py`（JSON 或 `accuracy=0.5,format=0.5`）。

## 诚实叙述

- 论文「密集奖励链」在代码里是 **多轮反思链级** `R_int`，不是 per-token shaping。
- 五维分数来自正则/启发式，应用 ablation 说明 **超参敏感性**，不能当作已复现学习到的过程质量头。

相关：[GPU_TRAINING.md](GPU_TRAINING.md)、[LOOPS.md](LOOPS.md)、[theory_code_matrix.yaml](theory_code_matrix.yaml)。
