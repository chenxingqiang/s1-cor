# Publication Readiness（顶会投稿工程清单）

本文面向 **ICML/NeurIPS 级** 审稿：闭合 **主张 — 代码 — 数字** 三角，避免 README/论文过度声称。与 `docs/theory_code_matrix.yaml`、`docs/DEFERRED_CLAIMS.md` 同步。

## 1. 审稿人 30 秒结论（当前仓库状态）

| 维度 | 状态 | 证据入口 |
|------|------|----------|
| **四分量奖励公式** | ✅ implemented | `pytest train/rewards/test_rewards.py` |
| **多轮反思 + GRPO** | ✅ implemented | `grpo.py`, `reflection_parsing.py` |
| **Benchmark 数字** | ⚠️ partial | 表内为 **design.md §9 目标**；全量复现需 GPU ckpt |
| **Token-level CoR** | ❌ deferred | [DEFERRED_CLAIMS.md](DEFERRED_CLAIMS.md) |
| **φ 双耦合头** | ❌ deferred | θ-only GRPO + ECE 代理 |
| **五维 R_int** | ⚠️ partial | 启发式 $r_d$，非学习 $Q_\phi$ |

**P0 阻塞发表级复现**：在 GPU 主机完成 `USE_MATH_GRADER=1` 训练 → `eval/commands.sh` → `compare_eval_to_paper` **三项 pass**（非 fixture）。见 [EVAL_REPRODUCTION.md](EVAL_REPRODUCTION.md)。

## 2. Abstract / Introduction 建议表述

**可写（有代码支撑）**：

- 内生自评 + 校准奖励项（`self_rating_calibration` implemented）
- 扩展 GRPO 奖励：$R_{\text{ext}}+\lambda R_{\text{int}}+\mu R_{\text{improve}}+\nu R_{\text{converge}}$
- 1K 规模数据与多轮反思数据格式

**须软化或放 Methods/Limitations**：

- 「逐步密集 CoR / 每 token 折扣」→ **链级** 启发式 $R_{\text{int}}$（deferred）
- 「φ–θ 双耦合梯度」→ **θ-only GRPO** + 校准代理指标（deferred φ 头）
- 「800× 样本效率」→ 需对照实验与 **公开 checkpoint**；GPQA 59.6 vs o1-preview 73.3 非全面超越

## 3. Results 表与复现

README / design.md 表格数字来源：

| 列 | 含义 |
|----|------|
| AIME24 / MATH500 / GPQA | `eval_repro_common.PAPER_TARGETS`（CoR-32B + K=2 目标行） |
| w/o CoR 基线 | design.md §9 消融阶梯（需 GPU 训练验证） |

**不得**将 `train/fixtures/lm_eval_sample_results.json` 当作真实跑分；其为 CPU `compare_eval_to_paper` 契约 fixture。

复现命令链：

```bash
cd s1-cor
make loop-benchmark-repro          # CPU：链路 + fixture 审计
# GPU 主机：
export USE_MATH_GRADER=1 WANDB_DISABLED=true
bash train/run_cor_pipeline.sh
python scripts/check_eval_readiness.py
cd eval/lm-evaluation-harness && bash ../commands.sh
python scripts/compare_eval_to_paper.py --results-dir <output> --json
```

依赖：`OPENAI_API_KEY`（MATH500/GPQA）、CUDA、vLLM、ckpt。见 [TRAIN_EVAL_GRADING.md](TRAIN_EVAL_GRADING.md)。

## 4. Ablation 叙述对齐

| 论文主张 | 仓库 CPU 代理 | GPU 需补 |
|----------|---------------|----------|
| λ/μ/α 超参 | `make loop-ablation` | 训练曲线 |
| 反思深度 K | `run_reflection_k_ablation.py` | K=1,2,3 评测 |
| w/o $R_{\text{int}}$ / w/o 自评 | design.md 表 | 独立 ckpt + lm_eval JSON |
| 五维 $w_d$ | `make loop-intrinsic-ablation` | 同上 |
| 0.5B 最小尺度 | `make loop-05b-theory` | `grpo_05b.sh` + cor-0.5B eval |
| 五维契约门 | `make loop-five-dim-contract` | GPU 上 w_d 训练曲线 |

## 5.1 最小尺度附录（0.5B）

CPU 代理（非 benchmark 分数）：

```bash
cd s1-cor
make loop-05b-theory   # theory_ok + 三阶段 mean_total 阶梯
```

GPU 全链路见 [MODEL_05B_TEST.md](MODEL_05B_TEST.md)。**不得**将 CPU `mean_total` 代理当作 AIME/MATH 发表数字。

## 5.2 五维 R_int 契约（partial）

```bash
make loop-five-dim-contract   # contract_ok + Pearson r + 维度敏感度
```

低 `pooled_pearson_r` 须在 Limitations 写明（启发式 $r_d$ ≠ 学习 $Q_\phi$）。见 [FIVE_DIM_INTRINSIC.md](FIVE_DIM_INTRINSIC.md)。

## 5. 投稿前检查清单

```
[ ] GPU：compare_eval_to_paper 全 pass（真实 lm_eval 输出路径）
[ ] 公开或审稿可访问 checkpoint（HF / 附录链接）
[ ] README Results 脚注 + PUBLICATION_READINESS 链接
[ ] 论文 main.tex Limitations 与 DEFERRED_CLAIMS 一致
[ ] make loop-publication-ready exit 0
[ ] make loop-verify + loop-product-verify exit 0
[ ] theory_code_matrix 无虚假 implemented
```

## 6. CPU 审计（本仓库默认可跑）

```bash
cd s1-cor
make loop-publication-ready
```

检查：关键文档存在、README 复现免责声明、matrix partial/deferred 与文档交叉引用。

## 7. 相关文档

- [theory_code_matrix.yaml](theory_code_matrix.yaml) — 契约 tier
- [DEFERRED_CLAIMS.md](DEFERRED_CLAIMS.md) — 诚实降级
- [FIVE_DIM_INTRINSIC.md](FIVE_DIM_INTRINSIC.md) — $R_{\text{int}}$ partial
- [EVAL_REPRODUCTION.md](EVAL_REPRODUCTION.md) — GPU 复现
- [LOOPS.md](LOOPS.md) — 工程闭环 R0–R20
