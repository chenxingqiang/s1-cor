# Deferred Theory Claims（诚实降级契约）

部分 `theory.md` / `target.md` 主张**尚未实现**，在 `docs/theory_code_matrix.yaml` 中标为 **deferred**。本文供论文叙述、PR 与 Loop 审计使用，避免将链级启发式误称为完整 CoR 理论。

## 索引

| Matrix ID | 理论主张 | 当前实现 | 可测代理 |
|-----------|----------|----------|----------|
| `token_level_reward_chain` | CoR(τ)=Σ γ^t r_int 逐步折扣 | 链级标量 `R_int`（`rewards/intrinsic.py`） | `make loop-intrinsic-ablation` |
| `dual_coupling_phi` | φ 头与 θ 双耦合梯度 | 仅 GRPO 更新 θ；φ 无独立头 | `make loop-calibration` / `loop-calibration-ablation` |

## token_level_reward_chain

**论文/理论**：奖励沿 token 或逐步思考状态密集分布（`target.md` §1–2）。

**代码现实**：

- `IntrinsicRewardCalculator` 对整条 `thinking_chain` 输出**一个**加权标量。
- 多轮反思通过 `reflection_parsing.py` → `R_improve` / `R_converge` 在**链级**建模，非 per-token shaping。
- matrix `code: null` — 无 token-level 实现文件。

**实验叙述应写**：「链级密集启发式 + 反思项」，而非「已实现 token-level CoR 折扣和」。

**若未来实现**：需新模块 + pytest + matrix tier→implemented + ablation 对比链级基线。

详见 [FIVE_DIM_INTRINSIC.md](FIVE_DIM_INTRINSIC.md)。

## dual_coupling_phi

**论文/理论**：校准质量头 φ 与策略 θ 联合演化（`theory.md` §5–6）。

**代码现实**：

- `grpo.py` 仅更新策略 θ；无 `φ` 参数块。
- `calibration_proxy_phi`（**implemented**）提供 CPU ECE 代理与 `calibration_bonus` α 扫参，**不等于**学习 φ。
- `dual_coupling_phi` 保留 **deferred**；代理入口：`run_calibration_report.py`、`run_calibration_bonus_ablation.py`。

**实验叙述应写**：「θ-only GRPO + 自评校准奖励项」，校准演化通过 reward shaping 间接体现。

## CPU 审计

```bash
cd s1-cor
make loop-deferred-claims   # matrix deferred 项 ↔ 本文档一致性
```

## 与元循环的关系

- **感知**：`loop_perceive` 的 `matrix_gaps` 含 deferred 计数。
- **策略**：不得将 deferred 项标为 implemented 以通过 Loop。
- **验证**：`make loop-deferred-claims` exit 0 表示文档与 matrix 同步。

见 [LOOPS.md](LOOPS.md)、[theory_code_matrix.yaml](theory_code_matrix.yaml)、[AGENTS.md](../AGENTS.md)、[PUBLICATION_READINESS.md](PUBLICATION_READINESS.md)。
