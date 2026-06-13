# Train vs Eval Grading（R_ext 契约）

训练 `R_ext` 与 benchmark 评测使用**不同但文档化**的判题路径。本文闭合 Loop R11 的 partial→implemented 叙述。

## 训练路径（GRPO / RewardCalculator）

| 模式 | 启用 | 实现 |
|------|------|------|
| 默认 | `use_math_grader=False` | 字符串规范化（`calculator._normalize_answer`） |
| 推荐 | `USE_MATH_GRADER=1` | `answer_grading.py`：boxed 提取 + sympy 等价（对齐 lm-eval metamathqa） |

CPU 审计：

```bash
cd s1-cor
make loop-r-ext-align    # string vs math；recommended_training_grader
```

当 `recommended_training_grader=math` 或 `math_fixes_string > 0` 时，GPU 训练应设 `USE_MATH_GRADER=1`（见 [GPU_TRAINING.md](GPU_TRAINING.md)）。

## 评测路径（lm-eval）

| 任务 | 判题 | 依赖 |
|------|------|------|
| AIME / 数值题 | metamathqa / aime utils | sympy（CPU 可解析） |
| MATH500 (`openai_math`) | OpenAI 提取 + 等价 | `OPENAI_API_KEY` |
| GPQA | OpenAI 提取 | `OPENAI_API_KEY` |

入口：`s1-cor/eval/commands.sh`（需 GPU + vLLM + ckpt）。

CPU 审计（不调 OpenAI API）：

```bash
cd s1-cor
make loop-eval-openai-grader   # 任务列表 + regex 提取 smoke + blockers
make loop-eval-grading-path    # 训练 vs eval 预-OpenAI 提取路径一致性
```

契约矩阵：`eval_openai_grader`（partial，eval-only）。

## 三角闭合状态

| 主张 | 训练 | 评测 |
|------|------|------|
| 答案正确性 R_ext | ✅ string + math | ✅ lm-eval（math 子集 CPU 可审计） |
| GPQA/MATH OpenAI | — | ⚠️ 需 API key，非训练默认 |

全量论文数字： [EVAL_REPRODUCTION.md](EVAL_REPRODUCTION.md)。
