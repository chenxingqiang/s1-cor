# CoR 论文与代码对齐分析

## 1. 理论公式与代码实现对比

### 1.1 奖励分解 (Section 3.1 Method)

**论文公式 (Eq. 2)**:
```
R(c) = R_ext(c) + λ * R_int(c)
```

**代码实现** (`train/rewards/calculator.py:232`):
```python
total = external + self.config.lambda_intrinsic * intrinsic
```
✅ **完全对齐**

---

### 1.2 外部奖励 (Eq. 3)

**论文公式**:
```
R_ext(c) = I[y_answer = y_gt]
```

**代码实现** (`train/rewards/calculator.py:112`):
```python
return 1.0 if answer_clean == gt_clean else 0.0
```
✅ **完全对齐**

---

### 1.3 内在奖励 (Eq. 4)

**论文公式**:
```
R_int(c) = Σ_{d=1}^{D} w_d * r_d(y_think) + w_self * r_self_rating_quality
```

**代码实现** (`train/rewards/calculator.py:182-190`):
```python
intrinsic_reward = (
    weighted_intrinsic +  # Σ w_d * r_d
    self.config.self_rating_weight * self_rating_reward  # w_self * r_self
) / total_weight
```
✅ **完全对齐**

---

### 1.4 自评分质量奖励 (Eq. 6-7)

**论文公式**:
```
r_self_rating_quality = (1/D) * Σ cal_d(self_rating_d/10, actual_quality_d)
cal_d(u, v) = 1 - |u - v|
```

**代码实现** (`train/rewards/self_rating.py`):
```python
def _calibration_score(self, predicted: float, actual: float) -> float:
    return 1.0 - abs(predicted - actual)

def compute_self_rating_reward(...):
    calibrations = []
    for dim in actual_qualities:
        if dim in self_ratings:
            cal = self._calibration_score(self_ratings[dim], actual_qualities[dim])
            calibrations.append(cal)
    return np.mean(calibrations) if calibrations else 0.5
```
✅ **完全对齐**

---

### 1.5 GRPO 优势函数 (Eq. 8-10)

**论文公式**:
```
A^(i) = (R(c^(i)) - μ_R) / (σ_R + ε)
```

**代码实现**: 使用 TRL `GRPOTrainer`，其内部实现了标准化优势计算。
✅ **通过 TRL 库实现**

---

### 1.6 GRPO 目标函数 (Eq. 11)

**论文公式**:
```
J(θ) = E_x[1/N Σ min(r_i*A^(i), clip(r_i,1-δ,1+δ)*A^(i))] - β*D_KL(π_θ||π_ref)
```

**代码实现** (`train/grpo.py:244-251`):
```python
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,  # For KL penalty
    args=grpo_args,       # Contains β (kl_penalty) and δ (clip_ratio)
    reward_funcs=reward_fn,
)
```
✅ **通过 TRL GRPOTrainer 实现**

---

## 2. 实验设置对比

### 2.1 训练流程

| 论文描述 | 代码状态 |
|---------|---------|
| SFT on Qwen2.5-32B-Instruct with CoR-1K | ✅ `train/sft.py` |
| GRPO training with CoR rewards | ✅ `train/grpo.py` |
| λ = 1.0 (intrinsic weight) | ✅ 默认配置 |
| N = 8 (candidates per group) | ⚠️ 代码默认 N=4，需调整 |
| β = 0.01 (KL penalty) | ⚠️ 需在 grpo.sh 中配置 |
| δ = 0.2 (clipping) | ⚠️ 需在 grpo.sh 中配置 |
| w_d = 0.2 for each dimension | ⚠️ 代码默认 w_d=0.25，需调整 |
| w_self = 0.2 | ✅ 默认配置 |

### 2.2 数据集

| 论文描述 | 代码状态 |
|---------|---------|
| CoR-1K: 1000 curated samples | ✅ `local_data/s1K_cor_full` (规则) |
| Self-ratings embedded in thinking | ✅ 已生成 |
| Format: [Self-Rating: Consistency=X/10, ...] | ✅ 已实现 |
| Distilled from Gemini Thinking | 原始 s1K 数据 |

### 2.3 评估

| 论文描述 | 代码状态 |
|---------|---------|
| AIME24 (30 problems) | ✅ `eval/lm-evaluation-harness` |
| MATH500 (500 samples) | ✅ 已集成 |
| GPQA Diamond (198 questions) | ✅ 已集成 |
| Temperature = 0 (greedy) | ✅ 默认设置 |

---

## 3. 理论结果对比 (Section 4: Theory)

### 3.1 已实现的理论保证

| 定理 | 论文内容 | 代码支持 |
|------|---------|---------|
| Theorem 1 (Policy Improvement) | 内在奖励引导策略改进 | ✅ GRPO框架 |
| Theorem 2 (Calibration Improvement) | 自评分提高校准度 | ✅ 校准奖励 |
| Theorem 3 (Convergence) | 有界奖励下收敛 | ✅ 奖励在[0,2]范围 |

### 3.2 需要验证的假设

| 假设 | 内容 | 验证状态 |
|------|------|---------|
| Assumption 1 | 有界奖励 | ✅ 代码强制 |
| Assumption 2 | 策略空间紧致 | 隐式满足 |
| Assumption 3 | 稀疏外部奖励 | ✅ 二值奖励 |

---

## 4. 实验结果对比

### 4.1 论文主要结果 (Table 1)

| 模型 | 训练样本 | AIME24 | MATH500 | GPQA |
|-----|---------|--------|---------|------|
| CoR-32B w/o CoR | 1K | 50.0 | 92.6 | 56.6 |
| CoR-32B | 1K | 56.7 | 93.0 | 59.6 |
| **提升** | - | **+6.7** | **+0.4** | **+3.0** |

### 4.2 实验验证待办

1. [ ] 运行 SFT baseline (w/o CoR)
2. [ ] 运行 GRPO + CoR 训练
3. [ ] 在 AIME24/MATH500/GPQA 上评估
4. [ ] 对比 baseline 和 CoR 结果
5. [ ] 验证校准度改进

---

## 5. 代码修正建议

### 5.1 配置对齐 (高优先级)

```python
# train/rewards/calculator.py - 修改默认权重
dimension_weights: Dict[str, float] = field(default_factory=lambda: {
    "consistency": 0.2,    # 改为 0.2
    "completeness": 0.2,   # 改为 0.2
    "accuracy": 0.2,       # 新增
    "clarity": 0.2,        # 改为 0.2
})

# train/grpo.py - 修改默认候选数
num_generations: int = field(default=8)  # 改为 8
```

### 5.2 训练脚本对齐

```bash
# train/grpo.sh - 添加论文参数
--num_generations 8 \
--kl_penalty 0.01 \
--clip_ratio 0.2 \
--lambda_intrinsic 1.0 \
```

### 5.3 数据集字段对齐

数据集需要包含：
- `question`: 问题
- `thinking_rated`: 带自评分的思维链
- `answer`: 最终答案
- `ground_truth`: 正确答案 (用于外部奖励)

---

## 6. 实验执行计划

### Phase 1: Baseline SFT (1-2小时)
```bash
cd s1/train
./sft.sh --dataset local_data/s1K_cor_full
```

### Phase 2: GRPO + CoR (4-8小时)
```bash
cd s1/train
./grpo.sh --ref_model ckpts/sft-baseline
```

### Phase 3: 评估 (1-2小时)
```bash
cd s1/eval
python generate.py --model ckpts/cor-grpo --benchmarks aime24,math500,gpqa
```

### Phase 4: 消融实验
1. CoR vs vanilla GRPO (无内在奖励)
2. 不同 λ 值对比
3. 自评分质量奖励贡献

---

## 7. 当前状态总结

### ✅ 已完成
- 奖励计算模块 (RewardCalculator)
- 自评分提取和评估
- GRPO 训练框架
- 数据集生成 (规则 + DeepSeek)
- 数据加载工具

### ⚠️ 需要调整
- 默认参数对齐论文
- grpo.sh 脚本完善
- 添加 accuracy 维度

### 🔄 进行中
- DeepSeek 增强数据集生成

### 📝 待验证
- 实际训练运行
- 基准测试结果
- 论文表格数据
