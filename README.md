# Chain of Reward (CoR) — 理论与代码对应文档

<p align="center">
  <strong>🔗 内生自评估的强化学习推理框架</strong>
</p>

<p align="center">
  <a href="#核心创新">核心创新</a> •
  <a href="#理论-代码对应">理论-代码对应</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#验证逻辑">验证逻辑</a>
</p>

---

## 核心创新

CoR (Chain of Reward) 是一个强化学习框架，具有三大创新：

1. **内生自评估 (Endogenous Self-Evaluation)**: 模型在推理过程中主动生成多维度自评分
2. **CoR-GRPO 双耦合**: 奖励信号与策略优化的双向协同进化
3. **自反省循环**: 通过迭代反省实现"越推理越智能"

---

## 理论-代码对应

### 📐 1. 核心奖励公式

| 理论公式 | 代码实现 |
|---------|---------|
| `R(c) = R_ext(c) + λ·R_int(c)` | `calculator.py:calculate_total_reward()` |

**理论 (theory.md §1):**
```
R(c) = R_ext(c) + λ·R_int(c)

- R_ext: 外部奖励（答案正确性，稀疏）
- R_int: 内在奖励（推理质量，稠密）
- λ: 平衡权重 = 1.0
```

**代码实现:**
```python
# s1-cor/train/rewards/calculator.py:245-260
def calculate_total_reward(self, thinking_chain, answer, ground_truth, ...):
    # 外部奖励
    external = self.calculate_external_reward(answer, ground_truth, grader_fn)
    
    # 内在奖励（含自评分质量）
    intrinsic, dim_scores = self.calculate_intrinsic_reward(
        thinking_chain,
        include_self_rating=True,
        final_answer_correct=(external > 0.5),
    )
    
    # 总奖励 = R_ext + λ * R_int
    total = external + self.config.lambda_intrinsic * intrinsic
    
    return RewardOutput(total_reward=total, ...)
```

---

### 📊 2. 五维度内在奖励

| 理论公式 | 代码实现 |
|---------|---------|
| `R_int = Σ w_d·r_d(y_think)` | `intrinsic.py:IntrinsicRewardCalculator` |

**理论 (theory.md §2):**
```
R_int(c) = Σ_{d=1}^5 w_d·r_d(y_think) + w_self·r_self_rating_quality

维度: Consistency, Completeness, Accuracy, Clarity, Format
权重: w_d = 0.2 (每维度)
```

**代码实现:**
```python
# s1-cor/train/rewards/intrinsic.py:353-385
class IntrinsicRewardCalculator:
    DEFAULT_WEIGHTS = {
        "consistency": 0.2,   # 逻辑一致性
        "completeness": 0.2,  # 步骤完整性
        "accuracy": 0.2,      # 事实准确性
        "clarity": 0.2,       # 推理清晰度
        "format": 0.2,        # 格式正确性
    }
    
    def __init__(self, weights=None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.reward_functions = {
            "consistency": ConsistencyReward(),
            "completeness": CompletenessReward(),
            "accuracy": AccuracyReward(),
            "clarity": ClarityReward(),
            "format": FormatReward(),
        }
```

**各维度评分函数:**

| 维度 | 类 | 评估逻辑 |
|-----|-----|---------|
| Consistency | `ConsistencyReward` | 检测逻辑词、步骤引用、无矛盾 |
| Completeness | `CompletenessReward` | 步骤数量、问题覆盖度 |
| Accuracy | `AccuracyReward` | 数学表达式、关键词使用 |
| Clarity | `ClarityReward` | 结构标记、可读性 |
| Format | `FormatReward` | 格式完整性、括号匹配 |

---

### 🎯 3. 自评分校准奖励

| 理论公式 | 代码实现 |
|---------|---------|
| `cal_d(u,v) = 1 - |u - v|` | `self_rating.py:compute_calibration()` |

**理论 (theory.md §3):**
```
cal_d(u, v) = 1 - |u - v|
r_self = (1/D) Σ cal_d(self_rating_d/10, actual_d)

- u: 模型自评分 (归一化到 0-1)
- v: 实际质量分数 (0-1)
- 高-高对齐奖励: +α (当 u>0.8 且 v>0.8)
```

**代码实现:**
```python
# s1-cor/train/rewards/self_rating.py:220-249
class SelfRatingEvaluator:
    def compute_calibration(self, self_rating, actual_quality, apply_bonus=True):
        """
        From THEORY.md Definition 8:
        cal_d(u, v) = 1 - |u - v|
        """
        # 基础校准
        cal = 1.0 - abs(self_rating - actual_quality)
        
        # 高-高对齐奖励
        if apply_bonus and self_rating > 0.8 and actual_quality > 0.8:
            cal += self.calibration_bonus  # α = 0.2
        
        return cal
```

**自评分提取:**
```python
# s1-cor/train/rewards/self_rating.py:84-146
class SelfRatingExtractor:
    """
    支持格式:
    - [Self-Rating: Consistency=8/10, Completeness=9/10]
    - [评分: 逻辑一致性=8/10, 步骤完整性=9/10]
    """
    def extract(self, thinking_chain) -> Dict[str, SelfRating]:
        # 尝试结构化格式: [Self-Rating: Dim1=X/10, ...]
        match = re.search(r'\[Self-Rating:\s*([^\]]+)\]', thinking_chain)
        if match:
            return self._parse_structured_rating(match.group(1))
        ...
```

---

### 🔄 4. 改进奖励（自反省）

| 理论公式 | 代码实现 |
|---------|---------|
| `R_improve = Q(c_{k+1}) - Q(c_k)` | `intrinsic.py:ImprovementRewardCalculator` |

**理论 (theory.md §9):**
```
R_improve(c_k, c_{k+1}) = Q(c_{k+1}) - Q(c_k)

累积改进:
R_total_improve = Σ_{k=0}^{K-1} γ^k · R_improve^{(k)}
```

**代码实现:**
```python
# s1-cor/train/rewards/intrinsic.py:460-530
class ImprovementRewardCalculator:
    def compute_improvement(self, chain_old, chain_new, **kwargs):
        """
        R_improve = Q(c_new) - Q(c_old)
        """
        q_old = self.compute_quality(chain_old, **kwargs)
        q_new = self.compute_quality(chain_new, **kwargs)
        return q_new - q_old
    
    def compute_cumulative_improvement(self, chain_sequence, gamma=0.9, **kwargs):
        """
        R_total = Σ_{k=0}^{K-1} γ^k * R_improve(c_k, c_{k+1})
        """
        total = 0.0
        for k in range(len(chain_sequence) - 1):
            improvement = self.compute_improvement(
                chain_sequence[k], 
                chain_sequence[k + 1]
            )
            total += (gamma ** k) * improvement
        return total
```

---

### ⚖️ 5. 收敛奖励

| 理论公式 | 代码实现 |
|---------|---------|
| `R_converge = 1 - |c_{k+1} - c_k|` | `intrinsic.py:ConvergenceRewardCalculator` |

**理论 (design.md §2.5):**
```
R_converge = -|c_{k+1} - c_k|  (归一化后)

鼓励模型收敛而非振荡
```

**代码实现:**
```python
# s1-cor/train/rewards/intrinsic.py:533-580
class ConvergenceRewardCalculator:
    def compute_convergence_reward(self, chain_old, chain_new, **kwargs):
        """收敛奖励 = 1 - divergence (归一化)"""
        divergence = self.compute_divergence(chain_old, chain_new, **kwargs)
        return max(0.0, 1.0 - divergence)
    
    def has_converged(self, chain_old, chain_new, threshold=0.1, **kwargs):
        """检查是否已收敛"""
        divergence = self.compute_divergence(chain_old, chain_new)
        return divergence < threshold
```

---

### 🔗 6. 扩展奖励公式（含反省）

| 理论公式 | 代码实现 |
|---------|---------|
| `R = R_ext + λ·R_int + μ·R_improve + ν·R_converge` | `calculator.py:calculate_reflection_reward()` |

**理论 (theory.md §11):**
```
R_CoR_Reflect = R_ext + λ·R_int + μ·R_improve + ν·R_converge

参数:
- λ = 1.0 (内在权重)
- μ = 0.5 (改进权重)
- ν = 0.1 (收敛权重)
```

**代码实现:**
```python
# s1-cor/train/rewards/calculator.py:275-350
def calculate_reflection_reward(self, chain_sequence, final_answer, ground_truth, ...):
    """
    Extended formula: R = R_ext + λ·R_int + μ·R_improve + ν·R_converge
    """
    # 1. 外部奖励
    external = self.calculate_external_reward(final_answer, ground_truth)
    
    # 2. 内在奖励
    intrinsic, dim_scores = self.calculate_intrinsic_reward(final_chain, ...)
    
    # 3. 改进奖励（累积）
    improvement = self.improvement_calculator.compute_cumulative_improvement(
        chain_sequence, gamma=self.config.improvement_discount
    )
    
    # 4. 收敛奖励
    convergence = self.convergence_calculator.compute_convergence_reward(
        chain_sequence[-2], chain_sequence[-1]
    )
    
    # 总奖励
    total = (
        external +
        self.config.lambda_intrinsic * intrinsic +      # λ = 1.0
        self.config.improvement_weight * improvement +   # μ = 0.5
        self.config.convergence_weight * convergence     # ν = 0.1
    )
```

---

### ⚙️ 7. 配置参数

| 理论参数 | 代码配置 | 值 |
|---------|---------|---|
| λ (intrinsic) | `RewardConfig.lambda_intrinsic` | 1.0 |
| μ (improve) | `RewardConfig.improvement_weight` | 0.5 |
| ν (converge) | `RewardConfig.convergence_weight` | 0.1 |
| K (rounds) | `RewardConfig.max_reflection_rounds` | 3 |
| α (bonus) | `RewardConfig.calibration_bonus` | 0.2 |
| N (candidates) | `CoRTrainingConfig.num_generations` | 8 |

**代码:**
```python
# s1-cor/train/rewards/calculator.py:23-55
@dataclass
class RewardConfig:
    lambda_intrinsic: float = 1.0       # λ
    improvement_weight: float = 0.5      # μ
    convergence_weight: float = 0.1      # ν
    max_reflection_rounds: int = 3       # K
    calibration_bonus: float = 0.2       # α
    
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "consistency": 0.2,
        "completeness": 0.2,
        "accuracy": 0.2,
        "clarity": 0.2,
        "format": 0.2,
    })
```

---

### 📝 8. GRPO 训练集成

| 理论算法 | 代码实现 |
|---------|---------|
| Algorithm 1: CoR-Reflect | `grpo.py:create_reward_fn()` |

**理论 (theory.md §13):**
```
for each batch:
    c^{(0)} ~ π_θ(·|x)                          # 初始生成
    for k = 0 to K-1:                           # 多轮反省
        self_rating_k = extract(c^{(k)})
        c^{(k+1)} ~ π_θ(·| x, reflection)
        R_improve^{(k)} = Q(c^{(k+1)}) - Q(c^{(k)})
    R_total = R_ext + λ·R_int + μ·Σ_k R_improve^{(k)}
    θ ← θ + α·∇_θ J(θ)                          # GRPO 更新
```

**代码实现:**
```python
# s1-cor/train/grpo.py:89-170
def create_reward_fn(config, enable_logging=True):
    calculator = RewardCalculator(reward_config)
    
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        for i, completion in enumerate(completions):
            # 提取反省轮次
            chain_sequence = extract_reflection_rounds(completion)
            
            if len(chain_sequence) > 1 and config.enable_reflection:
                # 多轮反省：使用扩展奖励
                output = calculator.calculate_reflection_reward(
                    chain_sequence, answer, gt
                )
                rewards.append(output.total_reward)
            else:
                # 单轮：标准 CoR 奖励
                output = calculator.calculate_total_reward(
                    thinking, answer, gt
                )
                rewards.append(output.total_reward)
        
        return rewards
    
    return reward_fn
```

---

## 📁 项目结构

```
s1-cor/
├── train/
│   ├── rewards/
│   │   ├── __init__.py           # 模块导出
│   │   ├── calculator.py         # RewardCalculator (核心)
│   │   ├── self_rating.py        # 自评分提取与校准
│   │   ├── intrinsic.py          # 5维度评分 + 反省奖励
│   │   └── training_logger.py    # 训练日志追踪
│   │
│   ├── grpo.py                   # GRPO 训练脚本
│   ├── sft_small.py              # SFT 训练脚本
│   └── validate_cor_logic.py     # CoR 逻辑验证
│
├── local_data/                   # 本地数据集
├── theory.md                     # 数学理论
├── design.md                     # 设计文档
└── README.md                     # 本文档
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers datasets trl accelerate
```

### 2. 验证 CoR 逻辑

```bash
cd s1-cor/train
python validate_cor_logic.py --dataset hf --samples 5
```

输出示例：
```
🔬 VALIDATING SAMPLE 0
----------------------------------------
1️⃣  SELF-RATING EXTRACTION
   ✅ Found 5 self-ratings:
      • consistency: 4.0/10 (normalized: 0.40)
      • completeness: 5.0/10 (normalized: 0.50)
      ...

2️⃣  INTRINSIC DIMENSION SCORES
   Consistency : [██████████] 1.000
   Completeness: [██░░░░░░░░] 0.250
   ...

3️⃣  SELF-RATING CALIBRATION
   📊 Average calibration: 0.680
   👍 Good calibration

4️⃣  TOTAL REWARD CALCULATION
   R_ext (external):  1.0000  ✅
   R_int (intrinsic): 0.6139
   R_total:           1.6139
```

### 3. 运行 SFT 训练

```bash
python train/sft_small.py --model_size 0.5B --dataset hf --push_to_hub
```

### 4. 运行 GRPO 训练

```bash
bash train/grpo.sh
```

---

## 验证逻辑

### 理论保证

| 定理 | 含义 | 验证方式 |
|-----|------|---------|
| **协同增益** | 双耦合进化快于独立组件 | 对比实验 |
| **收敛保证** | 压缩映射收敛到不动点 | 反省轮次追踪 |
| **单调改进** | 每轮反省质量提升 | R_improve > 0 |
| **Lyapunov 稳定** | 系统能量持续下降 | 训练曲线监控 |

### 日志追踪

训练时会输出详细日志：
```
📊 CoR Reward Log | Step 100 | Sample: sample_0...
======================================================================
🎯 REWARD BREAKDOWN:
   R_ext (external)     = 1.0000  ✅
   R_int (intrinsic)    = 0.6139
   R_improve (reflect)  = 0.1500
   R_converge (stable)  = 0.0800
   ─────────────────────────────
   R_total              = 1.8439

📐 DIMENSION SCORES (5-dim):
   Consistency : [██████████] 1.000
   Completeness: [██░░░░░░░░] 0.250
   Accuracy    : [█████░░░░░] 0.530
   Clarity     : [████░░░░░░] 0.400
   Format      : [████████░░] 0.800

🔍 SELF-RATING CALIBRATION:
   ✅ Self-ratings detected
   Calibration quality: 0.7034
```

---

## 数据格式

### 单轮推理（当前）
```
<thinking>
...推理步骤...
[Self-Rating: Consistency=7/10, Completeness=8/10, Accuracy=6/10, Clarity=7/10]
</thinking>
<answer>最终答案</answer>
```

### 多轮反省（扩展）
```
[Round 1]
<thinking>...初始推理...</thinking>
[Self-Rating: Consistency=4/10, Accuracy=3/10, ...]

[Reflection]
准确性较低 (3/10)。步骤 2 存在错误...

[Round 2]
<thinking>...修正后的推理...</thinking>
[Self-Rating: Consistency=8/10, Accuracy=9/10, ...]

[Convergence: Δ=+4.5, Stop=True]

<answer>最终答案</answer>
```

---

## 引用

如果您使用本项目，请引用：

```bibtex
@misc{cor2024,
  title={Chain of Reward: Endogenous Self-Evaluation for Reasoning},
  author={...},
  year={2024},
  howpublished={\url{https://github.com/chenxingqiang/s1-cor}}
}
```

---

## 许可证

MIT License

---

<p align="center">
  <strong>🎯 CoR: 让模型越推理越智能</strong>
</p>
