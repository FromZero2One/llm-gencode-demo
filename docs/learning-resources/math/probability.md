# 📐 概率论与信息论

> **阅读时间**: 15-20分钟  
> **重要性**: ⭐⭐⭐（理解采样策略和损失函数）

---

## 📋 目录

1. [概率基础](#1-概率基础)
2. [常见概率分布](#2-常见概率分布)
3. [信息论基础](#3-信息论基础)
4. [在Transformer中的应用](#4-在transformer中的应用)

---

## 1. 概率基础

### 1.1 基本概念

**概率公理**：
1. 非负性：P(A) ≥ 0
2. 归一化：P(Ω) = 1
3. 可加性：P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

**条件概率**：
```
P(A|B) = P(A ∩ B) / P(B)
```

**贝叶斯定理**：
```
P(A|B) = P(B|A) · P(A) / P(B)
```

---

## 2. 常见概率分布

### 2.1 离散分布

**伯努利分布**：
```
P(X=1) = p
P(X=0) = 1-p
```

**分类分布**（Categorical）：
```
P(X=k) = p_k, 其中 Σ p_k = 1
```

**在Transformer中的应用**：
- 模型输出的概率分布
- Token采样策略

---

### 2.2 连续分布

**高斯分布**（正态分布）：
```
N(x; μ, σ²) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
```

**均匀分布**：
```
U(x; a, b) = 1/(b-a), 当 a ≤ x ≤ b
```

---

## 3. 信息论基础

### 3.1 熵（Entropy）⭐⭐⭐⭐

**定义**：
```
H(X) = -Σ p(x) · log(p(x))
```

**含义**：
- 衡量随机变量的不确定性
- 熵越大，不确定性越高
- 均匀分布的熵最大

**代码示例**：
```python
import torch

# 计算概率分布的熵
probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
entropy = -(probs * torch.log(probs)).sum()
print(f"Entropy: {entropy:.4f}")
```

---

### 3.2 交叉熵（Cross-Entropy）⭐⭐⭐⭐⭐

**定义**：
```
H(p, q) = -Σ p(x) · log(q(x))
```

**在机器学习中的应用**：
- 分类任务的损失函数
- 衡量预测分布q与真实分布p的差异

**代码示例**：
```python
import torch.nn as nn

# CrossEntropy Loss
criterion = nn.CrossEntropyLoss()

# logits: 模型输出（未softmax）
# targets: 真实标签
loss = criterion(logits, targets)
```

---

### 3.3 KL散度（Kullback-Leibler Divergence）

**定义**：
```
KL(p || q) = Σ p(x) · log(p(x)/q(x))
```

**性质**：
- KL散度 ≥ 0
- KL(p || q) ≠ KL(q || p)（不对称）
- KL(p || p) = 0

**与交叉熵的关系**：
```
H(p, q) = H(p) + KL(p || q)
```

---

## 4. 在Transformer中的应用

### 4.1 采样策略的概率解释

**Greedy采样**：
```python
# 选择概率最大的token
next_token = torch.argmax(probs)
```

**Temperature采样**：
```python
# 调整概率分布的平滑程度
scaled_probs = torch.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(scaled_probs, 1)
```

**Top-K采样**：
```python
# 只从概率最大的K个token中采样
top_k_probs, top_k_indices = torch.topk(probs, k)
next_token = torch.multinomial(top_k_probs, 1)
```

**Top-P采样**（核采样）：
```python
# 从累积概率达到p的最小集合中采样
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
# 找到累积概率首次超过p的位置
```

---

### 4.2 Label Smoothing

**目的**：防止模型过于自信，提高泛化能力

**公式**：
```
p_smoothed = (1 - ε) · p_one_hot + ε / K
```

其中K是类别数，ε是平滑系数（通常0.1）

**代码示例**：
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 📝 练习题

### 练习1: 计算熵

计算以下概率分布的熵：
```
p = [0.5, 0.25, 0.25]
```

<details>
<summary>点击查看答案</summary>

```
H = -(0.5·log(0.5) + 0.25·log(0.25) + 0.25·log(0.25))
  = -(0.5·(-0.693) + 0.25·(-1.386) + 0.25·(-1.386))
  = 0.347 + 0.347 + 0.347
  = 1.04 bits
```
</details>

---

### 练习2: 贝叶斯定理

假设：
- P(疾病) = 0.01
- P(阳性|疾病) = 0.99
- P(阳性|无疾病) = 0.05

如果检测呈阳性，患病的概率是多少？

<details>
<summary>点击查看答案</summary>

```
P(疾病|阳性) = P(阳性|疾病) · P(疾病) / P(阳性)

P(阳性) = P(阳性|疾病)·P(疾病) + P(阳性|无疾病)·P(无疾病)
        = 0.99·0.01 + 0.05·0.99
        = 0.0099 + 0.0495
        = 0.0594

P(疾病|阳性) = 0.99·0.01 / 0.0594 ≈ 0.167
```

即使检测呈阳性，患病概率也只有16.7%！这就是贝叶斯定理的反直觉之处。
</details>

---

## 🔗 深入学习

如需更详细的内容，请参考：
- [TRANSFORMER_MATH_FOUNDATION.md](../TRANSFORMER_MATH_FOUNDATION.md) - 第3部分
- 《Pattern Recognition and Machine Learning》- Christopher Bishop

---

**祝您学习愉快！**

*最后更新: 2026-05-25 | 版本: v2.0-simplified*
