# 📐 线性代数基础

> **阅读时间**: 30-45分钟  
> **重要性**: ⭐⭐⭐⭐⭐（Transformer的核心数学基础）

---

## 📋 目录

1. [向量运算](#1-向量运算)
2. [矩阵运算](#2-矩阵运算)
3. [张量基础](#3-张量基础)
4. [在Transformer中的应用](#4-在transformer中的应用)

---

## 1. 向量运算

### 1.1 点积（内积）⭐⭐⭐⭐⭐

**定义**：
```
u · v = Σᵢ uᵢ·vᵢ = u₁v₁ + u₂v₂ + ... + uₙvₙ
```

**几何意义**：
```
u · v = ||u|| · ||v|| · cos(θ)
```

其中 θ 是两个向量的夹角。

**在Transformer中的应用**：
```python
# 注意力分数的计算
score = query · key = ||query|| · ||key|| · cos(θ)
```

**代码示例**：
```python
import torch
import torch.nn.functional as F

# 方法1：使用PyTorch内置函数
query = torch.randn(1, 768)
key = torch.randn(1, 768)
cos_sim = F.cosine_similarity(query, key, dim=-1)

# 方法2：手动计算（等价）
query_norm = F.normalize(query, p=2, dim=-1)
key_norm = F.normalize(key, p=2, dim=-1)
cos_sim = (query_norm * key_norm).sum(dim=-1)
```

---

### 1.2 向量范数 ⭐⭐⭐⭐⭐

**L2范数（欧几里得范数）**：
```
||v||₂ = √(Σ vᵢ²) = √(v₁² + v₂² + ... + vₙ²)
```

**几何意义**：向量的长度

**在Transformer中的应用**：
- Layer Normalization
- 梯度裁剪
- 权重衰减（L2正则化）

**代码示例**：
```python
# L2范数计算
u = torch.tensor([3.0, 4.0])
l2_norm = torch.norm(u)  # 5.0

# 归一化
u_normalized = u / l2_norm
print(torch.norm(u_normalized))  # 1.0 ✓

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 1.3 L1 vs L2 范数对比

| 特性 | L1范数 | L2范数 |
|------|--------|--------|
| **数学定义** | `Σ \|vᵢ\|` | `√(Σ vᵢ²)` |
| **稀疏性** | ✅ 产生稀疏解 | ❌ 产生密集解 |
| **可导性** | ❌ 在0点不可导 | ✅ 处处可导 |
| **应用场景** | L1正则化（Lasso） | LayerNorm、注意力机制 |

**为什么Transformer主要用L2范数？**
1. 可导性好（梯度下降需要）
2. 几何直观（符合欧几里得空间）
3. 计算稳定（与Softmax配合良好）
4. 理论支持（大量优化理论基于L2）

---

## 2. 矩阵运算

### 2.1 矩阵乘法 ⭐⭐⭐⭐⭐

**定义**：
```
C = A @ B
C[i,j] = Σₖ A[i,k] · B[k,j]
```

**在Transformer中的应用**：
```python
# 注意力分数计算
scores = Q @ K.T  # (batch, seq_len, seq_len)

# 线性变换
output = input @ W  # (batch, seq_len, d_model)
```

**代码示例**：
```python
# 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B  # shape: (3, 5)

# 批量矩阵乘法（常见于Transformer）
batch_A = torch.randn(32, 10, 64)  # (batch, seq, d_k)
batch_B = torch.randn(32, 64, 10)  # (batch, d_k, seq)
batch_C = torch.bmm(batch_A, batch_B)  # (32, 10, 10)
```

---

### 2.2 矩阵转置

**定义**：
```
A^T[i,j] = A[j,i]
```

**在Transformer中的应用**：
```python
# 计算注意力分数时需要转置K
scores = Q @ K.T / math.sqrt(d_k)
```

---

### 2.3 特殊矩阵

**单位矩阵**：
```python
I = torch.eye(5)  # 5x5单位矩阵
```

**对角矩阵**：
```python
D = torch.diag(torch.tensor([1, 2, 3, 4, 5]))
```

**下三角矩阵（Causal Mask）**：
```python
# Decoder中的Causal Mask
mask = torch.tril(torch.ones(5, 5))
# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1]]
```

---

## 3. 张量基础

### 3.1 什么是张量？

张量是向量和矩阵的高维推广。

| 阶数 | 名称 | 形状示例 | Transformer中的对应 |
|------|------|----------|---------------------|
| 0 | 标量 | `( )` | 单个损失值 |
| 1 | 向量 | `(d,)` | 单个token的embedding |
| 2 | 矩阵 | `(m, n)` | batch的embeddings |
| 3 | 3阶张量 | `(a, b, c)` | 整个input batch |
| N | N阶张量 | `(dim₁, ..., dimₙ)` | 注意力权重 |

---

### 3.2 Transformer中常见的张量形状

```python
# 1. 输入嵌入（Embedding）
input_embed = model.embedding(input_ids)
# shape: (batch_size, seq_len, d_model)
# 例如: (32, 128, 768)

# 2. 多头注意力输出
attn_output = attention(Q, K, V)
# shape: (batch_size, seq_len, d_model)

# 3. 注意力权重矩阵
attention_weights = softmax(Q @ K^T / √d_k)
# shape: (batch_size, n_head, seq_len, seq_len)
# 例如: (32, 12, 128, 128)
```

---

### 3.3 张量的基本运算

**形状变换（Reshape）**：
```python
x = torch.randn(2, 3, 4)
x_flat = x.reshape(2*3, 4)  # (6, 4)
```

**维度置换（Permute）**：
```python
x = torch.randn(2, 3, 4)
x_perm = x.permute(1, 0, 2)  # (3, 2, 4)
```

**压缩与扩展（Squeeze/Unsqueeze）**：
```python
x = torch.randn(2, 3, 1)
x_squeezed = x.squeeze()  # (2, 3)

y = torch.randn(2, 3)
z = y.unsqueeze(dim=1)  # (2, 1, 3)
```

---

## 4. 在Transformer中的应用

### 4.1 注意力机制的数学本质

```python
# 完整的注意力计算
def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 点积 + 缩放
    weights = torch.softmax(scores, dim=-1)              # 归一化
    output = weights @ V                                  # 加权求和
    return output, weights
```

**每一步的数学含义**：
1. `Q @ K.T`: 计算相似度（点积）
2. `/ √d_k`: 缩放，防止数值过大
3. `softmax`: 转换为概率分布
4. `@ V`: 加权求和

---

### 4.2 Layer Normalization

**公式**：
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

其中：
- μ = mean(x)
- σ² = variance(x)
- γ, β 是可学习参数

**代码实现**：
```python
import torch.nn as nn

layer_norm = nn.LayerNorm(d_model)
x_normalized = layer_norm(x)
```

**作用**：
- 稳定训练
- 加速收敛
- 让向量范数接近1，使点积纯粹反映方向相似度

---

### 4.3 位置编码

**正弦位置编码**：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**代码实现**：
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)
```

---

## 📝 练习题

### 练习1: 向量点积

计算以下向量的点积：
```python
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([4.0, 5.0, 6.0])

# 你的答案：
dot_product = ?  # 提示：1*4 + 2*5 + 3*6
```

<details>
<summary>点击查看答案</summary>

```python
dot_product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```
</details>

---

### 练习2: 矩阵乘法

计算以下矩阵的乘积：
```python
A = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
B = torch.tensor([[5, 6], [7, 8]])  # (2, 2)

# 你的答案：
C = A @ B = ?
```

<details>
<summary>点击查看答案</summary>

```python
C = [[1*5 + 2*7, 1*6 + 2*8],
     [3*5 + 4*7, 3*6 + 4*8]]
  = [[19, 22],
     [43, 50]]
```
</details>

---

### 练习3: 注意力分数计算

给定：
```python
Q = torch.tensor([[1.0, 0.0]])  # (1, 2)
K = torch.tensor([[1.0, 1.0],   # (2, 2)
                  [0.0, 1.0]])
d_k = 2
```

计算注意力分数（未softmax）：
```python
scores = Q @ K.T / math.sqrt(d_k)
```

<details>
<summary>点击查看答案</summary>

```python
Q @ K.T = [[1*1 + 0*1, 1*0 + 0*1]] = [[1, 0]]
scores = [[1, 0]] / √2 = [[0.707, 0]]
```
</details>

---

## 🔗 深入学习

如需更详细的数学推导和更多练习题，请参考：
- [TRANSFORMER_MATH_FOUNDATION.md](../TRANSFORMER_MATH_FOUNDATION.md) - 完整版（4315行）
- [3Blue1Brown - 线性代数本质](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

---

**祝您学习愉快！**

*最后更新: 2026-05-25 | 版本: v2.0-simplified*
