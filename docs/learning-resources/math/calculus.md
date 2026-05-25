# 📐 微积分与梯度

> **阅读时间**: 20-30分钟  
> **重要性**: ⭐⭐⭐⭐（理解反向传播的关键）

---

## 📋 目录

1. [导数基础](#1-导数基础)
2. [梯度与偏导数](#2-梯度与偏导数)
3. [Softmax函数及其导数](#3-softmax函数及其导数)
4. [链式法则在反向传播中的应用](#4-链式法则在反向传播中的应用)

---

## 1. 导数基础

### 1.1 单变量函数的导数

**定义**：
```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

**几何意义**：函数在某点的切线斜率

**常见函数的导数**：

| 函数 f(x) | 导数 f'(x) |
|-----------|------------|
| x^n | n·x^(n-1) |
| e^x | e^x |
| ln(x) | 1/x |
| sin(x) | cos(x) |
| cos(x) | -sin(x) |

---

### 1.2 求导法则

**线性法则**：
```
(af + bg)' = af' + bg'
```

**乘积法则**：
```
(fg)' = f'g + fg'
```

**链式法则** ⭐⭐⭐⭐⭐：
```
(f(g(x)))' = f'(g(x)) · g'(x)
```

**在深度学习中的应用**：
```python
# 链式法则是反向传播的核心
# 如果 loss = f(g(h(x)))
# 则 dloss/dx = df/dg · dg/dh · dh/dx
```

---

## 2. 梯度与偏导数

### 2.1 偏导数

**定义**：多元函数对其中一个变量求导，其他变量视为常数

**示例**：
```
f(x, y) = x² + xy + y²

∂f/∂x = 2x + y    （y视为常数）
∂f/∂y = x + 2y    （x视为常数）
```

---

### 2.2 梯度（Gradient）⭐⭐⭐⭐⭐

**定义**：所有偏导数组成的向量

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**几何意义**：
- 梯度指向函数增长最快的方向
- 梯度的大小表示增长的速率

**在深度学习中的应用**：
```python
# 梯度下降法
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model(input)
    loss.backward()  # 计算梯度
    optimizer.step() # 沿负梯度方向更新参数
```

---

## 3. Softmax函数及其导数

### 3.1 Softmax定义

**公式**：
```
softmax(x)_i = exp(x_i) / Σⱼ exp(x_j)
```

**性质**：
- 输出和为1（概率分布）
- 保持输入的相对大小关系
- 可导（便于反向传播）

**代码实现**：
```python
import torch
import torch.nn.functional as F

x = torch.tensor([1.0, 2.0, 3.0])
probs = F.softmax(x, dim=0)
print(probs)  # [0.0900, 0.2447, 0.6652]
print(probs.sum())  # 1.0 ✓
```

---

### 3.2 Softmax的导数推导 ⭐⭐⭐⭐⭐

**关键结论**：

当 i = j 时：
```
∂softmax(x)_i / ∂x_j = softmax(x)_i · (1 - softmax(x)_j)
```

当 i ≠ j 时：
```
∂softmax(x)_i / ∂x_j = -softmax(x)_i · softmax(x)_j
```

**矩阵形式**：
```
J = diag(s) - s·s^T
```

其中 s = softmax(x)，J是雅可比矩阵。

**为什么重要？**
- CrossEntropy Loss + Softmax 的梯度非常简洁
- 在Transformer的输出层广泛使用

---

## 4. 链式法则在反向传播中的应用

### 4.1 反向传播的本质

反向传播就是**反复应用链式法则**计算损失函数对每个参数的梯度。

**简单示例**：
```
z = wx + b
a = σ(z)        # σ是激活函数
L = (a - y)²    # 损失函数

# 反向传播：
dL/da = 2(a - y)
da/dz = σ'(z)
dz/dw = x
dz/db = 1

# 链式法则：
dL/dw = dL/da · da/dz · dz/dw
dL/db = dL/da · da/dz · dz/db
```

---

### 4.2 Transformer中的反向传播

**注意力机制的梯度流**：
```python
# 前向传播
scores = Q @ K.T / √d_k
weights = softmax(scores)
output = weights @ V

# 反向传播（自动计算）
loss.backward()

# PyTorch会自动应用链式法则：
# dloss/dQ = dloss/doutput · doutput/dweights · dweights/dscores · dscores/dQ
# dloss/dK = ...
# dloss/dV = ...
```

---

### 4.3 梯度消失与爆炸

**问题**：
- **梯度消失**：深层网络中梯度趋近于0
- **梯度爆炸**：梯度变得非常大

**解决方案**：

1. **Residual Connection**：
   ```python
   output = input + sublayer(input)  # 缓解梯度消失
   ```

2. **Layer Normalization**：
   ```python
   x_normalized = LayerNorm(x)  # 稳定梯度
   ```

3. **梯度裁剪**：
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. **合适的初始化**：
   ```python
   # Xavier初始化
   nn.init.xavier_uniform_(weight)
   ```

---

## 📝 练习题

### 练习1: 计算导数

求以下函数的导数：
```
f(x) = 3x² + 2x + 1
```

<details>
<summary>点击查看答案</summary>

```
f'(x) = 6x + 2
```
</details>

---

### 练习2: 链式法则

给定：
```
y = u²
u = 3x + 1

求 dy/dx
```

<details>
<summary>点击查看答案</summary>

```
dy/du = 2u
du/dx = 3

dy/dx = dy/du · du/dx = 2u · 3 = 6u = 6(3x + 1) = 18x + 6
```
</details>

---

### 练习3: Softmax计算

计算以下输入的softmax：
```python
x = torch.tensor([0.0, 1.0, 2.0])
```

<details>
<summary>点击查看答案</summary>

```python
exp_x = [e^0, e^1, e^2] = [1, 2.718, 7.389]
sum_exp = 1 + 2.718 + 7.389 = 11.107

softmax = [1/11.107, 2.718/11.107, 7.389/11.107]
        = [0.090, 0.245, 0.665]
```
</details>

---

## 🔗 深入学习

如需更详细的数学推导，请参考：
- [TRANSFORMER_MATH_FOUNDATION.md](../TRANSFORMER_MATH_FOUNDATION.md) - 第2部分
- [Stanford CS231n - 反向传播](http://cs231n.github.io/optimization-2/)

---

**祝您学习愉快！**

*最后更新: 2026-05-25 | 版本: v2.0-simplified*
