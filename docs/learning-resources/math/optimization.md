# 📐 优化理论

> **阅读时间**: 15-20分钟  
> **重要性**: ⭐⭐⭐⭐（理解模型训练过程）

---

## 📋 目录

1. [梯度下降法](#1-梯度下降法)
2. [AdamW优化器](#2-adamw优化器)
3. [学习率调度](#3-学习率调度)
4. [正则化技术](#4-正则化技术)

---

## 1. 梯度下降法

### 1.1 基本思想

**目标**：最小化损失函数 L(θ)

**更新规则**：
```
θ = θ - η · ∇L(θ)
```

其中：
- θ: 模型参数
- η: 学习率
- ∇L(θ): 损失函数的梯度

---

### 1.2 变体

**SGD（随机梯度下降）**：
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

**Mini-batch GD**：
- 每次使用一小批数据计算梯度
- 平衡了速度和稳定性

**Momentum**：
```
v = β·v + (1-β)·∇L(θ)
θ = θ - η·v
```

---

## 2. AdamW优化器

### 2.1 Adam算法

**特点**：
- 自适应学习率
- 结合Momentum和RMSProp的优点

**伪代码**：
```
m = β1·m + (1-β1)·g          # 一阶矩估计
v = β2·v + (1-β2)·g²         # 二阶矩估计
m_hat = m / (1-β1^t)         # 偏差修正
v_hat = v / (1-β2^t)
θ = θ - η·m_hat / (√v_hat + ε)
```

---

### 2.2 AdamW vs Adam

**AdamW的改进**：
- 解耦权重衰减（Weight Decay）
- 更好的泛化性能

**代码示例**：
```python
# 推荐：使用AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2,  # 权重衰减
    betas=(0.9, 0.999)
)
```

---

## 3. 学习率调度

### 3.1 Warmup策略 ⭐⭐⭐⭐⭐

**动机**：训练初期使用较小的学习率，避免不稳定

**公式**：
```
lr = base_lr · min(step_num^(-0.5), step_num · warmup_steps^(-1.5))
```

**代码示例**：
```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)

# 在每个step后调用
scheduler.step()
```

---

### 3.2 常见调度策略

**StepLR**：
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

**CosineAnnealing**：
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

**ReduceLROnPlateau**：
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)
```

---

## 4. 正则化技术

### 4.1 L2正则化（权重衰减）

**公式**：
```
L_total = L_original + λ·||w||²
```

**代码实现**：
```python
# 方法1：在优化器中设置
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-2)

# 方法2：手动添加
l2_penalty = sum(torch.sum(p**2) for p in model.parameters())
loss = original_loss + 1e-2 * l2_penalty
```

---

### 4.2 Dropout

**原理**：训练时随机丢弃部分神经元

**代码示例**：
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)  # 10% dropout
    
    def forward(self, x):
        x = self.dropout(x)
        return x
```

---

### 4.3 Early Stopping

**策略**：当验证集损失不再下降时停止训练

**代码示例**：
```python
best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        save_checkpoint()
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break
```

---

## 📝 练习题

### 练习1: 梯度下降更新

给定：
- 初始参数 θ = 5.0
- 学习率 η = 0.1
- 梯度 ∇L(θ) = 2.0

计算一次更新后的θ值。

<details>
<summary>点击查看答案</summary>

```
θ_new = θ - η · ∇L(θ)
      = 5.0 - 0.1 · 2.0
      = 5.0 - 0.2
      = 4.8
```
</details>

---

### 练习2: 学习率调度

使用Warmup策略，给定：
- base_lr = 0.001
- warmup_steps = 100
- 当前step = 50

计算当前学习率。

<details>
<summary>点击查看答案</summary>

```
lr = base_lr · min(step^(-0.5), step · warmup_steps^(-1.5))
   = 0.001 · min(50^(-0.5), 50 · 100^(-1.5))
   = 0.001 · min(0.141, 50 · 0.001)
   = 0.001 · min(0.141, 0.05)
   = 0.001 · 0.05
   = 0.00005
```

在warmup阶段，学习率线性增长。
</details>

---

## 🔗 深入学习

如需更详细的内容，请参考：
- [TRANSFORMER_MATH_FOUNDATION.md](../TRANSFORMER_MATH_FOUNDATION.md) - 第4部分
- 《Deep Learning》- Ian Goodfellow等，第8章

---

**祝您学习愉快！**

*最后更新: 2026-05-25 | 版本: v2.0-simplified*
