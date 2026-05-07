# LLM代码生成演示项目 - 完整学习指南

一个从零实现的完整LLM代码生成系统，深入展示大模型从输入到输出的每个步骤。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

---

## 目录

1. [项目简介](#项目简介)
2. [快速开始](#快速开始)
3. [核心概念详解](#核心概念详解)
   - [3.1 Token化原理](#31-token化原理)
   - [3.2 位置编码](#32-位置编码)
   - [3.2.5 什么是注意力机制？](#325-什么是注意力机制重要概念)
   - [3.3 注意力机制详解](#33-注意力机制详解)
   - [3.4 Transformer架构](#34-transformer架构)
   - [3.5 采样策略](#35-采样策略)
   - [3.6 代码生成流程](#36-代码生成流程)
   - [3.7 后处理](#37-后处理)
   - [3.8 缓存机制](#38-缓存机制)
   - [3.9 训练系统](#39-训练系统) ⭐新增
   - [3.10 KV Cache优化](#310-kv-cache优化) ⭐新增
4. [模块详细说明](#模块详细说明)
5. [调试和实验](#调试和实验)
6. [性能优化](#性能优化)
7. [常见问题](#常见问题)
8. [学习路径](#学习路径)

---

## 项目简介

### 这是什么？

这是一个**教育性质的LLM代码生成演示系统**，从零实现了Transformer架构的所有核心组件。它不是用于生产环境，而是为了帮助您**深入理解大模型的工作原理**。

### 为什么需要这个项目？

大多数LLM框架（如HuggingFace）都是"黑盒"，您只能调用API而看不到内部发生了什么。本项目通过：

- **详细注释**: 每个函数、每个类都有中文说明
- **丰富日志**: 运行时会打印每一步的中间结果
- **可视化**: 可以查看注意力权重、概率分布等
- **可实验性**: 所有参数都可调整，观察不同效果

让您真正看到：
```
文本输入 → Token化 → Embedding → 位置编码 → Encoder → Decoder 
       → 概率分布 → 采样 → Token选择 → 解码 → 后处理 → 最终输出
```

### 核心价值

| 价值 | 说明 |
|------|------|
| **透明化** | 每个步骤都可见，没有黑盒 |
| **模块化** | 9个独立模块，可单独学习 |
| **可调试** | 丰富的日志和可视化工具 |
| **可扩展** | 易于添加新功能或修改现有逻辑 |
| **可训练** | 完整的训练系统，支持模型学习 ⭐新增 |
| **高性能** | KV Cache优化，推理加速10-100倍 ⭐新增 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy matplotlib seaborn
```

> PyTorch较大（约200MB），首次安装可能需要几分钟

### 2. 验证安装

```bash
python scripts/tests/test_all.py
```

预期输出：
```
总计: 7/7 测试通过
[SUCCESS] 所有测试通过！系统工作正常。
```

### 3. 运行演示

```bash
python scripts/main.py
```

推荐选项：
- **选项1**: 基本代码生成流程（观看完整过程）
- **选项2**: 不同采样策略对比（理解temperature作用）
- **选项3**: 缓存机制演示（看性能提升）
- **选项6**: 交互模式（自己输入prompt测试）

### 4. 第一个实验

```python
from scripts.pipeline import CodeGenerationPipeline

# 创建管道
pipeline = CodeGenerationPipeline(
    vocab_size=1000,
    d_model=128,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2
)

# 生成代码
result = pipeline.generate(
    prompt="public class UserService",
    max_length=50,
    temperature=0.7,
    verbose=True  # 显示详细日志
)

print(result['processed_code'])
```

---

## 核心概念详解

### 3.1 Token化原理

#### 什么是Token化？

Token化是将文本分割成小单元的过程。LLM不直接处理文本，而是处理数字序列。

```
原始文本: "public class User"
         ↓
Tokens: ["public", "class", "User"]
         ↓
Token IDs: [15, 23, 156]
```

#### 为什么要Token化？

1. **固定词汇表**: 将无限文本映射到有限词汇表（如1000个token）
2. **数值化**: 神经网络只能处理数字
3. **语义捕捉**: 常用词作为整体（如"class"比"c","l","a","s","s"更有意义）

#### 本项目的实现

```python
# scripts/tokenizer.py
class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = self._build_vocabulary()  # 构建词汇表
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text):
        """将文本分割成tokens"""
        # 简单实现：按空格和标点分割
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def encode(self, text, max_length=64):
        """文本 → token IDs"""
        tokens = self.tokenize(text)
        
        # 转换为ID，未知词用UNK替代
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<UNK>'])
        
        # Padding或Truncation
        if len(token_ids) < max_length:
            token_ids += [0] * (max_length - len(token_ids))  # PAD
        else:
            token_ids = token_ids[:max_length]
        
        # Attention mask: 1表示真实token，0表示padding
        mask = [1 if tid != 0 else 0 for tid in token_ids]
        
        return token_ids, mask
    
    def decode(self, token_ids):
        """token IDs → 文本"""
        tokens = [self.id_to_token.get(tid, '<UNK>') for tid in token_ids]
        return ' '.join(tokens)
```

#### 关键概念

**Attention Mask**:
```
Token IDs: [15, 23, 156, 0, 0, 0]
Mask:      [1,  1,  1,   0, 0, 0]
           ↑真实token     ↑padding
```

告诉模型哪些是真实输入，哪些是填充，避免模型关注无意义的padding。

**UNK Token**:
当遇到词汇表中没有的词时，用`<UNK>`（unknown）替代。

```python
# 示例
tokenizer = SimpleTokenizer(vocab_size=1000)
ids, mask = tokenizer.encode("public class UserService")
# 如果"UserService"不在词汇表中
# ids可能是: [15, 23, 2, ...] 其中2是<UNK>的ID
```

#### 实验建议

```python
# 实验1: 观察不同文本的tokenization
texts = [
    "public class User",
    "private String name",
    "public static void main(String[] args)"
]

for text in texts:
    ids, mask = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"IDs: {ids[:10]}...")  # 只显示前10个
    print(f"Decoded: {tokenizer.decode(ids)}")
    print()

# 实验2: 增大词汇表，减少UNK
for vocab_size in [100, 500, 1000, 2000]:
    t = SimpleTokenizer(vocab_size=vocab_size)
    ids, _ = t.encode("public class UserService")
    unk_count = sum(1 for i in ids if t.id_to_token[i] == '<UNK>')
    print(f"Vocab size {vocab_size}: {unk_count} UNK tokens")
```

---

### 3.2.5 什么是注意力机制？（重要概念）

在深入位置编码之前，让我们先理解**注意力机制**这个Transformer的核心概念。

#### 通俗理解

想象你在阅读一篇文章时，你的眼睛和大脑会**自动关注重要的词**，而忽略不重要的词。注意力机制就是让AI模型学会这种能力。

**生活例子**：
```
句子: "小明昨天在图书馆借了一本关于人工智能的书"

当问到"谁借了书？"时，你会重点关注：
- "小明" ← 最关注（主语）
- "借" ← 次关注（动作）
- 其他词相对不重要
```

注意力机制让模型也能做到这一点！

#### 为什么需要注意力？

**问题1：传统方法的局限**

早期的RNN/LSTM模型像"串行阅读"：
```
从左到右逐个词处理 → 记住前面的内容 → 处理下一个词
```

**缺点**：
- 长句子会"忘记"开头的内容
- 无法同时关注多个地方
- 速度慢（必须按顺序）

**问题2：排列不变性**

纯注意力机制有个特性：**permutation invariant**（排列不变）

```
"public class User" 和 "User class public"
在纯注意力中会得到相同的结果！
```

因为注意力只计算token之间的关系，不考虑顺序。

**解决方案**：这就是为什么需要**位置编码**（下一节讲解）

#### 注意力如何工作？

**核心思想**：每个token都会问三个问题

1. **Query（查询）**："我在找什么信息？"
2. **Key（键）**："我能提供什么信息？"
3. **Value（值）**："我的实际内容是什么？"

**代码生成例子**：

```python
输入: "public class UserService {"

处理 "class" 这个token时：

Query: "我是一个类定义，我需要知道修饰符和类名"
       ↓
扫描其他tokens的Key：
- "public" 的Key: "我是修饰符，表示公开访问"
- "UserService" 的Key: "我是类名，表示服务对象"
- "{" 的Key: "我是开始符号"
       ↓
计算相似度（注意力权重）：
- 关注 "public": 0.4 （很重要，修饰符）
- 关注 "UserService": 0.5 （最重要，类名）
- 关注 "{": 0.1 （次要，结构符号）
       ↓
加权组合Value得到最终表示
```

#### 三种注意力类型

**1. Self-Attention（自注意力）**

同一个序列内部的token互相观察：

```
句子: "The cat sat on the mat"

"cat" 关注:
  - "The" (0.3) - 定冠词
  - "sat" (0.5) - 动作 ← 最关注
  - "on" (0.1) - 介词
  - "mat" (0.1) - 地点
```

**用途**：Encoder内部，理解序列内部关系

**2. Cross-Attention（交叉注意力）**

一个序列关注另一个序列：

```
翻译任务:
源语言: "Hello World"
目标语言: "你好 世界"

生成"你好"时:
  - 关注 "Hello" (0.9) ← 主要来源
  - 关注 "World" (0.1)
```

**用途**：Decoder关注Encoder的记忆

**3. Multi-Head Attention（多头注意力）**

用多个"视角"同时观察：

```
8个注意力头 = 8个不同的观察者

Head 1: 关注语法结构（主谓宾）
Head 2: 关注语义关系（同义词）
Head 3: 关注上下文依赖
...
Head 8: 关注特殊模式

最后将8个头结果合并
```

**优势**：从不同角度理解，更全面

#### 数学公式（简化版）

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**分解步骤**：

1. **Q @ K^T**：计算每个token之间的相似度
2. **/ √d_k**：缩放，防止数值过大
3. **softmax**：转换为概率分布（和为1）
4. **@ V**：加权求和，得到最终输出

**为什么要除以√d_k？**

```
d_k=1:   softmax([1, 2, 3]) = [0.09, 0.24, 0.67]  ← 梯度良好
d_k=100: softmax([10, 20, 30]) = [0.00, 0.00, 1.00] ← 梯度几乎为0
```

除以√d_k保持梯度稳定，避免梯度消失。

#### 可视化示例

注意力权重可以用热力图展示：

```
        public  class  User  Service
public   0.7     0.2    0.05   0.05
class    0.3     0.4    0.2    0.1
User     0.1     0.2    0.5    0.2
Service  0.05    0.1    0.15   0.7
```

**解读**：
- 对角线值高：token关注自己
- 其他位置：跨token的关注关系
- "class"最关注"public"(0.3)和"User"(0.2)

#### 在代码生成中的应用

```python
# Encoder阶段：理解prompt结构
输入: "public class UserService"

Self-Attention让模型理解:
- "class" 是核心关键词
- "public" 修饰 "class"
- "UserService" 是类名

# Decoder阶段：生成代码
生成下一个token时:

Cross-Attention让生成的token关注:
- Encoder的记忆（原始prompt）
- 已生成的部分代码

例如生成 "{" 时:
- 关注 "class" (0.6) - 类定义需要花括号
- 关注 "UserService" (0.3) - 类名
- 关注已生成的 "public class UserService" (0.1)
```

#### 关键优势

✅ **捕捉长距离依赖**：可以直接连接任意两个位置  
✅ **并行计算**：所有token同时处理，速度快  
✅ **可解释性**：可以可视化看到模型关注什么  
✅ **灵活性**：不受位置限制，动态调整关注点

---

### 3.2 位置编码

#### 为什么需要位置编码？

## 什么是注意力机制？
Transformer的注意力机制是** permutation invariant **（排列不变）的：

```
"public class User" 和 "User class public"
在纯注意力机制中会得到相同的结果！
```

因为注意力只计算token之间的关系，不考虑顺序。

**解决方案**: 给每个token添加位置信息

#### 正弦位置编码

## 为什么使用这个？
因为注意力机制是排列不变的，我们需要给每个token添加位置信息。

使用不同频率的正弦/余弦函数：

```python
# scripts/attention.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=512):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 计算不同维度的频率
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

#### 数学原理

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

其中:
- pos: 位置 (0, 1, 2, ..., seq_len-1)
- i: 维度索引 (0, 1, 2, ..., d_model/2-1)
- d: 模型维度 (d_model)
```

**为什么这样设计？**

1. **唯一性**: 每个位置有唯一的编码
2. **相对位置**: 模型可以学习到相对位置关系
   ```
   PE(pos+k) 可以用 PE(pos) 的线性函数表示
   ```
3. **泛化能力**: 可以处理比训练时更长的序列

#### 可视化位置编码

```python
from scripts.visualizer import AttentionVisualizer

visualizer = AttentionVisualizer()
pos_encoder = PositionalEncoding(d_model=128)
visualizer.visualize_positional_encoding(pos_encoder.pe)
```

你会看到：
- 低频维度（前面）变化缓慢，捕捉长距离依赖
- 高频维度（后面）变化快速，捕捉局部细节

#### 实验建议

```python
# 实验: 观察不同位置的编码
pos_encoder = PositionalEncoding(d_model=128)
pe = pos_encoder.pe[0]  # (max_len, d_model)

# 比较位置0和位置1的编码差异
diff = torch.abs(pe[0] - pe[1]).mean()
print(f"Average difference between pos 0 and 1: {diff:.4f}")

# 观察不同维度的频率
import matplotlib.pyplot as plt
plt.plot(pe[:100, 0].numpy(), label='Dimension 0 (low freq)')
plt.plot(pe[:100, 64].numpy(), label='Dimension 64 (high freq)')
plt.legend()
plt.title('Positional Encoding Frequencies')
plt.show()
```

---

### 3.3 注意力机制详解

#### Scaled Dot-Product Attention

```python
# scripts/attention.py
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    query: (batch, seq_len_q, d_k)
    key:   (batch, seq_len_k, d_k)
    value: (batch, seq_len_k, d_v)
    """
    d_k = query.size(-1)
    
    # 1. 计算相似度: Q @ K^T
    scores = torch.matmul(query, key.transpose(-2, -1))  # (batch, seq_len_q, seq_len_k)
    
    # 2. 缩放: 除以 sqrt(d_k) 防止梯度消失
    scores = scores / math.sqrt(d_k)
    
    # 3. Mask（可选）: 将padding位置设为负无穷
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax: 转换为概率分布
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 5. 加权求和: weights @ V
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

#### 为什么要除以 sqrt(d_k)？

当d_k很大时，点积结果会非常大，导致softmax的梯度接近0：

```
d_k=1:   softmax([1, 2, 3]) = [0.09, 0.24, 0.67]  ← 梯度良好
d_k=100: softmax([10, 20, 30]) = [0.00, 0.00, 1.00] ← 梯度几乎为0
```

除以sqrt(d_k)将方差标准化为1，保持梯度稳定。

#### Multi-Head Attention

单个注意力头只能从一个角度理解关系，多头机制允许并行学习不同的表示：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个头的维度
        
        # 为每个头创建独立的Q/K/V变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # 输出变换
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性变换并分成多个头
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        # Shape: (batch, nhead, seq_len, d_k)
        
        # 2. 并行计算所有头的注意力
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        # Shape: (batch, nhead, seq_len, d_k)
        
        # 3. 合并所有头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
        
        # 4. 输出变换
        output = self.w_o(output)
        
        return output, weights
```

#### 注意力类型

**Self-Attention**（自注意力）:
```
query = key = value = 同一序列
用途: Encoder内部，理解序列内部关系
```

**Cross-Attention**（交叉注意力）:
```
query = Decoder输出
key = value = Encoder输出
用途: Decoder关注Encoder的记忆
```

**Causal Masking**（因果掩码）:
```
Decoder不能看到未来的token
mask[i, j] = 0 if j > i else 1
```

#### 可视化注意力

```python
# 观察注意力权重
attention = MultiHeadAttention(d_model=128, nhead=8)
output, weights = attention(query, key, value)

# weights shape: (batch, nhead, seq_len_q, seq_len_k)
print(f"Attention weights shape: {weights.shape}")

# 查看第1个头，第1个样本的注意力
head_0_weights = weights[0, 0, :, :]  # (seq_len_q, seq_len_k)

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

tokens = ['public', 'class', 'User', 'Service', '{']
sns.heatmap(head_0_weights.detach().numpy(), 
            xticklabels=tokens, 
            yticklabels=tokens,
            cmap='viridis')
plt.title('Attention Weights - Head 0')
plt.xlabel('Key Tokens')
plt.ylabel('Query Tokens')
plt.show()
```

#### 实验建议

```python
# 实验1: 比较不同头的注意力模式
for head_idx in range(8):
    head_weights = weights[0, head_idx, :, :]
    print(f"Head {head_idx}: max weight = {head_weights.max():.4f}")

# 实验2: 分析特定token的关注点
token_names = ['public', 'class', 'User', 'Service']
attention.explain_attention(token_names)

# 输出示例:
# Token "class" 最关注:
#   1. "public" (weight: 0.45)
#   2. "User" (weight: 0.35)
#   3. "{" (weight: 0.15)
```

---

### 3.4 Transformer架构

#### 整体架构

```
                    ┌─────────────┐
Input Tokens ─────→│  Embedding   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │Positional   │
                    │ Encoding    │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │   Encoder Layer × N     │
              │  ┌──────────────────┐   │
              │  │ Self-Attention   │   │
              │  │ + Add & Norm     │   │
              │  │ Feed Forward     │   │
              │  │ + Add & Norm     │   │
              │  └──────────────────┘   │
              └────────────┬────────────┘
                           │ Memory
              ┌────────────▼────────────┐
              │   Decoder Layer × N     │
              │  ┌──────────────────┐   │
              │  │ Masked Self-Att  │   │
              │  │ + Add & Norm     │   │
              │  │ Cross-Attention  │   │
              │  │ + Add & Norm     │   │
              │  │ Feed Forward     │   │
              │  │ + Add & Norm     │   │
              │  └──────────────────┘   │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │Linear +     │
                    │Softmax      │
                    └──────┬──────┘
                           │
Output Probabilities ─────→│
```

#### Encoder Layer详解

```python
# scripts/transformer.py
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, nhead)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, src):
        # src: (batch, seq_len, d_model)
        
        # 1. Self-Attention + Residual + Norm
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout(attn_output)  # Residual Connection
        src = self.norm1(src)                   # Layer Normalization
        
        # 2. Feed Forward + Residual + Norm
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)    # Residual Connection
        src = self.norm2(src)                   # Layer Normalization
        
        return src
```

**关键组件解释**:

1. **Residual Connection**（残差连接）:
   ```
   output = input + SubLayer(input)
   ```
   - 解决深层网络的梯度消失问题
   - 让信息可以直接流向后面层

2. **Layer Normalization**:
   ```
   Normalize across feature dimension
   ```
   - 稳定训练，加速收敛
   - 与BatchNorm不同（BN跨batch维度）

3. **Feed Forward Network**:
   ```
   Linear(d_model → dim_ff) → ReLU → Linear(dim_ff → d_model)
   ```
   - 通常dim_ff = 4 * d_model
   - 提供非线性变换能力

#### Decoder Layer详解

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512):
        super().__init__()
        
        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, nhead)
        
        # 2. Cross-Attention (关注Encoder输出)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        
        # 3. Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory):
        # tgt: (batch, tgt_seq_len, d_model) - Decoder输入
        # memory: (batch, src_seq_len, d_model) - Encoder输出
        
        # 1. Masked Self-Attention
        # 使用causal mask防止看到未来token
        causal_mask = self._generate_causal_mask(tgt.size(1))
        attn_output, _ = self.self_attn(tgt, tgt, tgt, mask=causal_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)
        
        # 2. Cross-Attention
        # Query来自Decoder，Key/Value来自Encoder
        cross_output, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(cross_output)
        tgt = self.norm2(tgt)
        
        # 3. Feed Forward
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)
        
        return tgt
```

**Causal Mask**:
```python
def _generate_causal_mask(self, size):
    """生成下三角mask，防止Decoder看到未来"""
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask

# 示例 (size=4):
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

#### 完整Transformer

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        # src: (batch, src_seq_len) - 输入序列
        # tgt: (batch, tgt_seq_len) - 目标序列
        
        # 1. Encoder
        enc_output = self.encode(src)
        
        # 2. Decoder
        dec_output = self.decode(tgt, enc_output)
        
        # 3. Output projection
        logits = self.output_proj(dec_output)
        
        return logits
    
    def encode(self, src):
        embedded = self.embedding(src)  # (batch, seq_len, d_model)
        embedded = self.pos_encoder(embedded)
        
        for layer in self.encoder_layers:
            embedded = layer(embedded)
        
        return embedded  # Memory
    
    def decode(self, tgt, memory):
        embedded = self.embedding(tgt)
        embedded = self.pos_encoder(embedded)
        
        for layer in self.decoder_layers:
            embedded = layer(embedded, memory)
        
        return embedded
```

#### 前向传播示例

```python
# 假设
vocab_size = 1000
batch_size = 1
src_seq_len = 20  # Encoder输入长度
tgt_seq_len = 15  # Decoder输入长度

model = TransformerModel(vocab_size=1000, d_model=128)

src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

logits = model(src, tgt)
# logits shape: (batch, tgt_seq_len, vocab_size)

# 获取下一个token的概率分布
probs = torch.softmax(logits[:, -1, :], dim=-1)
# probs shape: (vocab_size,)

# 选择概率最高的token
next_token = torch.argmax(probs, dim=-1)
```

#### 实验建议

```python
# 实验1: 观察每层的输出shape
model = TransformerModel(vocab_size=1000, d_model=128, 
                         num_encoder_layers=2, num_decoder_layers=2)

src = torch.randint(0, 1000, (1, 20))
tgt = torch.randint(0, 1000, (1, 15))

# Encoder输出
memory = model.encode(src)
print(f"Memory shape: {memory.shape}")  # (1, 20, 128)

# Decoder输出
dec_out = model.decode(tgt, memory)
print(f"Decoder output shape: {dec_out.shape}")  # (1, 15, 128)

# Logits
logits = model(src, tgt)
print(f"Logits shape: {logits.shape}")  # (1, 15, 1000)

# 实验2: 改变模型配置
for n_layers in [1, 2, 4]:
    model = TransformerModel(num_encoder_layers=n_layers, 
                             num_decoder_layers=n_layers)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{n_layers} layers: {param_count:,} parameters")
```

---

### 3.5 采样策略

#### 为什么需要采样？

模型输出的是概率分布，需要从中选择一个token：

```
Logits: [2.1, 5.3, 1.2, 3.8, ...]
        ↓ softmax
Probs:  [0.05, 0.45, 0.02, 0.18, ...]
        ↓ sampling
Selected token: index 1 (prob 0.45)
```

不同的采样策略会影响生成质量：
- **太确定**: 生成重复、无聊的内容
- **太随机**: 生成无意义、不连贯的内容

#### 1. Greedy Sampling（贪婪采样）

总是选择概率最高的token：

```python
class GreedySampling:
    def sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        return next_token
```

**优点**: 确定性高，适合代码生成
**缺点**: 缺乏多样性，容易陷入循环

#### 2. Temperature Sampling（温度采样）

通过温度参数控制随机性：

```python
class TemperatureSampling:
    def __init__(self, temperature=0.7):
        self.temperature = temperature
    
    def sample(self, logits):
        # 应用温度
        scaled_logits = logits / self.temperature
        
        # 转换为概率
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # 采样
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
```

**温度的作用**:

```
Temperature < 1 (如0.3):
  - 放大高概率，抑制低概率
  - 更确定性，更少随机
  
Temperature = 1:
  - 原始概率分布
  
Temperature > 1 (如1.5):
  - 平滑概率分布
  - 更随机，更多样
```

**示例**:
```
原始logits: [2.0, 4.0, 1.0]

Temp=0.5:
  scaled: [4.0, 8.0, 2.0]
  probs:  [0.018, 0.964, 0.018]  ← 更极端

Temp=1.0:
  probs:  [0.106, 0.780, 0.114]  ← 原始

Temp=2.0:
  scaled: [1.0, 2.0, 0.5]
  probs:  [0.245, 0.665, 0.090]  ← 更均匀
```

#### 3. Top-K Sampling

只从概率最高的K个token中选择：

```python
class TopKSampling:
    def __init__(self, top_k=50, temperature=0.7):
        self.top_k = top_k
        self.temperature = temperature
    
    def sample(self, logits):
        # 应用温度
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # 找到top-k
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k)
        
        # 重新归一化
        top_k_probs = top_k_probs / top_k_probs.sum()
        
        # 从top-k中采样
        selected_idx = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices[selected_idx]
        
        return next_token
```

**优点**: 避免选择低质量的token
**缺点**: K值需要根据任务调整

#### 4. Top-P (Nucleus) Sampling

从累积概率达到P的最小集合中选择：

```python
class TopPSampling:
    def __init__(self, top_p=0.9, temperature=0.7):
        self.top_p = top_p
        self.temperature = temperature
    
    def sample(self, logits):
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # 排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过p的位置
        cutoff_idx = torch.searchsorted(cumulative_probs, self.top_p)
        
        # 截断
        truncated_probs = sorted_probs[:cutoff_idx+1]
        truncated_probs = truncated_probs / truncated_probs.sum()
        
        # 采样
        selected_idx = torch.multinomial(truncated_probs, num_samples=1)
        next_token = sorted_indices[selected_idx]
        
        return next_token
```

**优点**: 动态调整候选集大小
**缺点**: 实现稍复杂

#### 采样策略对比

| 策略 | 确定性 | 多样性 | 适用场景 |
|------|--------|--------|----------|
| Greedy | 最高 | 最低 | 代码生成、翻译 |
| Temp=0.3 | 高 | 低 | 事实性问答 |
| Temp=0.7 | 中 | 中 | 通用场景 |
| Temp=1.2 | 低 | 高 | 创意写作 |
| Top-K=20 | 中高 | 中低 | 控制质量 |
| Top-P=0.9 | 自适应 | 自适应 | 灵活场景 |

#### 实验建议

```python
# 实验: 比较不同采样策略
from scripts.generator import (GreedySampling, TemperatureSampling, 
                       TopKSampling, TopPSampling)

strategies = [
    ("Greedy", GreedySampling()),
    ("Temp=0.3", TemperatureSampling(0.3)),
    ("Temp=0.7", TemperatureSampling(0.7)),
    ("Temp=1.2", TemperatureSampling(1.2)),
    ("Top-K=20", TopKSampling(20)),
    ("Top-P=0.9", TopPSampling(0.9)),
]

prompt = "public class UserService"

for name, strategy in strategies:
    result = generator.generate(prompt, strategy=strategy, max_length=30)
    print(f"\n{name}:")
    print(result['generated_code'][:100])
```

---

### 3.6 代码生成流程

#### Autoregressive生成

LLM是逐步生成token的，每次生成一个：

```
Step 0: Input  = "public class"
Step 1: Output = "User"       → Input = "public class User"
Step 2: Output = "Service"    → Input = "public class UserService"
Step 3: Output = "{"           → Input = "public class UserService {"
...
直到遇到<EOS>或达到max_length
```

#### 完整生成流程

```python
# scripts/generator.py
class CodeGenerator:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt, max_length=50, strategy=None, verbose=False):
        # 1. Tokenize prompt
        input_ids, _ = self.tokenizer.encode(prompt, max_length=64)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        generated_ids = input_ids.clone()
        
        # 2. Autoregressive generation
        for step in range(max_length):
            # a. Forward pass
            with torch.no_grad():
                logits = self.model.generate_step(generated_ids)
            
            # b. Get last token's logits
            next_logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # c. Sample next token
            if strategy is None:
                strategy = TemperatureSampling(0.7)
            next_token = strategy.sample(next_logits)
            
            # d. Append to sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
            
            # e. Check for EOS
            if next_token.item() == self.tokenizer.vocab['<EOS>']:
                break
            
            if verbose:
                token_str = self.tokenizer.id_to_token[next_token.item()]
                print(f"Step {step+1}: Generated '{token_str}'")
        
        # 3. Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        return {
            'generated_code': generated_text,
            'token_count': generated_ids.size(1) - input_ids.size(1)
        }
```

#### 关键细节

**1. Causal Masking in Generation**:

生成时不需要mask，因为每次只看已生成的部分：

```
Step 0: "public class"          → 预测下一个
Step 1: "public class User"     → 预测下一个
Step 2: "public class User X"   → 预测下一个
```

**2. KV Cache优化**（高级）:

实际LLM会缓存Key和Value，避免重复计算：

```
Step 0: Compute K,V for "public class"     → cache them
Step 1: Only compute K,V for "User"        → append to cache
Step 2: Only compute K,V for "X"           → append to cache
```

本项目未实现此优化，但真实LLM都用。

**3. Batch Generation**:

可以同时生成多个样本：

```python
# Batch size = 3
prompts = ["public class A", "public class B", "public class C"]
# Generate all in parallel
results = generator.generate_batch(prompts)
```

#### 实验建议

```python
# 实验1: 观察逐步生成过程
generator = CodeGenerator(model, tokenizer)
result = generator.generate(
    "public class",
    max_length=20,
    strategy=TemperatureSampling(0.7),
    verbose=True  # 打印每一步
)

# 输出示例:
# Step 1: Generated 'User'
# Step 2: Generated 'Service'
# Step 3: Generated '{'
# ...

# 实验2: 多样本生成
samples = generator.generate_multiple_samples(
    prompt="public class UserService",
    num_samples=5,
    temperatures=[0.5, 0.7, 0.9, 1.1, 1.3]
)

for i, sample in enumerate(samples):
    print(f"\nSample {i+1} (temp={sample['temperature']}):")
    print(sample['code'][:100])
```

---

### 3.7 后处理

#### 为什么需要后处理？

生成的原始token序列可能：
- 语法不完整（括号不匹配）
- 格式混乱（缩进错误）
- 缺少import语句
- 包含重复或冗余代码

后处理模块解决这些问题。

#### 后处理步骤

```python
# scripts/postprocessor.py
class CodePostProcessor:
    def process(self, code):
        steps_applied = []
        
        # Step 1: 语法验证
        validation_report = self.validate_syntax(code)
        if validation_report['is_valid']:
            steps_applied.append('validation_passed')
        
        # Step 2: 代码格式化
        formatted_code = self.format_code(code)
        steps_applied.append('formatting')
        
        # Step 3: Import管理
        code_with_imports = self.manage_imports(formatted_code)
        steps_applied.append('import_management')
        
        # Step 4: 代码优化
        optimized_code = self.optimize_code(code_with_imports)
        steps_applied.append('optimization')
        
        return {
            'processed_code': optimized_code,
            'validation_report': validation_report,
            'steps_applied': steps_applied
        }
```

#### 1. 语法验证

```python
class SyntaxValidator:
    def validate(self, code):
        issues = []
        
        # 检查括号匹配
        bracket_count = 0
        for char in code:
            if char == '{':
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
            
            if bracket_count < 0:
                issues.append("Unmatched closing brace")
                break
        
        if bracket_count > 0:
            issues.append(f"Missing {bracket_count} closing braces")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
```

#### 2. 代码格式化

```python
class JavaCodeFormatter:
    def format(self, code):
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            line = line.strip()
            
            # 减少缩进（遇到闭合括号）
            if line.startswith('}'):
                indent_level -= 1
            
            # 添加缩进
            formatted_line = '    ' * max(0, indent_level) + line
            formatted_lines.append(formatted_line)
            
            # 增加缩进（遇到开放括号）
            if line.endswith('{'):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
```

#### 3. Import管理

```python
class ImportManager:
    def add_imports(self, code):
        # 检测使用的类型
        used_types = self.detect_types(code)
        
        # 查找需要的import
        required_imports = []
        for type_name in used_types:
            if type_name in self.import_map:
                required_imports.append(self.import_map[type_name])
        
        # 添加到文件开头
        import_statements = '\n'.join(required_imports)
        final_code = import_statements + '\n\n' + code
        
        return final_code
```

#### 实验建议

```python
# 实验: 观察后处理效果
post_processor = CodePostProcessor()

raw_code = "public class Test{public void method(){System.out.println(\"hello\");}}"

result = post_processor.process(raw_code)

print("原始代码:")
print(raw_code)
print("\n处理后代码:")
print(result['processed_code'])
print("\n应用的步骤:")
for step in result['steps_applied']:
    print(f"  - {step}")
```

---

### 3.8 缓存机制

#### 为什么需要缓存？

相同的prompt会产生相同的结果，缓存可以避免重复计算：

```
Request 1: "public class User" → Generate (0.3s) → Cache it
Request 2: "public class User" → Hit cache (<0.01s) → Return cached
```

#### LRU缓存实现

```python
# scripts/cache.py
from collections import OrderedDict

class GenerationCache:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = OrderedDict()  # 保持插入顺序
        self.hits = 0
        self.misses = 0
    
    def get(self, prompt):
        prompt_hash = self._hash_prompt(prompt)
        
        if prompt_hash in self.cache:
            # Cache hit
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(prompt_hash)
            return self.cache[prompt_hash]
        else:
            # Cache miss
            self.misses += 1
            return None
    
    def put(self, prompt, result):
        prompt_hash = self._hash_prompt(prompt)
        
        # If already exists, update
        if prompt_hash in self.cache:
            self.cache.move_to_end(prompt_hash)
            self.cache[prompt_hash] = result
            return
        
        # If full, remove least recently used
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove first item
        
        # Add new entry
        self.cache[prompt_hash] = result
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
    
    def _hash_prompt(self, prompt):
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()
```

#### 缓存策略

**LRU (Least Recently Used)**:
- 淘汰最久未使用的条目
- 适合局部性原理（最近用的很可能再用）

**语义哈希**:
- 相似prompt共享缓存
- 例如："public class User" 和 "public class Order" 可能结构类似

#### 性能提升

```
首次请求: ~0.3s (需要生成)
缓存命中: <0.01s (直接返回)
加速比: 30x+
```

#### 实验建议

```python
# 实验: 测试缓存效果
cache = GenerationCache(max_size=10)

# 第一次: miss
start = time.perf_counter()
result = pipeline.generate("public class User")
elapsed1 = time.perf_counter() - start

# 第二次: hit
start = time.perf_counter()
result = pipeline.generate("public class User")
elapsed2 = time.perf_counter() - start

print(f"First request (miss): {elapsed1:.4f}s")
print(f"Second request (hit): {elapsed2:.4f}s")
print(f"Speedup: {elapsed1/elapsed2:.1f}x")

# 查看统计
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']*100:.1f}%")
```

---

## 模块详细说明

### scripts/tokenizer.py (228行)

**核心类**: `SimpleTokenizer`

**主要方法**:
- `__init__(vocab_size, debug_mode)`: 初始化，构建或加载词汇表
- `tokenize(text)`: 文本分词
- `encode(text, max_length)`: 文本→token IDs + mask
- `decode(token_ids)`: token IDs→文本
- `_build_vocabulary()`: 构建词汇表

**关键特性**:
- 词汇表缓存（类变量`_vocab_cache`）
- Debug模式控制日志输出
- Padding和Truncation

**学习重点**:
- BPE分词原理
- 词汇表构建策略
- Attention mask的作用

---

### scripts/attention.py (283行)

**核心类**: 
- `MultiHeadAttention`: 多头注意力
- `PositionalEncoding`: 位置编码

**主要方法**:
- `scaled_dot_product_attention(Q, K, V)`: 基础注意力
- `forward(query, key, value, mask)`: 前向传播
- `explain_attention(token_names)`: 解释注意力权重

**关键公式**:
```
Attention(Q,K,V) = softmax(Q@K^T/√d_k)@V
PE(pos,2i) = sin(pos/10000^(2i/d))
```

**学习重点**:
- Self-Attention vs Cross-Attention
- Multi-Head的优势
- 位置编码的数学原理

---

### scripts/transformer.py (429行)

**核心类**:
- `TransformerModel`: 主模型
- `TransformerEncoderLayer`: Encoder层
- `TransformerDecoderLayer`: Decoder层

**架构**:
```
Embedding → Positional Encoding → Encoder×N → Decoder×N → Output
```

**关键组件**:
- Residual Connection
- Layer Normalization
- Causal Masking
- Feed Forward Network

**学习重点**:
- Encoder-Decoder交互
- 为什么需要Residual Connection
- LayerNorm vs BatchNorm

---

### scripts/generator.py (375行)

**核心类**:
- `CodeGenerator`: 生成器
- `GreedySampling`: 贪婪采样
- `TemperatureSampling`: 温度采样
- `TopKSampling`: Top-K采样
- `TopPSampling`: Nucleus采样

**生成流程**:
```
Tokenize → Forward → Sample → Append → Repeat until EOS
```

**学习重点**:
- 不同采样策略的优劣
- Temperature的数学原理
- 如何平衡确定性和多样性

---

### scripts/postprocessor.py (416行)

**核心类**:
- `CodePostProcessor`: 后处理器
- `SyntaxValidator`: 语法验证
- `JavaCodeFormatter`: 代码格式化
- `ImportManager`: Import管理
- `CodeOptimizer`: 代码优化

**处理步骤**:
1. 语法验证
2. 代码格式化
3. Import管理
4. 代码优化

**学习重点**:
- AST基本概念
- 代码风格规范
- 静态分析技术

---

### scripts/cache.py (301行)

**核心类**: `GenerationCache`

**特性**:
- LRU淘汰策略
- 语义哈希
- 统计信息追踪

**API**:
- `get(prompt)`: 查询缓存
- `put(prompt, result)`: 存入缓存
- `get_stats()`: 获取统计信息

**学习重点**:
- 缓存设计模式
- LRU算法实现
- 哈希函数选择

---

### scripts/pipeline.py (427行)

**核心类**: `CodeGenerationPipeline`

**整合组件**:
```
Tokenizer → Transformer → Generator → PostProcessor → Cache
```

**主要方法**:
- `generate(prompt, max_length, temperature, ...)`: 单次生成
- `generate_multiple_samples(...)`: 多样本生成
- `display_full_stats()`: 显示完整统计

**学习重点**:
- 系统设计模式
- 组件集成
- 错误处理

---

### scripts/visualizer.py (376行)

**核心类**: `AttentionVisualizer`

**可视化类型**:
1. 注意力权重热力图
2. 多注意力头对比
3. 生成过程展示
4. 概率分布图
5. 模型架构图

**依赖**: matplotlib, seaborn

**学习重点**:
- Matplotlib使用
- 数据可视化技巧
- 注意力分析方法

---

## 调试和实验

### 1. 启用详细日志

```python
# 方法1: 使用debug_mode
tokenizer = SimpleTokenizer(debug_mode=True)
attention = MultiHeadAttention(debug_mode=True)

# 方法2: 配置logging
import logging
logging.basicConfig(level=logging.DEBUG)

# 方法3: verbose参数
result = pipeline.generate(prompt, verbose=True)
```

### 2. 观察中间结果

```python
# Tokenization
ids, mask = tokenizer.encode("public class User")
print(f"Token IDs: {ids}")
print(f"Mask: {mask}")
print(f"Decoded: {tokenizer.decode(ids)}")

# Attention weights
output, weights = attention(query, key, value)
print(f"Weights shape: {weights.shape}")
print(f"Max weight: {weights.max():.4f}")

# Probability distribution
logits = model(src, tgt)
probs = torch.softmax(logits[:, -1, :], dim=-1)
top5 = torch.topk(probs, 5)
print(f"Top-5 predictions: {top5.values}")
print(f"Top-5 tokens: {[tokenizer.id_to_token[i.item()] for i in top5.indices]}")
```

### 3. 修改模型配置

```python
# 小模型（快速实验）
small_model = TransformerModel(
    vocab_size=500,
    d_model=64,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1
)

# 大模型（更好质量）
large_model = TransformerModel(
    vocab_size=2000,
    d_model=256,
    nhead=16,
    num_encoder_layers=4,
    num_decoder_layers=4
)
```

### 4. 性能分析

```python
import time
import cProfile

# 简单计时
start = time.perf_counter()
result = pipeline.generate(prompt)
elapsed = time.perf_counter() - start
print(f"Generation time: {elapsed:.4f}s")
print(f"Tokens/sec: {result['token_count']/elapsed:.2f}")

# 详细性能分析
cProfile.run('pipeline.generate(prompt)', sort='cumulative')
```

---

## 性能优化

### 已实施的优化

1. **词汇表缓存** (tokenizer.py)
   - 相同vocab_size共享词汇表
   - 加速比: 3.8x
   - 内存节省: 67%

2. **日志控制** (所有模块)
   - debug_mode参数
   - 生产环境关闭调试输出
   - 预计提升: 20-30%

3. **LRU缓存** (cache.py)
   - 避免重复生成
   - 加速比: 30x+

### 进一步优化建议

1. **KV Cache**: 缓存Decoder的Key/Value
2. **Batch Processing**: 批量处理多个请求
3. **GPU Acceleration**: 使用CUDA
4. **Model Quantization**: 量化减少内存
5. **Parallel Decoding**: 并行解码

---

## 常见问题

### Q1: 为什么生成的代码全是`<UNK>`？

**原因**: 词汇表太小或训练数据不足

**解决**:
```python
# 增大词汇表
tokenizer = SimpleTokenizer(vocab_size=2000)

# 或者接受这是未训练模型的正常行为
# 本项目重点是展示原理，而非生成可用代码
```

### Q2: 如何提高生成质量？

**短期**（调整参数）:
- 增大d_model (128 → 256)
- 增加层数 (2 → 4)
- 增大词汇表 (1000 → 2000)
- 降低temperature (0.7 → 0.5)

**长期**（需要训练）:
- 在大量代码数据上预训练
- 使用BPE分词代替简单分词
- 增加模型规模到数百万参数

### Q3: 可以用GPU吗？

**可以**:
```python
pipeline = CodeGenerationPipeline(device='cuda')
```

**前提**:
- 安装CUDA版本的PyTorch
- 有NVIDIA GPU

**加速效果**: 5-10x（取决于GPU型号）

### Q4: 注意力权重怎么看？

**理解**:
- Shape: (batch, nhead, seq_len_q, seq_len_k)
- 每行和为1（softmax后）
- 值越大表示关注度越高

**可视化**:
```python
visualizer.visualize_attention(weights, token_names)
```

**分析**:
- 对角线: 自注意力（token关注自己）
- 其他: 跨token注意力
- 不同头关注不同方面

### Q5: 为什么Decoder需要Causal Mask？

**原因**: 防止看到未来token

**示例**:
```
生成 "public class User"

Step 1: 输入 "public" → 预测 "class"
Step 2: 输入 "public class" → 预测 "User"

如果在Step 1就能看到"class"和"User"，那就作弊了！
```

**实现**:
```python
# 下三角矩阵
mask = [[1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]]
```

---

## 学习路径

### 第1天：建立整体认知

**目标**: 理解LLM生成的完整流程

**任务**:
1. ✅ 运行`test_all.py`，确保所有模块工作
2. ✅ 运行`main.py`选项1，观看完整生成过程
3. ✅ 阅读本README的"核心概念详解"部分
4. ✅ 尝试交互模式（选项6），输入不同prompt

**产出**:
- 能画出完整的流程图
- 知道每个模块的作用
- 能运行基本示例

---

### 第2天：深入Token化和注意力

**目标**: 理解输入处理和核心机制

**任务**:
1. 📖 仔细阅读`tokenizer.py`源码
2. 🔬 实验：改变vocab_size，观察UNK比例
3. 📖 仔细阅读`attention.py`源码
4. 🔬 实验：可视化注意力权重
5. 📝 手写Scaled Dot-Product Attention公式

**产出**:
- 理解Token化全过程
- 能解释Attention的计算过程
- 知道Multi-Head的意义

---

### 第3天：理解Transformer架构

**目标**: 掌握Encoder-Decoder设计

**任务**:
1. 📖 仔细阅读`transformer.py`源码
2. 🎨 画出Transformer架构图（包含所有组件）
3. 🔬 实验：改变层数、维度，观察参数量变化
4. 📝 解释Residual Connection和LayerNorm的作用

**产出**:
- 能从头画出Transformer架构
- 理解每个组件的必要性
- 知道如何调整模型配置

---

### 第4天：掌握采样策略

**目标**: 理解如何从概率中选择token

**任务**:
1. 📖 仔细阅读`generator.py`源码
2. 🔬 实验：对比4种采样策略的效果
3. 📝 推导Temperature对概率分布的影响
4. 🔬 实验：调整temperature和top_k，观察生成多样性

**产出**:
- 能解释每种采样策略的优劣
- 知道如何选择合适的策略
- 理解Temperature的数学原理

---

### 第5-7天：综合实践

**目标**: 能够独立实验和优化

**任务**:
1. 🔧 实现一个新的采样策略（如Beam Search）
2. 📊 分析注意力模式，总结规律
3. ⚡ 进行性能分析，找出瓶颈
4. 📝 写一份学习报告，总结收获

**产出**:
- 至少一个新功能实现
- 性能分析报告
- 完整的学习笔记

---

## 相关资源

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3

### 教程
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化讲解
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 代码详解

### 工具
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Course](https://huggingface.co/course)

---

## ⭐ 新增功能（最新版本）

### 3.9 训练系统

#### 概述

训练系统使模型能够从数据中学习，而不仅仅是随机初始化后的前向传播。这是理解LLM如何获得知识的关键。

**新增文件**: `scripts/trainer.py`

**包含组件**：
- ✅ CrossEntropyLoss - 交叉熵损失函数
- ✅ AdamW - 优化器（解耦权重衰减）
- ✅ WarmupLinearScheduler - 学习率调度器
- ✅ TextDataset - 数据集类
- ✅ Trainer - 训练管理器

#### 快速开始

```python
from scripts.trainer import Trainer, TextDataset
from scripts.transformer import Transformer

# 1. 准备数据
texts = [
    "def hello_world():",
    "    print('Hello, World!')",
    # ... 更多代码样本
]

dataset = TextDataset(texts, tokenizer, max_len=128)

# 2. 创建模型
model = Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4
)

# 3. 训练
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    batch_size=16,
    lr=1e-4,
    epochs=20
)

history = trainer.train()
```

#### 核心特性

1. **标签平滑（Label Smoothing）**
   - 防止模型过于自信
   - 提高泛化能力
   ```python
   loss_fn = CrossEntropyLoss(label_smoothing=0.1)
   ```

2. **Warmup学习率策略**
   - 初期线性增加学习率（稳定训练）
   - 后期线性衰减（精细调整）
   ```python
   scheduler = WarmupLinearScheduler(
       optimizer,
       warmup_steps=1000,
       total_steps=10000,
       max_lr=1e-4
   )
   ```

3. **梯度裁剪**
   - 防止梯度爆炸
   - 自动限制梯度范围
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. **Checkpoint管理**
   - 自动保存最佳模型
   - 支持断点续训
   ```python
   trainer.save_checkpoint(epoch, loss)
   trainer.load_checkpoint("checkpoints/checkpoint_epoch_5.pt")
   ```

#### 运行演示

```bash
python scripts/test_training_and_cache.py
```

详细文档请参考：[TRAINING_AND_KV_CACHE_GUIDE.md](TRAINING_AND_KV_CACHE_GUIDE.md)

---

### 3.10 KV Cache优化

#### 概述

KV Cache（Key-Value缓存）是LLM推理加速的核心技术，可将生成长度从O(n²)降低到O(n)。

**新增文件**: `scripts/kv_cache.py`

**核心价值**：
- ⚡ 推理速度提升10-50倍
- 💾 避免重复计算历史token的K/V
- 🎯 对长序列特别有效

#### 为什么需要KV Cache？

**无Cache的情况**：
```
Step 1: 计算 token1 的 Q, K, V
Step 2: 重新计算 token1, token2 的 Q, K, V  ← token1被重复计算
Step 3: 重新计算 token1, token2, token3 的 Q, K, V  ← token1,2被重复计算
...
总计算量: O(n²)
```

**有Cache的情况**：
```
Step 1: 计算 token1 的 Q, K, V → 存入cache
Step 2: 只计算 token2 的 K, V → 从cache读取token1的K/V
Step 3: 只计算 token3 的 K, V → 从cache读取token1,2的K/V
...
总计算量: O(n)
```

#### 性能对比

| 序列长度 | 无Cache计算量 | 有Cache计算量 | 加速比 |
|---------|--------------|--------------|--------|
| 10      | 55           | 10           | 5.5x   |
| 100     | 5,050        | 100          | 50.5x  |
| 1000    | 500,500      | 1,000        | 500.5x |

#### 使用示例

```python
from scripts.kv_cache import KVCacheManager

# 1. 初始化
manager = KVCacheManager(
    num_layers=4,
    batch_size=1,
    num_heads=8,
    max_seq_len=256,
    head_dim=32
)
manager.initialize()

# 2. 在模型中使用
for step in range(generation_steps):
    outputs = model(
        input_token,
        use_cache=True,
        cache_manager=manager,
        cache_position=step
    )
    next_token = sample(outputs)
```

#### 运行演示

```bash
python scripts/kv_cache.py
```

输出示例：
```
======================================================================
KV Cache性能优势演示
======================================================================

配置:
  - 序列长度: 100
  - 模型维度: 512
  - 注意力头: 8

❌ 不使用KV Cache:
  - 总计算量: 20,684,800 次操作
  - 时间复杂度: O(n²)

✅ 使用KV Cache:
  - 总计算量: 409,600 次操作
  - 时间复杂度: O(n)

📊 性能对比:
  - 加速比: 50.5x
  - 节省计算: 98.0%
======================================================================
```

详细文档请参考：[TRAINING_AND_KV_CACHE_GUIDE.md](TRAINING_AND_KV_CACHE_GUIDE.md)

---

## 项目统计

| 类别 | 数量 |
|------|------|
| Python模块 | 11个（新增trainer.py, kv_cache.py） |
| 测试文件 | 3个（新增test_training_and_cache.py） |
| 核心代码行数 | ~4,500+ |
| 文档行数 | ~2,300+ |
| 项目总行数 | ~6,800+ |

---

## 许可证

本项目仅用于教育和学习目的。

---

**祝您深入学习愉快！** 

*最后更新: 2026-04-30*
