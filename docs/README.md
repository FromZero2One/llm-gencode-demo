# LLM代码生成演示项目

一个从零实现的完整LLM代码生成系统，深入展示大模型从输入到输出的每个步骤。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

---

## 📋 目录

1. [项目简介](#项目简介)
2. [快速开始](#快速开始)
3. [核心概念](#核心概念)
   - [Token化](#token化)
   - [注意力机制](#注意力机制)
   - [Transformer架构](#transformer架构)
   - [采样策略](#采样策略)
   - [训练系统](#训练系统) ⭐
   - [KV Cache](#kv-cache) ⭐
4. [模块说明](#模块说明)
5. [调试工具](#调试工具)
6. [常见问题](#常见问题)
7. [学习路径](#学习路径)
8. [🎓 新手学习指南](#-新手学习指南) ⭐新增

---

## 项目简介

### 这是什么？

这是一个**教育性质的LLM代码生成演示系统**，从零实现了Transformer架构的所有核心组件。它不是用于生产环境，而是为了帮助您**深入理解大模型的工作原理**。

### 核心价值

| 价值 | 说明 |
|------|------|
| **透明化** | 每个步骤都可见，没有黑盒 |
| **模块化** | 11个独立模块，可单独学习 |
| **可调试** | 丰富的日志和可视化工具 |
| **可扩展** | 易于添加新功能或修改现有逻辑 |
| **可训练** | 完整的训练系统 ⭐新增 |
| **高性能** | KV Cache优化，推理加速10-100倍 ⭐新增 |

### 项目统计

- **Python模块**: 11个
- **核心代码**: ~4,500+ 行
- **文档**: ~2,000+ 行
- **测试**: 4个测试文件，全部通过

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy matplotlib seaborn
```

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

## 核心概念

### Token化

#### 什么是Token化？

Token化是将文本分割成小单元的过程。LLM不直接处理文本，而是处理数字序列。

```
原始文本: "public class User"
         ↓
Tokens: ["public", "class", "User"]
         ↓
Token IDs: [15, 23, 156]
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

---

### 注意力机制

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

#### 三种注意力类型

**1. Self-Attention（自注意力）**
同一个序列内部的token互相观察，用于Encoder内部理解序列内部关系。

**2. Cross-Attention（交叉注意力）**
一个序列关注另一个序列，用于Decoder关注Encoder的记忆。

**3. Multi-Head Attention（多头注意力）**
用多个"视角"同时观察，从不同角度理解，更全面。

#### 数学公式（简化版）

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**分解步骤**：
1. **Q @ K^T**：计算每个token之间的相似度
2. **/ √d_k**：缩放，防止数值过大
3. **softmax**：转换为概率分布（和为1）
4. **@ V**：加权求和，得到最终输出

---

### Transformer架构

#### 整体架构

```
Input Tokens → Embedding → Positional Encoding 
             → Encoder×N → Decoder×N 
             → Linear + Softmax → Output Probabilities
```

#### Encoder Layer

每个Encoder层包含：
1. **Self-Attention**: 让每个token关注序列中的所有其他token
2. **Residual Connection**: 原始输入 + 注意力输出，缓解梯度消失
3. **Layer Normalization**: 归一化，稳定训练
4. **Feed Forward Network**: 非线性特征变换

#### Decoder Layer

每个Decoder层包含：
1. **Masked Self-Attention**: 防止看到未来token
2. **Cross-Attention**: 关注Encoder的输出
3. **Feed Forward Network**: 非线性特征变换

---

### 采样策略

#### 为什么需要采样？

模型输出的是概率分布，需要从中选择一个token。不同的采样策略会影响生成质量。

#### 主要策略

| 策略 | 确定性 | 多样性 | 适用场景 |
|------|--------|--------|----------|
| Greedy | 最高 | 最低 | 代码生成、翻译 |
| Temperature=0.3 | 高 | 低 | 事实性问答 |
| Temperature=0.7 | 中 | 中 | 通用场景 |
| Top-K=50 | 中高 | 中低 | 控制质量 |
| Top-P=0.9 | 自适应 | 自适应 | 灵活场景 |

#### Temperature的作用

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

---

### 训练系统 ⭐新增

#### 概述

训练系统使模型能够从数据中学习，而不仅仅是随机初始化后的前向传播。

**核心组件**：
- ✅ CrossEntropyLoss - 交叉熵损失函数
- ✅ AdamW - 优化器（解耦权重衰减）
- ✅ WarmupLinearScheduler - 学习率调度器
- ✅ TextDataset - 数据集类
- ✅ Trainer - 训练管理器

#### 快速开始

```python
from scripts.trainer import Trainer, TextDataset
from scripts.transformer import TransformerModel

# 1. 准备数据
texts = ["def hello_world():", "    print('Hello, World!')"]
dataset = TextDataset(texts, tokenizer, max_len=128)

# 2. 创建模型
model = TransformerModel(vocab_size=1000, d_model=256, nhead=8)

# 3. 训练
trainer = Trainer(model=model, train_dataset=dataset, batch_size=16, lr=1e-4, epochs=20)
history = trainer.train()
```

#### 核心特性

1. **标签平滑（Label Smoothing）**: 防止模型过于自信
2. **Warmup学习率策略**: 初期线性增加，后期线性衰减
3. **梯度裁剪**: 防止梯度爆炸
4. **Checkpoint管理**: 自动保存最佳模型

---

### KV Cache ⭐新增

#### 概述

KV Cache（Key-Value缓存）是LLM推理加速的核心技术，可将生成长度从O(n²)降低到O(n)。

**核心价值**：
- ⚡ 推理速度提升10-50倍
- 💾 避免重复计算历史token的K/V
- 🎯 对长序列特别有效

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
manager = KVCacheManager(num_layers=4, batch_size=1, num_heads=8, max_seq_len=256, head_dim=32)
manager.initialize()

# 2. 在模型中使用
for step in range(generation_steps):
    outputs = model(input_token, use_cache=True, cache_manager=manager, cache_position=step)
    next_token = sample(outputs)
```

---

## 模块说明

### 项目结构

```
llm-codegen-demo/
├── docs/                    # 文档目录
│   ├── README.md           # 主文档
│   ├── QUICK_START.md      # 快速开始
│   └── LEARNING_GUIDE.md   # 学习指南
├── scripts/                # 脚本目录
│   ├── main.py             # 主程序入口
│   ├── tokenizer.py        # Tokenizer (471行)
│   ├── attention.py        # 注意力机制 (283行)
│   ├── transformer.py      # Transformer (970行)
│   ├── generator.py        # 代码生成器 (381行)
│   ├── postprocessor.py    # 后处理器 (416行)
│   ├── cache.py            # 缓存机制 (301行)
│   ├── pipeline.py         # 完整管道 (442行)
│   ├── visualizer.py       # 可视化工具 (376行)
│   ├── trainer.py          # 训练系统 (~600行)
│   ├── kv_cache.py         # KV Cache (~400行)
│   ├── logger.py           # 日志管理
│   └── tests/              # 测试目录
│       ├── test_all.py         # 完整测试
│       ├── test_performance.py # 性能测试
│       ├── test_new_features.py # 新功能测试
│       ├── verify_math.py      # 数学验证
│       ├── debug_tokenizer.py  # Tokenizer调试
│       └── tokenizer_interactive.py  # 交互式调试
├── run_demo.py             # 项目入口脚本
├── requirements.txt        # 依赖包列表
└── .gitignore              # Git忽略文件
```

### 核心模块

| 模块 | 行数 | 功能 |
|------|------|------|
| [tokenizer.py](file:///home/wsm/codes/llm-gencode-demo/scripts/tokenizer.py) | 471 | 文本分词器 |
| [attention.py](file:///home/wsm/codes/llm-gencode-demo/scripts/attention.py) | 283 | 多头注意力机制 |
| [transformer.py](file:///home/wsm/codes/llm-gencode-demo/scripts/transformer.py) | 970 | Transformer完整架构 |
| [generator.py](file:///home/wsm/codes/llm-gencode-demo/scripts/generator.py) | 381 | 代码生成器 |
| [postprocessor.py](file:///home/wsm/codes/llm-gencode-demo/scripts/postprocessor.py) | 416 | 代码后处理 |
| [cache.py](file:///home/wsm/codes/llm-gencode-demo/scripts/cache.py) | 301 | 缓存机制 |
| [pipeline.py](file:///home/wsm/codes/llm-gencode-demo/scripts/pipeline.py) | 442 | 完整流程整合 |
| [visualizer.py](file:///home/wsm/codes/llm-gencode-demo/scripts/visualizer.py) | 376 | 可视化工具 |
| [trainer.py](file:///home/wsm/codes/llm-gencode-demo/scripts/trainer.py) | ~600 | 训练系统 |
| [kv_cache.py](file:///home/wsm/codes/llm-gencode-demo/scripts/kv_cache.py) | ~400 | KV Cache优化 |

### 辅助模块

- **[logger.py](file:///home/wsm/codes/llm-gencode-demo/scripts/logger.py)**: 统一日志管理
- **[main.py](file:///home/wsm/codes/llm-gencode-demo/scripts/main.py)**: 演示入口

---

## 调试工具

### 1. 日志系统

**启用详细日志**:
```python
from scripts.logger import logging_context

with logging_context(__file__):
    print("消息会同时显示在控制台和保存到日志文件")
```

**Debug模式**:
```python
tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)
attention = MultiHeadAttention(d_model=128, nhead=8, debug_mode=True)
model = TransformerModel(vocab_size=1000, d_model=128, debug_mode=True)
pipeline = CodeGenerationPipeline(vocab_size=1000, d_model=128, debug_mode=True)
```

**Verbose参数**:
```python
result = pipeline.generate(prompt="public class UserService", max_length=50, verbose=True)
```

### 2. Tokenizer调试工具

```bash
# 完整分析
python scripts/tests/debug_tokenizer.py

# 交互式调试
python scripts/tests/tokenizer_interactive.py
```

**功能**:
- ✅ Step-by-step分析
- ✅ Token可视化（已知/未知）
- ✅ 词汇表统计
- ✅ 多文本对比
- ✅ UNK检测

### 3. 观察中间结果

```python
# Tokenization
ids, mask = tokenizer.encode("public class User")
print(f"Token IDs: {ids}")
print(f"Mask: {mask}")

# Attention weights
output, weights = attention(query, key, value)
print(f"Weights shape: {weights.shape}")

# Probability distribution
logits = model(src, tgt)
probs = torch.softmax(logits[:, -1, :], dim=-1)
top5 = torch.topk(probs, 5)
print(f"Top-5 predictions: {top5.values}")
```

### 4. 性能分析

```python
import time

start = time.perf_counter()
result = pipeline.generate(prompt)
elapsed = time.perf_counter() - start

print(f"Generation time: {elapsed:.4f}s")
print(f"Tokens/sec: {result['token_count']/elapsed:.2f}")
```

### 5. 可视化工具

```python
from scripts.visualizer import AttentionVisualizer

visualizer = AttentionVisualizer()

# 注意力权重
visualizer.visualize_attention(weights, token_names)

# 模型架构
visualizer.visualize_model_architecture(num_encoder_layers=2, num_decoder_layers=2)

# 位置编码
visualizer.visualize_positional_encoding(pos_encoder.pe)
```

### 6. 测试套件

```bash
python scripts/tests/test_all.py              # 所有测试
python scripts/tests/test_performance.py      # 性能测试
python scripts/tests/test_new_features.py     # 新功能测试
python scripts/tests/verify_math.py           # 数学验证
python scripts/kv_cache.py                    # KV Cache演示
```

---

## 常见问题

### Q1: 为什么生成的代码全是`<UNK>`？

**原因**: 词汇表太小或训练数据不足

**解决**:
```python
# 增大词汇表
tokenizer = SimpleTokenizer(vocab_size=2000)
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

### Q5: 为什么Decoder需要Causal Mask？

**原因**: 防止看到未来token

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
3. ✅ 阅读本README的"核心概念"部分
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

## 🎓 新手学习指南

如果你是第一次接触这个项目，我们为你准备了详细的学习计划：

📖 **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - 完整的7-14天学习计划

该指南包含：
- ✅ 学习前准备（环境搭建、心理准备）
- ✅ 第1-2天：建立整体认知
- ✅ 第3-4天：深入核心模块
- ✅ 第5-6天：掌握高级功能
- ✅ 第7-8天：实践与扩展
- ✅ 学习资源推荐
- ✅ 常见问题解答
- ✅ 学习检查清单

**适合人群**:
- 🎯 对Transformer和大模型感兴趣的开发者
- 🎯 想要深入理解LLM工作原理的学习者
- 🎯 希望从零实现一个简化版LLM的实践者

---

## 📂 文档说明

本项目采用极简文档结构，仅保留3个核心文档：

- **[README.md](README.md)** - 主文档，包含项目介绍、核心概念、模块说明、调试工具等
- **[QUICK_START.md](QUICK_START.md)** - 快速开始指南，常用命令速查
- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - 完整学习指南，7-14天学习计划

**备份文档**: `docs/backup_old_docs/` 目录中保留了之前版本的完整文档，如需查阅可参考。

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

## 项目进度

想了解项目的开发进度和未来计划？请查看：

📋 **[PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)**

**当前版本**: v1.2 (2026-05-07)  
**最新功能**: 训练系统 + KV Cache优化

---

## 许可证

本项目仅用于教育和学习目的。

---

**祝您深入学习愉快！** 

*最后更新: 2026-05-10*
