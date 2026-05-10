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
8. [🎓 学习建议](#-学习建议)

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

**交互式主菜单提供7个演示选项**:
- **选项1**: 基本代码生成流程（观看完整过程，含debug日志）
- **选项2**: 不同采样策略对比（理解temperature作用）
- **选项3**: 缓存机制演示（看性能提升）
- **选项4**: 多样本生成（同时生成多个版本）
- **选项5**: 可视化功能（生成注意力热力图）
- **选项6**: 交互模式（自己输入prompt测试）
- **选项7**: 运行所有演示

**统一入口脚本**:
```bash
python run_demo.py  # 提供简化的菜单界面
```

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

**多样本生成示例**:
```python
# 同时生成多个不同版本
samples = pipeline.generate_multiple_samples(
    prompt="public class ProductService",
    num_samples=3,
    max_length=35,
    temperatures=[0.5, 0.7, 1.0]  # 不同的temperature
)

for sample in samples:
    print(f"Sample {sample['sample_id']} (temp={sample['temperature']}):")
    print(sample['code'])
    print()
```

### 5. 常用命令速查

#### 测试命令
```bash
python scripts/tests/test_all.py              # 运行所有核心测试（7项）
python scripts/tests/test_performance.py      # 性能基准测试
python scripts/tests/test_new_features.py     # 新功能测试（训练+KV Cache）
python scripts/tests/verify_math.py           # 数学公式验证
python scripts/test_training_and_cache.py     # 训练和KV Cache集成演示
```

#### 演示命令
```bash
python scripts/main.py                        # 交互式主菜单（7个演示）
python run_demo.py                            # 统一入口脚本（简化菜单）
```

#### 调试命令
```bash
python scripts/tests/debug_tokenizer.py       # Tokenizer完整分析
python scripts/tests/tokenizer_interactive.py # 交互式Tokenizer调试
python scripts/tests/tokenizer_pdb_debug.py   # PDB断点调试示例
python scripts/kv_cache.py                    # KV Cache独立演示
```

#### 可视化命令
```python
# 在Python代码中使用
from scripts.visualizer import AttentionVisualizer

visualizer = AttentionVisualizer()
visualizer.visualize_attention(weights, tokens)        # 注意力热力图
visualizer.visualize_model_architecture()              # 模型架构图
visualizer.visualize_positional_encoding(pe)           # 位置编码可视化
```

### 6. 硬件要求

**最低配置**:
- CPU: 任意现代CPU
- 内存: 4GB RAM
- 存储: 100MB可用空间

**推荐配置**:
- GPU: NVIDIA GPU with CUDA support (可选，加速5-10倍)
- 内存: 8GB+ RAM
- Python: 3.8+

**性能参考** (Intel i7, 无GPU):
- 短序列生成 (<50 tokens): ~1-2秒
- 中等序列生成 (50-100 tokens): ~3-5秒
- 启用KV Cache后: 加速10-50倍

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

#### 多样本生成

可以同时使用不同的temperature生成多个版本，对比效果：

```python
samples = pipeline.generate_multiple_samples(
    prompt="public class OrderService",
    num_samples=3,
    temperatures=[0.5, 0.7, 1.0]
)
```

这有助于理解temperature对生成结果的影响。

---

### 后处理器 ⭐重要

#### 概述

后处理器对生成的代码进行格式化、验证和优化，提高代码质量。

**核心功能**:
- ✅ **括号匹配检查**: 检测未闭合的括号
- ✅ **语法验证**: 检查常见语法错误
- ✅ **代码格式化**: 自动缩进和换行
- ✅ **警告系统**: 提示潜在问题

#### 使用示例

```python
from scripts.postprocessor import CodePostProcessor

postprocessor = CodePostProcessor()

# 处理生成的代码
result = postprocessor.process(generated_code)

print(f"Valid: {result['is_valid']}")
print(f"Errors: {result['errors']}")
print(f"Warnings: {result['warnings']}")
print(f"Formatted code:\n{result['formatted_code']}")
```

#### 在Pipeline中使用

```python
# 默认启用了后处理
result = pipeline.generate(
    prompt="public class UserService",
    do_post_process=True  # 启用后处理
)

# 查看后处理结果
if result['post_processed']:
    print("Original:", result['raw_code'])
    print("Processed:", result['processed_code'])
```

---

### 训练系统 ⭐新增

#### 概述

训练系统使模型能够从数据中学习，而不仅仅是随机初始化后的前向传播。

**核心组件**：
- ✅ CrossEntropyLoss - 交叉熵损失函数（支持标签平滑）
- ✅ AdamW - 优化器（解耦权重衰减）
- ✅ WarmupLinearScheduler - 学习率调度器
- ✅ TextDataset - 数据集类
- ✅ Trainer - 训练管理器

#### 快速开始

```python
from scripts.trainer import Trainer, TextDataset
from scripts.transformer import TransformerModel
from scripts.tokenizer import SimpleTokenizer

# 1. 准备数据
texts = [
    "def hello_world():",
    "    print('Hello, World!')",
    "class UserService:",
    "    def __init__(self):"
]
tokenizer = SimpleTokenizer(vocab_size=1000)
dataset = TextDataset(texts, tokenizer, max_len=128)

# 2. 创建模型
model = TransformerModel(
    vocab_size=1000,
    d_model=256,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2
)

# 3. 训练
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    batch_size=16,
    lr=1e-4,
    epochs=20,
    warmup_steps=100,
    label_smoothing=0.1
)

history = trainer.train()
```

#### 核心特性

1. **标签平滑（Label Smoothing）**:
   ```python
   # 防止模型过于自信，提高泛化能力
   loss_fn = CrossEntropyLoss(label_smoothing=0.1)
   ```

2. **Warmup学习率策略**:
   ```python
   # 初期线性增加，后期线性衰减
   scheduler = WarmupLinearScheduler(
       optimizer=optimizer,
       warmup_steps=100,
       total_steps=1000
   )
   ```

3. **梯度裁剪**:
   ```python
   # 防止梯度爆炸
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. **Checkpoint管理**:
   ```python
   # 自动保存最佳模型
   trainer.save_checkpoint('best_model.pth')
   
   # 加载模型
   trainer.load_checkpoint('best_model.pth')
   ```

#### 训练监控

```python
# 查看训练历史
print(f"Final Loss: {history['loss'][-1]:.4f}")
print(f"Best Loss: {min(history['loss']):.4f}")

# 可视化训练曲线（需要matplotlib）
import matplotlib.pyplot as plt
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
```

#### 集成演示

```bash
python scripts/test_training_and_cache.py  # 完整的训练和KV Cache演示
```

---

### KV Cache ⭐新增

#### 概述

KV Cache（Key-Value缓存）是LLM推理加速的核心技术，可将生成长度从O(n²)降低到O(n)。

**核心价值**：
- ⚡ 推理速度提升10-50倍
- 💾 避免重复计算历史token的K/V
- 🎯 对长序列特别有效

**注意**: KV Cache不同于结果缓存（cache.py），它是模型内部的优化机制。

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

#### 独立演示

```bash
python scripts/kv_cache.py  # 查看详细的性能对比
```

---

### 结果缓存（Result Cache）

#### 概述

结果缓存（cache.py）用于缓存完整的生成结果，当相同的prompt再次出现时直接返回缓存结果。

**与KV Cache的区别**:
- **KV Cache**: 模型内部优化，加速单次生成的每一步
- **结果缓存**: 应用层优化，避免重复生成相同内容

**适用场景**:
- API服务中重复的请求
- 测试和调试时的快速响应
- 减少不必要的计算

#### 使用示例

```python
from scripts.pipeline import CodeGenerationPipeline

pipeline = CodeGenerationPipeline(
    vocab_size=1000,
    d_model=128,
    cache_size=20  # 缓存大小
)

# 第一次请求（生成并缓存）
result1 = pipeline.generate(prompt="public class User", max_length=30)
print(f"Source: {result1['source']}")  # 'generated'

# 第二次相同请求（从缓存获取）
result2 = pipeline.generate(prompt="public class User", max_length=30)
print(f"Source: {result2['source']}")  # 'cache'
print(f"Speedup: {result1['processing_time']/result2['processing_time']:.1f}x")
```

#### 缓存统计

```python
pipeline.cache.display_stats()
# 输出:
# Cache Statistics:
#   Total requests: 10
#   Cache hits: 3
#   Cache misses: 7
#   Hit rate: 30.0%
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
| [tokenizer.py](file:///home/wsm/codes/llm-gencode-demo/scripts/tokenizer.py) | 471 | 文本分词器（支持UNK检测、mask生成） |
| [attention.py](file:///home/wsm/codes/llm-gencode-demo/scripts/attention.py) | 283 | 多头注意力机制（Self/Cross/Multi-Head） |
| [transformer.py](file:///home/wsm/codes/llm-gencode-demo/scripts/transformer.py) | 970 | Transformer完整架构（Encoder+Decoder） |
| [generator.py](file:///home/wsm/codes/llm-gencode-demo/scripts/generator.py) | 381 | 代码生成器（4种采样策略） |
| [postprocessor.py](file:///home/wsm/codes/llm-gencode-demo/scripts/postprocessor.py) | 416 | 代码后处理（格式化、语法验证、优化） |
| [cache.py](file:///home/wsm/codes/llm-gencode-demo/scripts/cache.py) | 301 | 结果缓存机制（加速重复查询） |
| [pipeline.py](file:///home/wsm/codes/llm-gencode-demo/scripts/pipeline.py) | 442 | 完整流程整合（端到端管道） |
| [visualizer.py](file:///home/wsm/codes/llm-gencode-demo/scripts/visualizer.py) | 376 | 可视化工具（注意力热力图、架构图） |
| [trainer.py](file:///home/wsm/codes/llm-gencode-demo/scripts/trainer.py) | ~600 | 训练系统（Loss/Optimizer/Scheduler） |
| [kv_cache.py](file:///home/wsm/codes/llm-gencode-demo/scripts/kv_cache.py) | ~400 | KV Cache优化（推理加速10-50倍） |

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

# PDB断点调试示例
python scripts/tests/tokenizer_pdb_debug.py
```

**功能**:
- ✅ Step-by-step分析
- ✅ Token可视化（已知/未知）
- ✅ 词汇表统计
- ✅ 多文本对比
- ✅ UNK检测
- ✅ 交互式测试

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

**性能测试命令**:
```bash
python scripts/tests/test_performance.py      # 完整性能基准测试
python scripts/tests/test_new_features.py     # 新功能性能测试
```

### 5. 可视化工具

```python
from scripts.visualizer import AttentionVisualizer

visualizer = AttentionVisualizer()

# 注意力权重热力图
visualizer.visualize_attention(weights, token_names, head_idx=0, layer_idx=0)

# 模型架构图
visualizer.visualize_model_architecture(
    num_encoder_layers=2,
    num_decoder_layers=2,
    save_path='model_architecture.png'
)

# 位置编码可视化
visualizer.visualize_positional_encoding(pos_encoder.pe)
```

**依赖安装**:
```bash
pip install matplotlib seaborn
```

### 6. 测试套件

```bash
# 核心测试
python scripts/tests/test_all.py              # 所有核心测试（7项）

# 专项测试
python scripts/tests/test_performance.py      # 性能测试
python scripts/tests/test_new_features.py     # 新功能测试（训练+KV Cache）
python scripts/tests/verify_math.py           # 数学公式验证

# 集成演示
python scripts/test_training_and_cache.py     # 训练和KV Cache集成演示
python scripts/kv_cache.py                    # KV Cache独立演示
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

## 🎓 学习建议

### 新手入门（推荐顺序）

1. **第一步**: 阅读下方的"快速开始"章节，运行第一个演示（5分钟）
2. **第二步**: 阅读"核心概念"章节，理解基本原理（30分钟）
3. **第三步**: 跟随"学习路径"章节，系统性学习（7天）
4. **第四步**: 使用"调试工具"章节，深入探索（按需查阅）

### 快速查阅

- **想了解项目结构**: 查看"模块说明"章节
- **想调试代码**: 查看"调试工具"章节
- **遇到问题**: 查看"常见问题"章节
- **想深入学习**: 跟随"学习路径"章节

---

## 📂 文档说明

本项目采用极简文档结构，仅保留1个核心文档：

- **[README.md](README.md)** - 唯一文档，包含所有必要信息
  - 项目介绍和快速开始
  - 核心概念详解
  - 模块说明和项目结构
  - 调试工具和测试套件
  - 常见问题解答
  - 学习路径和建议

**备份文档**: `docs/backup_old_docs/` 目录中保留了之前版本的完整文档，如需查阅历史资料可参考。

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

**当前版本**: v1.2 (2026-05-10)  
**最新功能**: 训练系统 + KV Cache优化

**已完成**:
- ✅ 完整的Transformer架构实现
- ✅ 多种采样策略（Greedy, Temperature, Top-K, Top-P）
- ✅ 训练系统（CrossEntropyLoss, AdamW, Scheduler）
- ✅ KV Cache优化（推理加速50倍+）
- ✅ 丰富的调试和可视化工具
- ✅ 完整的文档和学习指南

**未来计划**:
- 🔮 在真实数据集上训练模型
- 🔮 实现BPE Tokenizer
- 🔮 添加Beam Search等高级采样策略
- 🔮 性能优化（FP16、梯度累积）

---

## 许可证

本项目仅用于教育和学习目的。

---

**祝您深入学习愉快！** 

*最后更新: 2026-05-10*
