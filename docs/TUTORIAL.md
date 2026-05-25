# 📖 详细教程 - Transformer代码生成系统

> **阅读时间**: 20-30分钟  
> **目标**: 从快速入门到深入理解Transformer架构和代码生成原理

---

## 📋 目录

1. [5分钟快速开始](#1-5分钟快速开始)
2. [核心概念详解](#2-核心概念详解)
3. [模块学习路径](#3-模块学习路径)
4. [实战练习](#4-实战练习)
5. [常见问题](#5-常见问题)

---

## 1. 5分钟快速开始

### 1.1 安装依赖

```bash
pip install torch numpy matplotlib seaborn
```

### 1.2 运行第一个示例

**最简用法**（只需2行代码）:

```python
from scripts.pipeline import CodeGenerationPipeline

# 创建Pipeline并生成代码
pipeline = CodeGenerationPipeline(preset='small')
result = pipeline.generate("public class UserService")
print(result['processed_code'])
```

### 1.3 尝试不同的Preset配置

| Preset | 适用场景 | 参数量 | 速度 |
|--------|---------|--------|------|
| **tiny** | 快速测试、教学演示 | ~50K | ⚡⚡⚡ 最快 |
| **small** | 常规实验、课堂演示 | ~200K | ⚡⚡ 平衡（默认） |
| **medium** | 深入研究、性能测试 | ~800K | ⚡ 更强 |

```python
# Tiny preset - 最快，适合快速测试
pipeline_tiny = CodeGenerationPipeline(preset='tiny')

# Small preset - 平衡性能和速度（默认推荐）
pipeline_small = CodeGenerationPipeline(preset='small')

# Medium preset - 更强表达能力
pipeline_medium = CodeGenerationPipeline(preset='medium')
```

### 1.4 完整示例

```python
from scripts.pipeline import CodeGenerationPipeline

# 创建Pipeline
pipeline = CodeGenerationPipeline(preset='small')

# 生成代码
result = pipeline.generate(
    prompt="public class UserService",
    max_length=50,
    temperature=0.7
)

# 查看结果
if result['success']:
    print("生成的代码:")
    print(result['processed_code'])
    print(f"\n耗时: {result['processing_time']:.2f}s")
    print(f"Token数: {result['token_count']}")
```

**恭喜！** 你已经成功运行了第一个代码生成示例。接下来让我们深入理解背后的原理。

---

## 2. 核心概念详解

### 2.1 Tokenization（文本分词）

#### 什么是Token？

Token是LLM处理的最小单元。例如：

```
原始文本: "public class UserService"
         ↓ 分词
Tokens: ["public", "class", "User", "Service"]
         ↓ 转换为ID
IDs:    [15, 23, 156, 289]
```

#### 为什么需要Tokenization？

计算机只能处理数字，不能直接处理文本。Tokenization将文本转换为数字序列。

#### Attention Mask的作用

当序列长度不一致时，需要padding到相同长度：

```
Token IDs: [15, 23, 156, 0, 0, 0]
Mask:      [1,  1,  1,   0, 0, 0]
           ↑真实token     ↑padding
```

Mask告诉模型哪些位置是真实的token，哪些是填充。

#### 代码示例

```python
from scripts.core.tokenizer import SimpleTokenizer

# 创建tokenizer
tokenizer = SimpleTokenizer(vocab_size=1000)

# 编码
text = "public class User"
token_ids, mask = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Token IDs: {token_ids}")
print(f"Mask: {mask}")

# 解码
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")
```

---

### 2.2 Attention Mechanism（注意力机制）

#### 通俗理解

想象你在阅读时，眼睛会**自动关注重要的词**。注意力机制让AI学会这种能力。

**生活例子**：
```
句子: "小明昨天在图书馆借了一本关于人工智能的书"

当问到"谁借了书？"时，你会重点关注：
- "小明" ← 最关注（主语）
- "借" ← 次关注（动作）
- 其他词相对不重要
```

#### 数学公式

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**分解步骤**：
1. **Q @ K^T**: 计算每个token之间的相似度
2. **/ √d_k**: 缩放，防止数值过大
3. **softmax**: 转换为概率分布（和为1）
4. **@ V**: 加权求和，得到最终输出

#### 三种注意力类型

**1. Self-Attention（自注意力）**
- 同一个序列内部的token互相观察
- 用于Encoder理解序列内部关系

**2. Cross-Attention（交叉注意力）**
- 一个序列关注另一个序列
- 用于Decoder关注Encoder的输出

**3. Multi-Head Attention（多头注意力）**
- 用多个"视角"同时观察
- 从不同角度理解，更全面

#### 代码示例

```python
from scripts.core.attention import MultiHeadAttention
import torch

# 创建注意力层
attention = MultiHeadAttention(d_model=128, nhead=8)

# 准备输入 (batch_size=1, seq_len=10, d_model=128)
query = torch.randn(1, 10, 128)
key = torch.randn(1, 10, 128)
value = torch.randn(1, 10, 128)

# 前向传播
output, weights = attention(query, key, value)

print(f"Output shape: {output.shape}")      # (1, 10, 128)
print(f"Attention weights shape: {weights.shape}")  # (1, 8, 10, 10)
```

---

### 2.3 Transformer Architecture

#### 整体架构

```
Input Tokens → Embedding → Positional Encoding 
             → Encoder×N → Decoder×N 
             → Linear + Softmax → Output Probabilities
```

#### Encoder Layer结构

每个Encoder层包含：
1. **Self-Attention**: token之间互相观察
2. **Residual Connection**: 原始输入 + 注意力输出
3. **Layer Normalization**: 归一化，稳定训练
4. **Feed Forward Network**: 非线性特征变换

#### Decoder Layer结构

每个Decoder层包含：
1. **Masked Self-Attention**: 防止看到未来token
2. **Cross-Attention**: 关注Encoder的输出
3. **Feed Forward Network**: 非线性特征变换

#### 关键设计

**Positional Encoding**:
- Transformer没有循环结构，无法感知位置
- 通过添加位置编码，让模型知道token的顺序

**Residual Connection**:
- 解决深层网络的梯度消失问题
- 公式: `output = input + sublayer(input)`

**Layer Normalization**:
- 归一化每层的输出
- 加速训练收敛

#### 代码示例

```python
from scripts.core.transformer import TransformerModel
import torch

# 创建模型
model = TransformerModel(
    vocab_size=1000,
    d_model=128,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2
)

# 准备输入
src = torch.randint(0, 1000, (1, 10))  # 源序列
tgt = torch.randint(0, 1000, (1, 5))   # 目标序列

# 前向传播
output = model(src, tgt)

print(f"Output shape: {output.shape}")  # (1, 5, 1000)
```

---

### 2.4 Code Generation（代码生成）

#### Auto-regressive生成

代码生成是**逐步生成token**的过程：

```
Prompt: "public class User"
Step 1: 生成 "{"        → "public class User{"
Step 2: 生成 "private"  → "public class User{private"
Step 3: 生成 "String"   → "public class User{private String"
...
```

#### Temperature采样

Temperature控制生成的随机性：

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

#### 四种采样策略

| 策略 | 确定性 | 多样性 | 适用场景 |
|------|--------|--------|----------|
| Greedy | 最高 | 最低 | 代码生成、翻译 |
| Temperature | 可调 | 可调 | 通用场景 |
| Top-K | 中高 | 中低 | 控制质量 |
| Top-P | 自适应 | 自适应 | 灵活场景 |

#### 代码示例

```python
from scripts.pipeline import CodeGenerationPipeline

# 创建pipeline
pipeline = CodeGenerationPipeline(preset='small')

# 生成代码
result = pipeline.generate(
    prompt="public class UserService",
    max_length=50,
    temperature=0.7
)

if result['success']:
    print("Generated code:")
    print(result['processed_code'])
```

---

### 2.5 Post-processing（后处理）

#### 为什么需要后处理？

模型生成的代码可能存在：
- ❌ 括号不匹配
- ❌ 语法错误
- ❌ 格式混乱

后处理器修复这些问题。

#### 核心功能

- ✅ **括号匹配检查**: 检测未闭合的括号
- ✅ **语法验证**: 检查常见语法错误
- ✅ **代码格式化**: 自动缩进和换行
- ✅ **警告系统**: 提示潜在问题

#### 代码示例

```python
from scripts.generation.code_generator import SimplePostProcessor

postprocessor = SimplePostProcessor()

# 处理生成的代码
code = "public class User{private String name;}"
result = postprocessor.process(code)

print(f"Formatted:\n{result}")
```

---

### 2.6 Caching（缓存机制）

#### 两种缓存类型

**1. KV Cache（模型内部优化）**
- 避免重复计算历史token的K/V
- 加速单次生成的每一步
- 对长序列特别有效（加速10-50倍）

**2. Result Cache（应用层优化）**
- 缓存完整的生成结果
- 避免重复生成相同内容
- 适用于API服务中的重复请求

#### KV Cache性能对比

| 序列长度 | 无Cache计算量 | 有Cache计算量 | 加速比 |
|---------|--------------|--------------|--------|
| 10      | 55           | 10           | 5.5x   |
| 100     | 5,050        | 100          | 50.5x  |
| 1000    | 500,500      | 1,000        | 500.5x |

#### 代码示例

```python
# Result Cache（默认启用）
pipeline = CodeGenerationPipeline(preset='small', cache_size=20)

# 第一次请求（生成并缓存）
result1 = pipeline.generate(prompt="public class User")
print(f"Source: {result1['source']}")  # 'generated'

# 第二次相同请求（从缓存获取）
result2 = pipeline.generate(prompt="public class User")
print(f"Source: {result2['source']}")  # 'cache'
```

---

## 3. 模块学习路径

### 3.1 推荐学习顺序

按照以下顺序深入学习各个模块：

```
1. Tokenizer（最简单，建立信心）
   ↓
2. Attention（核心机制，重点学习）
   ↓
3. Transformer（整合Encoder-Decoder）
   ↓
4. Generator（理解采样策略）
   ↓
5. PostProcessor（实用技巧）
   ↓
6. Cache/KV Cache（性能优化）
   ↓
7. Pipeline（整体流程）
```

### 3.2 每个模块的学习方法

**Step 1: 阅读源码**
- 打开对应的`.py`文件
- 仔细阅读类和方法的docstring
- 理解每个函数的作用

**Step 2: 运行测试**
- 每个模块都可以独立运行
- 查看输出，理解工作流程

**Step 3: 修改参数**
- 改变配置参数，观察效果
- 尝试不同的输入，理解边界情况

**Step 4: 手写实现**
- 尝试从头实现简化版本
- 加深理解

---

## 4. 实战练习

### 练习1: Tokenizer实验

**目标**: 理解Tokenization过程

```python
from scripts.core.tokenizer import SimpleTokenizer

# 任务1: 创建不同大小的词汇表
tokenizer_small = SimpleTokenizer(vocab_size=100)
tokenizer_large = SimpleTokenizer(vocab_size=2000)

# 任务2: 编码同一段代码
code = "public class UserService { private String name; }"

ids_small, _ = tokenizer_small.encode(code)
ids_large, _ = tokenizer_large.encode(code)

# 任务3: 统计UNK比例
unk_count_small = sum(1 for x in ids_small if x == tokenizer_small.unk_id)
unk_count_large = sum(1 for x in ids_large if x == tokenizer_large.unk_id)

print(f"Small vocab UNK count: {unk_count_small}")
print(f"Large vocab UNK count: {unk_count_large}")
```

**思考题**:
- 为什么词汇表大小会影响UNK数量？
- 如何平衡词汇表大小和内存占用？

---

### 练习2: Temperature对比

**目标**: 理解Temperature对生成的影响

```python
from scripts.pipeline import CodeGenerationPipeline

pipeline = CodeGenerationPipeline(preset='tiny')

prompt = "def fibonacci(n):"

# 使用不同temperature生成
temperatures = [0.3, 0.7, 1.0, 1.5]

for temp in temperatures:
    result = pipeline.generate(
        prompt=prompt,
        max_length=30,
        temperature=temp
    )
    print(f"\nTemperature={temp}:")
    print(result['processed_code'])
```

**思考题**:
- 哪个temperature生成的代码质量最高？
- 为什么过高的temperature会导致代码混乱？

---

### 练习3: 自定义Preset

**目标**: 掌握配置系统设计

```python
from scripts.config.presets import PRESETS
from scripts.pipeline import CodeGenerationPipeline

# 查看现有preset
print("Available presets:", list(PRESETS.keys()))

# 创建自己的preset
PRESETS['custom'] = {
    'name': 'Custom (我的配置)',
    'description': '用于特定场景的配置',
    'vocab_size': 1500,
    'd_model': 192,
    'nhead': 8,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'dim_feedforward': 512,
    'max_seq_length': 128,
    'cache_size': 30,
    'use_case': '平衡性能和资源'
}

# 使用自定义preset
pipeline = CodeGenerationPipeline(preset='custom')
```

**思考题**:
- 如何根据硬件条件选择合适的preset？
- d_model和nhead之间有什么关系？

---

## 5. 常见问题

### Q1: 为什么生成的代码全是`<UNK>`？

**原因**: 词汇表太小或训练数据不足

**解决方案**:
```python
# 增大词汇表
pipeline = CodeGenerationPipeline(preset='medium')  # vocab_size=2000

# 或者手动指定
pipeline = CodeGenerationPipeline(
    preset='small',
    vocab_size=3000  # 覆盖preset
)
```

---

### Q2: 如何提高生成质量？

**短期方案**（调整参数）:
- 增大d_model (128 → 256)
- 增加层数 (2 → 4)
- 增大词汇表 (1000 → 2000)
- 降低temperature (0.7 → 0.5)

**长期方案**（需要训练）:
- 在大量代码数据上预训练
- 使用BPE分词代替简单分词
- 增加模型规模到数百万参数

---

### Q3: 可以用GPU吗？

**可以**:
```python
pipeline = CodeGenerationPipeline(preset='small', device='cuda')
```

**前提**:
- 安装CUDA版本的PyTorch
- 有NVIDIA GPU

**加速效果**: 5-10x（取决于GPU型号）

---

### Q4: KV Cache和Result Cache有什么区别？

| 特性 | KV Cache | Result Cache |
|------|----------|--------------|
| 层级 | 模型内部 | 应用层 |
| 作用 | 加速单步计算 | 避免重复生成 |
| 存储 | K/V张量 | 完整结果 |
| 适用 | 所有生成都受益 | 仅重复请求受益 |

---

### Q5: 如何选择sampling strategy？

**建议**:
- **代码生成**: Greedy或Temperature=0.5-0.7
- **创意写作**: Temperature=0.8-1.2
- **事实问答**: Temperature=0.3-0.5
- **多样本对比**: 使用多种temperature

---

## 🎓 下一步

完成本教程后，你可以：

1. **阅读参考手册**: [REFERENCE.md](REFERENCE.md) - 高级配置、API文档、扩展开发
2. **动手实践**: 尝试第4节的所有练习
3. **深入学习数学基础**: [learning-resources/math/](learning-resources/math/) - 线性代数、微积分、概率论
4. **贡献代码**: 添加新功能或改进现有实现

---

**祝您学习愉快！**

*最后更新: 2026-05-25 | 版本: v2.0-simplified*
