# 🎓 大模型底层原理 - 从零开始的完整教程

> **阅读时间**: 60-90分钟  
> **适合人群**: 完全零基础的学习者  
> **学习目标**: 理解Transformer架构的每个细节，能够独立解释大模型的工作原理

---

## 📖 目录

1. [前言：为什么需要理解大模型？](#1-前言为什么需要理解大模型)
2. [第一章：从文本到数字 - Tokenization](#2-第一章从文本到数字---tokenization)
3. [第二章：让机器"看见"文字 - Embedding](#3-第二章让机器看见文字---embedding)
4. [第三章：告诉模型"顺序" - Positional Encoding](#4-第三章告诉模型顺序---positional-encoding)
5. [第四章：注意力机制 - Attention（核心！）](#5-第四章注意力机制---attention核心)
6. [第五章：多头注意力 - Multi-Head Attention](#6-第五章多头注意力---multi-head-attention)
7. [第六章：Encoder - 理解输入](#7-第六章encoder---理解输入)
8. [第七章：Decoder - 生成输出](#8-第七章decoder---生成输出)
9. [第八章：完整的Transformer架构](#9-第八章完整的transformer架构)
10. [第九章：自回归生成 - Auto-regressive Generation](#10-第九章自回归生成---auto-regressive-generation)
11. [第十章：采样策略 - Temperature & Top-K](#11-第十章采样策略---temperature--top-k)
12. [第十一章：性能优化 - KV Cache](#12-第十一章性能优化---kv-cache)
13. [第十二章：后处理 - Post-processing](#13-第十二章后处理---post-processing)
14. [第十三章：完整流程总结](#14-第十三章完整流程总结)
15. [附录：实战练习与常见问题](#15-附录实战练习与常见问题)

---

## 1. 前言：为什么需要理解大模型？

### 1.1 什么是大语言模型（LLM）？

**通俗定义**：大语言模型就是一个"超级聪明的文字接龙游戏玩家"。

你给它一段话的开头，它能根据上下文预测下一个最可能出现的词，然后不断重复这个过程，最终生成完整的文章、代码、对话等。

**生活中的类比**：

```
想象你在玩成语接龙：
你："一马当先"
朋友："先..." → 思考 → "先发制人"
你："人..." → 思考 → "人山人海"

大模型做的就是这件事，但它：
✅ 词汇量巨大（数万个词）
✅ 记忆力超强（能记住几千个词的上下文）
✅ 反应极快（毫秒级预测）
✅ 知识渊博（读过互联网上几乎所有文本）
```

### 1.2 为什么要学习底层原理？

| 学习者类型 | 学习收益 |
|-----------|---------|
| **学生/研究者** | 深入理解AI技术，为科研打基础 |
| **开发者** | 能够调优模型参数，解决实际工程问题 |
| **产品经理** | 了解模型能力边界，设计更好的AI产品 |
| **普通用户** | 不被营销忽悠，理性看待AI技术 |

### 1.3 本教程的特点

✅ **零基础友好**：不需要数学背景，所有公式都有直观解释  
✅ **代码驱动**：结合真实项目代码，理论与实践结合  
✅ **图文并茂**：每个概念都配有图表和示例  
✅ **循序渐进**：从最简单的Tokenization到复杂的Attention机制  

---

## 2. 第一章：从文本到数字 - Tokenization

### 2.1 核心问题：计算机只能处理数字

**问题**：计算机不理解"public class User"这样的文字，它只能处理数字。

**解决方案**：将文本转换为数字序列，这个过程叫**Tokenization（分词）**。

### 2.2 什么是Token？

**Token = 文本的最小处理单元**

可以是一个词、一个子词、甚至一个字符。

**示例对比**：

```
原始文本: "public class UserService"

方式1: 按单词分割（Word-level）
Tokens: ["public", "class", "UserService"]
IDs:    [15,     23,    156]

方式2: 按子词分割（Subword-level，更常用）
Tokens: ["public", "class", "User", "Service"]
IDs:    [15,     23,    89,   234]

方式3: 按字符分割（Character-level）
Tokens: ["p", "u", "b", "l", "i", "c", " ", "c", ...]
IDs:    [112, 117, 98, 108, 105, 99, 32, 99, ...]
```

**为什么选择Subword？**
- ✅ 平衡了词汇表大小和表达能力
- ✅ 能处理未知词（如"ChatGPT"可以拆成"Chat" + "GPT"）
- ✅ 真实LLM（GPT、BERT、Llama）都使用这种方式

### 2.3 特殊Token：BOS、EOS、PAD、UNK

在实际应用中，我们需要一些特殊标记来处理边界情况：

| Token | 全称 | 作用 | ID | 示例 |
|-------|------|------|----|----|
| **BOS** | Begin Of Sequence | 标记序列开始 | 2 | `[BOS] public class` |
| **EOS** | End Of Sequence | 标记序列结束 | 1 | `public class [EOS]` |
| **PAD** | Padding | 填充到固定长度 | 0 | `[PAD][PAD][PAD]` |
| **UNK** | Unknown | 处理词汇表外的词 | 3 | "ChatGPT" → `[UNK]` |

**为什么需要这些特殊Token？**

```python
# 示例1：BOS和EOS的作用
text = "public class User"
tokens = [BOS, "public", "class", "User", EOS]
#      = [2,   15,      23,     89,   1]

# BOS告诉模型："这是序列的起点，从这里开始理解"
# EOS告诉模型："序列到此结束，后面没有了"

# 示例2：PAD的作用（批量处理时）
batch = [
    "public class User",           # 5个tokens（含BOS/EOS）
    "return null;",                # 4个tokens
    "if (x > 0) { return x; }"    # 10个tokens
]

# 为了批量处理，需要统一长度（假设max_length=12）
batch_padded = [
    [2, 15, 23, 89, 1, 0, 0, 0, 0, 0, 0, 0],  # 5 + 7个PAD
    [2, 45, 67, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # 4 + 8个PAD
    [2, 12, 34, 56, 78, 90, 1, 0, 0, 0, 0, 0] # 10 + 2个PAD
]
# PAD让所有序列长度一致，便于GPU并行计算
```

### 2.4 代码实现详解

让我们看看项目中[tokenizer.py](file:///E:/Project/llm-codegen-demo/scripts/core/tokenizer.py)的实现：

```python
class SimpleTokenizer:
    def __init__(self, vocab_size: int = 1000):
        # 特殊token
        self.PAD_TOKEN = '<PAD>'   # ID = 0
        self.EOS_TOKEN = '<EOS>'   # ID = 1
        self.BOS_TOKEN = '<BOS>'   # ID = 2
        self.UNK_TOKEN = '<UNK>'   # ID = 3
        
        # 构建词汇表
        self.vocab = self._build_vocabulary()
        # 反向映射：ID → token
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def _build_vocabulary(self):
        """构建词汇表"""
        vocab = {}
        idx = 0
        
        # 1. 添加特殊token
        vocab['<PAD>'] = idx; idx += 1  # 0
        vocab['<EOS>'] = idx; idx += 1  # 1
        vocab['<BOS>'] = idx; idx += 1  # 2
        vocab['<UNK>'] = idx; idx += 1  # 3
        
        # 2. 添加Java关键字
        java_keywords = [
            'public', 'private', 'class', 'return', 
            'if', 'else', 'for', 'while', ...
        ]
        for kw in java_keywords:
            vocab[kw] = idx; idx += 1
        
        # 3. 添加常用符号和标识符
        common_tokens = [
            '{', '}', '(', ')', ';', ',', '.', 
            'Service', 'Controller', 'List', 'Map', ...
        ]
        for token in common_tokens:
            vocab[token] = idx; idx += 1
        
        return vocab
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分割为tokens"""
        # 使用正则表达式匹配标识符、运算符等
        pattern = r'[a-zA-Z_]\w*|[{}();,.=\[\]<>!&|+\-*/%@#$]|\S+'
        tokens = re.findall(pattern, text)
        return tokens
    
    def encode(self, text: str, max_length: int = 128):
        """完整的编码流程"""
        # Step 1: 分词
        tokens = self.tokenize(text)
        # 例如: "public class User" → ["public", "class", "User"]
        
        # Step 2: 添加特殊token
        tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        # 例如: [BOS, "public", "class", "User", EOS]
        
        # Step 3: 转换为IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.UNK_TOKEN])
        # 例如: [2, 15, 23, 89, 1]
        
        # Step 4: 截断或填充
        if len(token_ids) > max_length:
            # 截断：保留前面的tokens，确保EOS在末尾
            token_ids = token_ids[:max_length-1] + [self.vocab[self.EOS_TOKEN]]
        elif len(token_ids) < max_length:
            # 填充：用PAD补齐
            pad_id = self.vocab[self.PAD_TOKEN]
            padding_length = max_length - len(token_ids)
            token_ids.extend([pad_id] * padding_length)
        # 例如(max_length=10): [2, 15, 23, 89, 1, 0, 0, 0, 0, 0]
        
        # Step 5: 创建Attention Mask
        attention_mask = [1] * original_length + [0] * (max_length - original_length)
        # 例如: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        #       ↑^^^^有效内容^^^^^↑↑^^^padding^^^↑
        
        return token_ids, attention_mask
```

### 2.5 Attention Mask（注意力标记）的作用

**核心问题**：如何告诉模型哪些位置是有效内容，哪些是padding？

**答案**：使用Attention Mask！

```python
# 示例
text = "public class User"
max_length = 10

token_ids =      [2, 15, 23, 89, 1, 0, 0, 0, 0, 0]
                 ↑^^^^有效内容^^^^^↑↑^^^padding^^^↑

attention_mask = [1, 1,  1,  1, 1, 0, 0, 0, 0, 0]
                 ↑^^^^有效=1^^^^^↑↑^^padding=0^^^↑
```

**工作原理**：

在Self-Attention计算中，mask会被应用到注意力分数上：

```python
# 伪代码
masked_scores = scores + (1 - mask) * (-1e9)
#              = scores + [0, 0, 0, 0, 0, -1e9, -1e9, ...]

# Softmax后，padding位置的权重趋近于0
attention_weights = softmax(masked_scores)
#                  ≈ [0.2, 0.3, 0.1, 0.4, 0.0, 0.0, 0.0, ...]
```

**为什么需要Attention Mask？**

1. **批量处理**：同时处理多个不同长度的文本
2. **提高效率**：避免模型浪费计算资源在padding上
3. **保证准确性**：防止padding影响模型的注意力分布

### 2.6 图解Tokenization流程

```
┌─────────────────────────────────────────────────────┐
│  原始文本                                            │
│  "public class UserService"                         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: Tokenize（分词）                            │
│  使用正则表达式分割                                    │
│  ["public", "class", "User", "Service"]             │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: 添加特殊Token                               │
│  [BOS, "public", "class", "User", "Service", EOS]   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: 查找词汇表，转换为IDs                        │
│  [2, 15, 23, 89, 234, 1]                            │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: 填充到固定长度（max_length=10）              │
│  [2, 15, 23, 89, 234, 1, 0, 0, 0, 0]                │
│   ↑^^^有效内容^^^^^^^^↑↑^^^^padding^^^^↑             │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 5: 创建Attention Mask                          │
│  [1, 1,  1,  1,  1,   1, 0, 0, 0, 0]                │
│   ↑^^^有效=1^^^^^^^^^↑↑^^^^padding=0^^^↑             │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  输出                                                │
│  token_ids:      [2, 15, 23, 89, 234, 1, 0, 0, 0, 0]│
│  attention_mask: [1, 1,  1,  1,  1,   1, 0, 0, 0, 0]│
└─────────────────────────────────────────────────────┘
```

### 2.7 实战练习

**练习1：手动Tokenization**

给定文本 `"if (x > 0) return x;"`，请完成以下步骤：

1. 分词（不考虑特殊token）
2. 添加BOS和EOS
3. 假设词汇表中：`if=12, (=34, x=56, >=78, 0=90, )=23, return=45, ;=67`
4. 转换为IDs
5. 填充到max_length=12
6. 创建attention_mask

**答案**：
```python
# Step 1: 分词
tokens = ["if", "(", "x", ">", "0", ")", "return", "x", ";"]

# Step 2: 添加特殊token
tokens = [BOS, "if", "(", "x", ">", "0", ")", "return", "x", ";", EOS]

# Step 3: 转换为IDs
token_ids = [2, 12, 34, 56, 78, 90, 23, 45, 56, 67, 1]

# Step 4: 填充到max_length=12
token_ids = [2, 12, 34, 56, 78, 90, 23, 45, 56, 67, 1, 0]

# Step 5: 创建attention_mask
attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
```

---

## 3. 第二章：让机器"看见"文字 - Embedding（嵌入）

### 3.1 核心问题：ID只是编号，没有语义

**问题**：Token IDs `[15, 23, 89]` 只是编号，模型无法理解它们的含义。

**解决方案**：将每个ID映射为一个**向量（Vector）**，这个过程叫**Embedding**。

### 3.2 什么是Embedding？

**Embedding = 将离散符号映射为连续向量[demo_embedding.py](demo_embedding.py)**

💡 **实践演示**: 运行 [demo_embedding.py](demo_embedding.py) 脚本,亲眼看到Embedding如何将ID转换为向量!

```
Token ID → 向量（d_model维）

ID: 15 ("public")  → [0.2, -0.5, 0.8, ..., 0.3]  (128维)
ID: 23 ("class")   → [-0.3, 0.7, -0.1, ..., -0.2]
ID: 89 ("User")    → [0.5, 0.1, 0.3, ..., 0.6]
```

**为什么需要向量？**

1. **捕捉语义关系**：相似的词有相似的向量
2. **支持数学运算**：可以进行加减乘除等操作
3. **神经网络输入**：深度学习模型只能处理连续数值

### 3.3 Embedding矩阵

Embedding本质上是一个**查找表（Lookup Table）**：

```python
# Embedding矩阵形状: [vocab_size, d_model]
# 例如: [1000, 128] 表示1000个token，每个128维

embedding_matrix = nn.Embedding(vocab_size=1000, embedding_dim=128)

# 查询过程
token_ids = [15, 23, 89]
embeddings = embedding_matrix(token_ids)
# embeddings.shape = [3, 128]
```

**可视化Embedding矩阵**：

```
         Dim 0   Dim 1   Dim 2   ...  Dim 127
ID 0     0.1     -0.3    0.5     ...  0.2     ← <PAD>
ID 1     -0.2    0.4     -0.1    ...  -0.5    ← <EOS>
ID 2     0.3     0.1     0.6     ...  0.4     ← <BOS>
ID 3     -0.4    -0.2    0.3     ...  -0.1    ← <UNK>
ID 4     0.5     -0.6    0.2     ...  0.7     ← "public"
ID 5     -0.1    0.8     -0.3    ...  0.2     ← "private"
...
ID 999   0.2     0.3     -0.4    ...  0.6
```

### 3.4 代码实现

来自[transformer.py](file:///E:/Project/llm-codegen-demo/scripts/core/transformer.py#L445)：

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, ...):
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, src, tgt):
        # 将token IDs转换为向量
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        # src.shape: [batch_size, seq_len]
        # src_embedded.shape: [batch_size, seq_len, d_model]
        
        # 例如:
        # src = [[2, 15, 23, 89, 1]]  # [1, 5]
        # src_embedded = [[[0.3, 0.1, ...],   # BOS的向量
        #                  [0.5, -0.6, ...],  # "public"的向量
        #                  [-0.1, 0.8, ...],  # "class"的向量
        #                  [0.2, 0.4, ...],   # "User"的向量
        #                  [-0.2, 0.4, ...]]] # EOS的向量
        # shape: [1, 5, 128]
```

**为什么要乘以 `sqrt(d_model)`？**

这是Transformer论文中的一个技巧，用于缩放embedding值，防止数值过大导致梯度不稳定。

### 3.5 Embedding的可视化

假设有3个词："king"、"queen"、"apple"，它们的embedding可能是：

```
King:   [0.9, 0.8, -0.2, 0.1, ...]  ← 男性、皇室相关
Queen:  [0.8, 0.9, -0.1, 0.2, ...]  ← 女性、皇室相关
Apple:  [-0.1, -0.2, 0.9, -0.3, ...] ← 水果、食物相关

观察：
- King和Queen的前两维很相似（都是皇室）
- Apple与前两者差异很大（完全不同类别）
```

**著名的Embedding算术**：

```
King - Man + Woman ≈ Queen

向量运算：
[0.9, 0.8, -0.2] - [0.7, -0.1, 0.3] + [-0.2, 0.9, 0.1]
= [0.9-0.7+(-0.2), 0.8-(-0.1)+0.9, -0.2-0.3+0.1]
= [0.0, 1.8, -0.4]
≈ Queen的向量 [0.1, 1.7, -0.3]
```

这说明Embedding捕捉到了语义关系！

### 3.6 图解Embedding过程

```
┌─────────────────────────────────────────────────────┐
│  Token IDs                                           │
│  [2, 15, 23, 89, 1]                                 │
│   ↑  ↑   ↑   ↑   ↑                                  │
│  BOS pub cls Usr EOS                                │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Embedding Lookup（查表）                             │
│                                                       │
│  ID 2  → embedding_matrix[2]  → [0.3, 0.1, 0.6, ...]│
│  ID 15 → embedding_matrix[15] → [0.5, -0.6, 0.2, ..]│
│  ID 23 → embedding_matrix[23] → [-0.1, 0.8, -0.3,..]│
│  ID 89 → embedding_matrix[89] → [0.2, 0.4, 0.7, ...]│
│  ID 1  → embedding_matrix[1]  → [-0.2, 0.4, -0.1,..]│
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  输出：嵌入矩阵                                       │
│  Shape: [5, 128]                                     │
│                                                       │
│  [[0.3,  0.1,  0.6, ..., 0.4],   ← BOS             │
│   [0.5, -0.6,  0.2, ..., 0.7],   ← "public"        │
│   [-0.1, 0.8, -0.3, ..., 0.2],   ← "class"         │
│   [0.2,  0.4,  0.7, ..., 0.6],   ← "User"          │
│   [-0.2, 0.4, -0.1, ..., -0.5]]  ← EOS             │
└─────────────────────────────────────────────────────┘
```

### 3.7 关键概念总结

| 概念 | 说明 | 形状变化 |
|------|------|---------|
| **Token IDs** | 离散的整数序列 | `[batch, seq_len]` |
| **Embedding Matrix** | 查找表 | `[vocab_size, d_model]` |
| **Embeddings** | 连续的向量序列 | `[batch, seq_len, d_model]` |

**重要参数**：
- `vocab_size`：词汇表大小（如1000、50000）
- `d_model`：向量维度（如128、768、4096）
- 总参数量 = `vocab_size × d_model`

🔧 **动手实践**：
```bash
# 运行Embedding演示脚本
python docs/learning-resources/llm-fundamentals/demo_embedding.py

# 你将看到:
# 1. Embedding矩阵的创建和初始化
# 2. Token ID到向量的查找过程
# 3. 训练前后向量的变化
# 4. 可视化Embedding空间
# 5. 数学原理解释
```

---

## 4. 第三章：告诉模型"顺序" - Positional Encoding

### 4.1 核心问题：Transformer不知道顺序

**问题**：Transformer的Self-Attention机制是**无序的**，它不知道"ABC"和"CBA"的区别。

```
示例：
句子1: "I love you"
句子2: "you love I"

如果没有位置信息，Self-Attention看到的都是相同的词集合：
{"I", "love", "you"}
```

**解决方案**：给每个位置添加独特的"位置指纹"，这就是**Positional Encoding（位置编码）**。

### 4.2 什么是Positional Encoding？

**Positional Encoding = 给每个位置添加唯一的向量标识**

```
原始Embedding + 位置编码 = 带位置信息的Embedding

Token 0 ("I"):    [0.3, 0.1, 0.6, ...] + [0.0, 1.0, 0.0, ...] = [0.3, 1.1, 0.6, ...]
Token 1 ("love"): [0.5, -0.6, 0.2, ...] + [0.84, 0.54, 0.84, ...] = [1.34, -0.06, 1.04, ...]
Token 2 ("you"):  [-0.1, 0.8, -0.3, ...] + [0.91, -0.42, 0.91, ...] = [0.81, 0.38, 0.61, ...]
```

### 4.3 为什么不用简单的位置ID？

**方案1：直接使用位置ID（不推荐）**
```
Position 0 → [0, 0, 0, ..., 0]
Position 1 → [1, 1, 1, ..., 1]
Position 2 → [2, 2, 2, ..., 2]
```
❌ 问题：数值范围不一致，难以训练

**方案2：One-hot编码（不推荐）**
```
Position 0 → [1, 0, 0, 0, ...]
Position 1 → [0, 1, 0, 0, ...]
Position 2 → [0, 0, 1, 0, ...]
```
❌ 问题：维度太高（如果max_length=1000，就需要1000维）

**方案3：正弦/余弦函数（Transformer采用）✅**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
✅ 优点：
- 每个位置有唯一的编码
- 相邻位置的编码相似但不同
- 可以外推到更长的序列

### 4.4 位置编码公式详解

```python
# 来自 attention.py 第442-466行
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=500):
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # position: [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # div_term: [d_model/2]
        # 使用指数衰减让不同维度关注不同频率
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        
        # 添加到buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x.shape: [batch, seq_len, d_model]
        # 截取相应长度的位置编码并添加
        x = x + self.pe[:, :seq_len, :]
        return x
```

**公式解读**：

```
PE(pos, i) = 
  sin(pos / 10000^(i/d_model))  如果i是偶数
  cos(pos / 10000^(i/d_model))  如果i是奇数

其中：
- pos: 位置（0, 1, 2, ..., max_len-1）
- i: 维度索引（0, 1, 2, ..., d_model-1）
- 10000: 基频，控制波长范围
```

**直观理解**：

```
d_model = 8 的例子：

位置0: [sin(0/1), cos(0/1), sin(0/100), cos(0/100), sin(0/10000), cos(0/10000), ...]
      = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

位置1: [sin(1/1), cos(1/1), sin(1/100), cos(1/100), sin(1/10000), cos(1/10000), ...]
      = [0.84, 0.54, 0.01, 1.0, 0.0001, 1.0, ...]

位置2: [sin(2/1), cos(2/1), sin(2/100), cos(2/100), sin(2/10000), cos(2/10000), ...]
      = [0.91, -0.42, 0.02, 1.0, 0.0002, 1.0, ...]

观察：
- 低维度（0,1）变化快 → 捕捉局部位置信息
- 高维度（4,5,6,7）变化慢 → 捕捉全局位置信息
```

### 4.5 图解Positional Encoding

```
┌─────────────────────────────────────────────────────┐
│  Embeddings（不带位置信息）                           │
│  Shape: [5, 128]                                     │
│                                                       │
│  [[0.3,  0.1,  0.6, ..., 0.4],   ← "public"        │
│   [0.5, -0.6,  0.2, ..., 0.7],   ← "class"         │
│   [-0.1, 0.8, -0.3, ..., 0.2],   ← "User"          │
│   [0.2,  0.4,  0.7, ..., 0.6],   ← "Service"       │
│   [-0.2, 0.4, -0.1, ..., -0.5]]  ← EOS             │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Positional Encoding（位置编码）                      │
│  Shape: [5, 128]                                     │
│                                                       │
│  [[0.0,  1.0,  0.0, ..., 1.0],   ← Position 0      │
│   [0.84, 0.54, 0.84, ..., 0.54], ← Position 1      │
│   [0.91, -0.42, 0.91, ..., -0.42],← Position 2     │
│   [0.14, 0.99, 0.14, ..., 0.99], ← Position 3      │
│   [-0.76, 0.65, -0.76, ..., 0.65]]← Position 4     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  相加：Embeddings + PE                               │
│  Shape: [5, 128]                                     │
│                                                       │
│  [[0.3,  1.1,  0.6, ..., 1.4],   ← "public"@Pos0   │
│   [1.34, -0.06, 1.04, ..., 1.24],← "class"@Pos1    │
│   [0.81, 0.38, 0.61, ..., -0.22],← "User"@Pos2     │
│   [0.34, 1.39, 0.84, ..., 1.59], ← "Service"@Pos3  │
│   [-0.96, 1.05, -0.86, ..., 0.15]]← EOS@Pos4       │
│                                                       │
│  现在每个向量都包含了位置信息！                        │
└─────────────────────────────────────────────────────┘
```

### 4.6 可视化位置编码

位置编码可以用热力图展示：

```
Dimension →
     0    1    2    3    4    5    6    7
Pos 0 [██░░░░░░] [████████] [██░░░░░░] [████████] ...
Pos 1 [█████░░░] [████░░░░] [█████░░░] [████░░░░] ...
Pos 2 [██████░░] [██░░░░░░] [██████░░] [██░░░░░░] ...
Pos 3 [████░░░░] [███████░] [████░░░░] [███████░] ...
Pos 4 [██░░░░░░] [█████░░░] [██░░░░░░] [█████░░░] ...
  ↓
  每行的颜色模式都不同，这就是"位置指纹"
```

### 4.7 关键要点

**为什么位置编码有效？**

1. **唯一性**：每个位置有独特的编码模式
2. **相对位置**：模型可以学习到"位置i和位置j的距离"
3. **可外推**：即使训练时只看到100个位置，推理时可以处理更长的序列

**实际模型的位置编码**：

| 模型 | 位置编码类型 | 最大长度 |
|------|------------|---------|
| BERT | Sinusoidal | 512 |
| GPT-2 | Learned | 1024 |
| Llama | RoPE (Rotary) | 2048-8192 |
| Claude | ALiBi | 100K+ |

---

## 5. 第四章：注意力机制 - Attention（核心！）

> ⚠️ **这是整个Transformer最重要的部分！** 请仔细阅读。

### 5.1 核心思想：像人类一样"关注重点"

**生活例子**：

当你阅读这句话时：
> "小明昨天在图书馆借了一本关于人工智能的书"

如果被问到"**谁借了书？**"，你的大脑会：
- 🔍 **重点关注**："小明"（主语）、"借"（动作）
- 👀 **次要关注**："昨天"、"图书馆"
- ❌ **忽略**："一本"、"关于"

**注意力机制就是让AI学会这种能力！**

### 5.2 什么是Attention？

**Attention = 让每个token知道应该关注序列中的哪些其他token**

```
句子: "public class UserService extends BaseService"

当处理"class"这个词时：
- 高度关注: "public"（修饰符）、"UserService"（类名）
- 中度关注: "extends"（继承关键字）
- 低度关注: "BaseService"（父类，距离较远）

Attention机制会自动学习这种关注模式！
```

### 5.3 Q、K、V的概念

Attention的核心是三个向量：**Query（查询）、Key（键）、Value（值）**

**类比数据库查询**：

```python
# 想象你在图书馆找书

Query (Q): 你的查询请求
  "我想找关于Python编程的书"

Key (K): 每本书的标签/索引
  Book1: "Python入门"
  Book2: "Java高级"
  Book3: "Python实战"

Value (V): 书的实际内容
  Book1: <Python入门的详细内容>
  Book2: <Java高级的详细内容>
  Book3: <Python实战的详细内容>

Attention的过程：
1. 计算Q和每个K的相似度 → 找到相关的书
2. 根据相似度加权求和V → 获取相关内容
```

**在Transformer中**：

```
对于每个token，我们都有Q、K、V三个向量：

Token "class":
  Q_class = [0.5, -0.3, 0.8, ...]  # "我在找什么？"
  K_class = [0.3, 0.7, -0.2, ...]  # "我有什么特征？"
  V_class = [0.1, 0.9, 0.4, ...]  # "我的实际内容是什么？"

Token "public":
  Q_public = [...]
  K_public = [...]
  V_public = [...]

Token "User":
  Q_User = [...]
  K_User = [...]
  V_User = [...]
```

### 5.4 Attention的计算步骤

**公式**（来自[attention.py](file:///E:/Project/llm-codegen-demo/scripts/core/attention.py#L159-L160)）：

```python
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

让我们一步步拆解：

#### Step 1: 计算相似度分数

```python
scores = Q @ K^T
```

**示例**：

```
假设有3个token，d_k=4：

Q = [[q1_1, q1_2, q1_3, q1_4],   # Token 1的Query
     [q2_1, q2_2, q2_3, q2_4],   # Token 2的Query
     [q3_1, q3_2, q3_3, q3_4]]   # Token 3的Query

K = [[k1_1, k1_2, k1_3, k1_4],   # Token 1的Key
     [k2_1, k2_2, k2_3, k2_4],   # Token 2的Key
     [k3_1, k3_2, k3_3, k3_4]]   # Token 3的Key

scores = Q @ K^T = 
  [[Q1·K1, Q1·K2, Q1·K3],   # Token 1对各个token的关注度
   [Q2·K1, Q2·K2, Q2·K3],   # Token 2对各个token的关注度
   [Q3·K1, Q3·K2, Q3·K3]]   # Token 3对各个token的关注度

其中 Q1·K1 = q1_1*k1_1 + q1_2*k1_2 + q1_3*k1_3 + q1_4*k1_4 （点积）
```

**物理意义**：
- `scores[i][j]` 表示 **Token i 对 Token j 的关注程度**
- 值越大表示越相关

#### Step 2: 缩放（Scaling）

```python
scaled_scores = scores / √d_k
```

**为什么要除以√d_k？**

当d_k较大时，点积结果会很大，导致softmax的梯度接近0（梯度消失）。

```
示例：
d_k = 4 → √d_k = 2
scores = [[8.0, 2.0, 1.0],
          [3.0, 7.0, 2.0],
          [1.0, 2.0, 6.0]]

scaled_scores = scores / 2 = 
               [[4.0, 1.0, 0.5],
                [1.5, 3.5, 1.0],
                [0.5, 1.0, 3.0]]
```

#### Step 3: Softmax归一化

```python
attention_weights = softmax(scaled_scores)
```

**Softmax的作用**：将分数转换为概率分布（每行的和为1）

```
softmax([4.0, 1.0, 0.5]) = 
  [e^4.0/(e^4.0+e^1.0+e^0.5), e^1.0/(...), e^0.5/(...)]
= [0.84, 0.11, 0.05]

含义：Token 1有84%的注意力在Token 1自己，11%在Token 2，5%在Token 3
```

**完整的attention_weights矩阵**：

```
attention_weights = 
  [[0.84, 0.11, 0.05],   # Token 1的关注分布
   [0.15, 0.78, 0.07],   # Token 2的关注分布
   [0.08, 0.12, 0.80]]   # Token 3的关注分布

每行的和都是1.0！
```

#### Step 4: 加权求和

```python
output = attention_weights @ V
```

**含义**：根据注意力权重，从Value中提取相关信息

```
V = [[v1_1, v1_2, v1_3, v1_4],   # Token 1的Value
     [v2_1, v2_2, v2_3, v2_4],   # Token 2的Value
     [v3_1, v3_2, v3_3, v3_4]]   # Token 3的Value

output[0] = 0.84 * V[0] + 0.11 * V[1] + 0.05 * V[2]
          = 84%的Token1信息 + 11%的Token2信息 + 5%的Token3信息

output[1] = 0.15 * V[0] + 0.78 * V[1] + 0.07 * V[2]
          = 15%的Token1信息 + 78%的Token2信息 + 7%的Token3信息

output[2] = 0.08 * V[0] + 0.12 * V[1] + 0.80 * V[2]
          = 8%的Token1信息 + 12%的Token2信息 + 80%的Token3信息
```

**结果**：每个token的输出都融合了全局的上下文信息！

### 5.5 完整示例

让我们用一个具体的例子走一遍整个Attention流程：

```python
# 输入句子: "I love AI"
# 假设d_k = 2（简化版）

# Step 0: 获取Q、K、V（通过线性变换）
Q = [[0.5, 0.3],   # "I"的Query
     [0.2, 0.8],   # "love"的Query
     [0.7, 0.1]]   # "AI"的Query

K = [[0.4, 0.6],   # "I"的Key
     [0.1, 0.9],   # "love"的Key
     [0.8, 0.2]]   # "AI"的Key

V = [[0.3, 0.7],   # "I"的Value
     [0.6, 0.4],   # "love"的Value
     [0.9, 0.1]]   # "AI"的Value

# Step 1: 计算scores = Q @ K^T
scores = [[0.5*0.4 + 0.3*0.6,  0.5*0.1 + 0.3*0.9,  0.5*0.8 + 0.3*0.2],
          [0.2*0.4 + 0.8*0.6,  0.2*0.1 + 0.8*0.9,  0.2*0.8 + 0.8*0.2],
          [0.7*0.4 + 0.1*0.6,  0.7*0.1 + 0.1*0.9,  0.7*0.8 + 0.1*0.2]]
       
       = [[0.38, 0.32, 0.46],
          [0.56, 0.74, 0.32],
          [0.34, 0.16, 0.58]]

# Step 2: 缩放（d_k=2，√2≈1.414）
scaled_scores = scores / 1.414
              = [[0.27, 0.23, 0.33],
                 [0.40, 0.52, 0.23],
                 [0.24, 0.11, 0.41]]

# Step 3: Softmax
attention_weights = [[0.32, 0.31, 0.37],   # "I"关注: 32%自己, 31%"love", 37%"AI"
                     [0.29, 0.37, 0.34],   # "love"关注: 29%"I", 37%自己, 34%"AI"
                     [0.30, 0.27, 0.43]]   # "AI"关注: 30%"I", 27%"love", 43%自己

# Step 4: 加权求和 output = attention_weights @ V
output[0] = 0.32*[0.3,0.7] + 0.31*[0.6,0.4] + 0.37*[0.9,0.1]
          = [0.096,0.224] + [0.186,0.124] + [0.333,0.037]
          = [0.615, 0.385]

output[1] = 0.29*[0.3,0.7] + 0.37*[0.6,0.4] + 0.34*[0.9,0.1]
          = [0.087,0.203] + [0.222,0.148] + [0.306,0.034]
          = [0.615, 0.385]

output[2] = 0.30*[0.3,0.7] + 0.27*[0.6,0.4] + 0.43*[0.9,0.1]
          = [0.090,0.210] + [0.162,0.108] + [0.387,0.043]
          = [0.639, 0.361]

# 最终输出
output = [[0.615, 0.385],
          [0.615, 0.385],
          [0.639, 0.361]]
```

### 5.6 图解Attention流程

```
┌─────────────────────────────────────────────────────┐
│  Input Tokens                                        │
│  ["I", "love", "AI"]                                │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Linear Projections (W_q, W_k, W_v)                 │
│                                                       │
│  Q = [[0.5, 0.3],   K = [[0.4, 0.6],   V = [[0.3, 0.7],│
│       [0.2, 0.8],        [0.1, 0.9],        [0.6, 0.4],│
│       [0.7, 0.1]]        [0.8, 0.2]]        [0.9, 0.1]]│
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: Compute Scores (Q @ K^T)                   │
│                                                       │
│  scores = [[0.38, 0.32, 0.46],                      │
│            [0.56, 0.74, 0.32],                      │
│            [0.34, 0.16, 0.58]]                      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Scale (/ √d_k)                             │
│                                                       │
│  scaled = [[0.27, 0.23, 0.33],                      │
│            [0.40, 0.52, 0.23],                      │
│            [0.24, 0.11, 0.41]]                      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Softmax                                    │
│                                                       │
│  attn_weights = [[0.32, 0.31, 0.37],                │
│                  [0.29, 0.37, 0.34],                │
│                  [0.30, 0.27, 0.43]]                │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: Weighted Sum (attn_weights @ V)            │
│                                                       │
│  output = [[0.615, 0.385],                          │
│            [0.615, 0.385],                          │
│            [0.639, 0.361]]                          │
│                                                       │
│  每个token的输出都融合了全局信息！                    │
└─────────────────────────────────────────────────────┘
```

### 5.7 代码实现

来自[attention.py](file:///E:/Project/llm-codegen-demo/scripts/core/attention.py#L140-L394)：

```python
class MultiHeadAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Step 1: 线性变换并分割成多个头
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        # Q, K, V shape: [batch, nhead, seq_len, d_k]
        
        # Step 2: 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: [batch, nhead, seq_len_q, seq_len_k]
        
        # Step 3: 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Softmax
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: 加权求和
        context = torch.matmul(attention_weights, V)
        # context shape: [batch, nhead, seq_len, d_k]
        
        # Step 6: 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # context shape: [batch, seq_len, d_model]
        
        # Step 7: 输出线性变换
        output = self.W_o(context)
        
        return output, attention_weights
```

### 5.8 关键要点总结

**Attention的本质**：
1. **Query-Key匹配**：找出相关的token
2. **Softmax归一化**：转换为概率分布
3. **Value加权**：融合相关信息

**为什么Attention强大？**
- ✅ 捕捉长距离依赖（不像RNN那样受限于步数）
- ✅ 并行计算（不像RNN那样需要逐步计算）
- ✅ 可解释性强（可以可视化注意力权重）

**实际应用中的Attention**：
- Self-Attention：同一个序列内部的token互相观察
- Cross-Attention：一个序列关注另一个序列（如翻译任务）
- Masked Attention：防止看到未来token（用于Decoder）

---

## 6. 第五章：多头注意力 - Multi-Head Attention

### 6.1 为什么需要"多头"？

**单头的局限性**：

如果只有一个注意力头，模型只能从**一个角度**理解token之间的关系。

```
句子: "public class UserService extends BaseService implements IUserService"

单头注意力可能只学到：
- 语法关系: "class" → "public"

但错过了：
- 语义关系: "UserService" → "Service"
- 结构关系: "extends" → "BaseService"
- 接口关系: "implements" → "IUserService"
```

**多头的优势**：

用多个"视角"同时观察，每个头学习不同的关系模式。

```
8个注意力头可能分别学习：
Head 0: 语法修饰符关系 (public → class)
Head 1: 命名模式 (UserService → Service)
Head 2: 继承关系 (extends → BaseService)
Head 3: 接口实现 (implements → IUserService)
Head 4: 关键字关联 (class → extends)
Head 5: 类型推断 (UserService → User)
Head 6: 访问控制 (public → private)
Head 7: 方法签名 (findById → Long id)

最后将所有头的信息拼接起来 → 全面理解！
```

### 6.2 Multi-Head Attention的结构

**架构图**：

```
Input: [batch, seq_len, d_model]
        ↓
   ┌────────┬────────┬────────┐
   │ Head 0 │ Head 1 │ ...    │  ← 并行计算
   │        │        │        │
   │ Q0,K0,V0│Q1,K1,V1│ ...   │
   │   ↓    │   ↓    │        │
   │ Attn 0 │ Attn 1 │ ...    │
   └────────┴────────┴────────┘
        ↓
  Concatenate (拼接)
        ↓
  Linear Projection (W_o)
        ↓
Output: [batch, seq_len, d_model]
```

**关键参数**：

```python
d_model = 128    # 模型总维度
nhead = 8        # 注意力头数
d_k = d_model / nhead = 16  # 每个头的维度
```

**形状变化**：

```
Input:  [batch, seq_len, d_model]
        ↓ Split into heads
Q/K/V:  [batch, nhead, seq_len, d_k]
        ↓ Attention computation
Output per head: [batch, nhead, seq_len, d_k]
        ↓ Concatenate
Concatenated: [batch, seq_len, nhead * d_k] = [batch, seq_len, d_model]
        ↓ Linear projection
Final Output: [batch, seq_len, d_model]
```

### 6.3 代码实现详解

来自[attention.py](file:///E:/Project/llm-codegen-demo/scripts/core/attention.py#L77-L139)：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个头的维度
        
        # Q, K, V的线性变换（每个头共享）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # ════════════════════════════════════════════
        # Step 1: 线性变换并分割成多个头
        # ════════════════════════════════════════════
        # 
        # 形状变换详解：
        # [batch, seq_len, d_model]
        #   ↓ self.W_q(query)  (线性投影: d_model → d_model)
        # [batch, seq_len, d_model]
        #   ↓ .view(batch, seq_len, nhead, d_k)  (重塑)
        # [batch, seq_len, 8, 16]
        #   ↓ .transpose(1, 2)  (交换维度，方便矩阵运算)
        # [batch, 8, seq_len, 16]  (最终Q/K/V的形状)
        
        Q = self.W_q(query).view(batch_size, seq_len_q, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        
        # ════════════════════════════════════════════
        # Step 2: 计算注意力分数 (Scaled Dot-Product)
        # ════════════════════════════════════════════
        # 
        # Q: [batch, 8, seq_len_q, 16]
        # K^T: [batch, 8, 16, seq_len_k]  (转置最后两维)
        #   ↓ torch.matmul(Q, K.transpose(-2, -1))
        # scores: [batch, 8, seq_len_q, seq_len_k]
        #   ↓ / math.sqrt(self.d_k)
        # scores: [batch, 8, seq_len_q, seq_len_k]  (缩放后)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # ════════════════════════════════════════════
        # Step 3: 应用mask（如果有）
        # ════════════════════════════════════════════
        if mask is not None:
            # mask中值为0的位置会被填充为-1e9
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # ════════════════════════════════════════════
        # Step 4: Softmax获取注意力权重
        # ════════════════════════════════════════════
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        
        # ════════════════════════════════════════════
        # Step 5: 加权求和 (Context聚合)
        # ════════════════════════════════════════════
        # attention_weights: [batch, 8, seq_len_q, seq_len_k]
        # V:                 [batch, 8, seq_len_k, 16]
        #   ↓ torch.matmul(attention_weights, V)
        # context:           [batch, 8, seq_len_q, 16]
        
        context = torch.matmul(attention_weights, V)
        
        # ════════════════════════════════════════════
        # Step 6: 合并多头
        # ════════════════════════════════════════════
        # [batch, 8, seq_len_q, 16]  (8个头分开)
        #   ↓ .transpose(1, 2)  (交换维度)
        # [batch, seq_len_q, 8, 16]
        #   ↓ .contiguous().view(batch, seq_len_q, 8*16)  (重塑)
        # [batch, seq_len_q, 128]  (合并回d_model)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # ════════════════════════════════════════════
        # Step 7: 输出线性变换
        # ════════════════════════════════════════════
        output = self.W_o(context)
        
        return output, attention_weights
```

### 6.4 可视化多头注意力

假设有4个token和4个头：

```
Head 0的注意力分布:
        public  class  User  Service
public   0.7    0.2    0.05   0.05
class    0.3    0.5    0.1    0.1
User     0.1    0.1    0.6    0.2
Service  0.05   0.05   0.3    0.6

Head 1的注意力分布:
        public  class  User  Service
public   0.6    0.3    0.05   0.05
class    0.2    0.4    0.2    0.2
User     0.05   0.15   0.5    0.3
Service  0.05   0.1    0.25   0.6

Head 2的注意力分布:
        public  class  User  Service
public   0.8    0.15   0.03   0.02
class    0.4    0.45   0.1    0.05
User     0.15   0.1    0.55   0.2
Service  0.1    0.05   0.2    0.65

Head 3的注意力分布:
        public  class  User  Service
public   0.5    0.35   0.1    0.05
class    0.25   0.35   0.25   0.15
User     0.1    0.2    0.45   0.25
Service  0.05   0.15   0.3    0.5

观察：
- 不同头的注意力模式不同
- Head 0更关注自身（对角线值大）
- Head 3更均匀分布
- 这体现了"多视角"的优势
```

### 6.5 真实模型的配置对比

| 模型 | d_model | nhead | d_k | 说明 |
|------|---------|-------|-----|------|
| **本项目** | 128 | 8 | 16 | 教学简化版 |
| **BERT-base** | 768 | 12 | 64 | 经典配置 |
| **GPT-2 small** | 768 | 12 | 64 | 与BERT类似 |
| **GPT-3** | 12288 | 96 | 128 | 超大模型 |
| **Llama-2-7B** | 4096 | 32 | 128 | 现代架构 |

**经验法则**：
- `d_k` 通常在 64-128 之间
- `nhead = d_model / d_k`
- `d_model` 越大，模型表达能力越强

### 6.6 关键要点

**为什么Multi-Head比Single-Head好？**

1. **多视角学习**：每个头可以关注不同类型的关系
2. **稳定性**：多个头的平均效果更稳定
3. **表达能力**：相当于多个子空间的组合

**计算复杂度**：
- Single-Head: O(seq_len² × d_model)
- Multi-Head: O(seq_len² × d_model) （相同，因为并行计算）

**内存占用**：
- 需要存储nhead个注意力权重矩阵
- 但对于现代GPU来说不是问题

---

由于文档非常长，我已经创建了前半部分。这份文档涵盖了：

1. ✅ Tokenization（分词）
2. ✅ Embedding（词嵌入）
3. ✅ Positional Encoding（位置编码）
4. ✅ Attention Mechanism（注意力机制）- 核心章节
5. ✅ Multi-Head Attention（多头注意力）

接下来还需要继续编写：
- Encoder架构
- Decoder架构
- 完整Transformer
- 自回归生成
- 采样策略
- KV Cache优化
- 后处理
- 完整流程总结
- 实战练习

这份文档的特点是：
- 📚 **极其详细**：每个概念都有多层次的解释
- 🎨 **图文并茂**：大量ASCII图表和流程图
- 💻 **代码驱动**：结合项目实际代码
- 🎯 **初学者友好**：从最基础的概念开始，逐步深入
- ✨ **实用导向**：包含实战练习和常见问题

你想让我继续完成剩余的章节吗？
