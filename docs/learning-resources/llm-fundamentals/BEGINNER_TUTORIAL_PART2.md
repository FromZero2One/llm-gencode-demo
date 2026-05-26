# 🎓 大模型底层原理 - 从零开始的完整教程(下)

> **阅读时间**: 45-60分钟  
> **适合人群**: 已阅读上篇(1-5章)的学习者  
> **学习目标**: 掌握Encoder-Decoder架构、生成流程、优化策略等高级主题

---

## 📖 目录

7. [第六章:Encoder - 理解输入](#7-第六章encoder---理解输入)
8. [第七章:Decoder - 生成输出](#8-第七章decoder---生成输出)
9. [第八章:完整的Transformer架构](#9-第八章完整的transformer架构)
10. [第九章:自回归生成](#10-第九章自回归生成)
11. [第十章:采样策略](#11-第十章采样策略)
12. [第十一章:性能优化 - KV Cache](#12-第十一章性能优化---kv-cache)
13. [第十二章:后处理](#13-第十二章后处理)
14. [第十三章:完整流程总结](#14-第十三章完整流程总结)

---

## 7. 第六章:Encoder - 理解输入

### 7.1 Encoder的作用

**通俗类比**:Encoder就像一个"阅读理解专家"。

```
你给它一篇文章,它会:
1. 仔细阅读每个词
2. 理解词与词之间的关系
3. 提取关键信息
4. 形成对整篇文章的深度理解

最终输出:每个词的"上下文感知表示"
```

**在代码生成场景中的作用**:
- 输入:`public class UserService {`
- Encoder理解:这是一个类定义的开始,"UserService"是类名,"public"是访问修饰符
- 输出:每个token都携带了完整的上下文信息

### 7.2 Encoder的内部结构

一个Encoder Block由以下组件组成:

```
┌─────────────────────────────────────┐
│         Encoder Block (重复N次)      │
│                                     │
│  输入: X (序列长度 × 隐藏维度)       │
│           ↓                         │
│  ┌──────────────────────────┐       │
│  │ Multi-Head Self-Attention│       │
│  └──────────────────────────┘       │
│           ↓                         │
│  ┌──────────────────────────┐       │
│  │    Add & LayerNorm       │       │
│  └──────────────────────────┘       │
│           ↓                         │
│  ┌──────────────────────────┐       │
│  │   Feed Forward Network   │       │
│  └──────────────────────────┘       │
│           ↓                         │
│  ┌──────────────────────────┐       │
│  │    Add & LayerNorm       │       │
│  └──────────────────────────┘       │
│           ↓                         │
│  输出: X' (增强后的表示)             │
└─────────────────────────────────────┘
```

**关键点**:
- ✅ "Self" Attention:每个token关注序列中的其他token
- ✅ Add & LayerNorm:残差连接 + 层归一化,防止梯度消失
- ✅ Feed Forward:非线性变换,增强表达能力
- ✅ 通常有6-12个这样的Block堆叠

### 7.3 逐层详解

#### 7.3.1 Multi-Head Self-Attention

**问题**:为什么需要"Multi-Head"?

**答案**:不同的"头"关注不同的关系类型。

**示例**:对于句子 `"UserService类提供了用户管理功能"`

```
Head 1(语法关系):
  "UserService" ←→ "类"        (名词-类别关系)
  "提供" ←→ "功能"              (动词-宾语关系)

Head 2(语义关系):
  "用户" ←→ "UserService"      (实体-服务关系)
  "管理" ←→ "用户"              (动作-对象关系)

Head 3(位置关系):
  相邻词之间的局部依赖

Head 4(全局关系):
  整个句子的主题聚焦
```

**代码实现**(来自[attention.py](file:///E:/Project/llm-codegen-demo/scripts/core/attention.py)):

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 512/8 = 64
        
        # 为每个head创建独立的Q/K/V投影
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 分割成多个head
        Q = Q.view(batch, seq_len, 8, 64).transpose(1, 2)
        K = K.view(batch, seq_len, 8, 64).transpose(1, 2)
        V = V.view(batch, seq_len, 8, 64).transpose(1, 2)
        
        # 并行计算attention
        attention_output = scaled_dot_product_attention(Q, K, V)
        
        # 合并所有head
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch, seq_len, 512)
        
        output = self.W_o(attention_output)
        return output
```

#### 7.3.2 Add & LayerNorm

**残差连接**:
```python
x_after_attention = x + multi_head_attention(x)
```

**为什么需要残差连接?**
```
问题:深层网络容易出现梯度消失

解决方案:残差连接让梯度可以直接"跳过"某些层传播

类比:高速公路上的立交桥
- 主路(残差连接):信息直接传递
- 匝道(Sublayer):进行复杂处理
- 即使匝道堵车,主路仍然畅通
```

**Layer Normalization**:
```python
x_normalized = LayerNorm(x)
# 使均值为0,方差为1
```

**作用**:稳定训练、加速收敛、减少对学习率敏感度

#### 7.3.3 Feed Forward Network (FFN)

```python
class FeedForward(nn.Module):
    def __init__(self, hidden_size=512, ff_size=2048):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, ff_size),   # 升维: 512 → 2048
            nn.ReLU(),
            nn.Linear(ff_size, hidden_size),   # 降维: 2048 → 512
        )
```

**为什么需要先升维再降维?**
```
类比:思考问题的过程

输入(512维) → 扩展思路(2048维) → 提炼结论(512维)

1. 升维:将信息映射到更高维空间,更容易发现模式
2. ReLU:引入非线性,增强表达能力
3. 降维:压缩回原始维度,便于下一层处理
```

### 7.4 Encoder的完整流程

```python
class Encoder(nn.Module):
    def __init__(self, num_layers=6, hidden_size=512, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(hidden_size, num_heads) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
```

### 7.5 实际运行示例

假设输入:`"public class User {"`

**经过Encoder后的变化**:

```
第0层(输入):
  "public":  [0.1, 0.2, ..., 0.9]
  "class":   [0.3, 0.4, ..., 0.7]
  "User":    [0.5, 0.6, ..., 0.5]
  "{":       [0.7, 0.8, ..., 0.3]

第6层后(最终输出):
  "public":  [0.2, 0.3, ..., 0.8]  # 深度上下文感知表示
  "class":   [0.4, 0.5, ..., 0.6]
  "User":    [0.6, 0.7, ..., 0.4]
  "{":       [0.8, 0.9, ..., 0.2]
```

**关键理解**:
- 每个token的向量不再是孤立的
- 它编码了整个序列的信息
- "User"不仅知道自己是一个标识符,还知道前面有"public class",后面有"{"

---

## 8. 第七章:Decoder - 生成输出

### 8.1 Decoder的作用

**通俗类比**:Decoder就像一个"创意写作者"。

```
Encoder理解了输入后,Decoder负责:
1. 基于理解的内容
2. 一个字一个字地生成输出
3. 保证生成的内容连贯、合理
```

### 8.2 Decoder的特殊之处

| 特性 | Encoder | Decoder |
|------|---------|---------|
| **注意力类型** | Self-Attention | Masked Self-Attention + Cross-Attention |
| **可见范围** | 看到整个序列 | 只能看到已生成的部分 |
| **处理方式** | 并行处理 | 逐个生成(自回归) |
| **目的** | 理解输入 | 生成输出 |

### 8.3 Decoder的内部结构

```
┌─────────────────────────────────────────────┐
│         Decoder Block (重复N次)              │
│                                             │
│  输入: Y (已生成的序列)                      │
│         Encoder_Output (编码后的输入)        │
│           ↓                                 │
│  ┌──────────────────────────────┐           │
│  │ Masked Self-Attention        │           │
│  └──────────────────────────────┘           │
│           ↓                                 │
│  ┌──────────────────────────────┐           │
│  │    Add & LayerNorm           │           │
│  └──────────────────────────────┘           │
│           ↓                                 │
│  ┌──────────────────────────────┐           │
│  │ Cross-Attention              │           │
│  │ (关注Encoder的输出)           │           │
│  └──────────────────────────────┘           │
│           ↓                                 │
│  ┌──────────────────────────────┐           │
│  │    Add & LayerNorm           │           │
│  └──────────────────────────────┘           │
│           ↓                                 │
│  ┌──────────────────────────────┐           │
│  │   Feed Forward Network       │           │
│  └──────────────────────────────┘           │
│           ↓                                 │
│  ┌──────────────────────────────┐           │
│  │    Add & LayerNorm           │           │
│  └──────────────────────────────┘           │
│           ↓                                 │
│  输出: Y'                                   │
└─────────────────────────────────────────────┘
```

### 8.4 核心组件详解

#### 8.4.1 Masked Self-Attention

**为什么要"Masked"?** 防止模型"偷看"未来的token。

**Mask矩阵可视化**:

```
假设生成序列长度为4: ["public", "class", "User", "{"]

应用Causal Mask后:
        public  class  User  {
public   1.0     0.0    0.0   0.0
class    0.45    0.55   0.0   0.0
User     0.30    0.35   0.35  0.0
{        0.22    0.25   0.28  0.25

结果:每个token只关注自己和前面的token
```

#### 8.4.2 Cross-Attention

**Cross-Attention是什么?** 让Decoder在生成时"参考"Encoder的理解。

**类比**:
```
写作文的场景:
- Encoder:阅读理解题目材料,提取关键信息
- Decoder:开始写作文
- Cross-Attention:写作过程中不断回头看题目材料,确保不跑题
```

**工作流程**:
```python
# Query: 来自Decoder(当前正在生成的内容)
Q = decoder_embedding

# Key, Value: 来自Encoder(输入的理解)
K = encoder_output
V = encoder_output

# Attention计算
attention_scores = Q @ K.transpose(-2, -1)
attention_weights = softmax(attention_scores / sqrt(d_k))
output = attention_weights @ V
```

### 8.5 Decoder的完整实现

```python
class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        
        self.masked_attention = MaskedMultiHeadAttention(hidden_size, num_heads)
        self.cross_attention = CrossAttention(hidden_size, num_heads)
        self.ffn = FeedForward(hidden_size)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
    
    def forward(self, x, encoder_output, mask=None):
        # Masked Self-Attention
        attn1 = self.masked_attention(x, mask=mask)
        x = self.norm1(x + attn1)
        
        # Cross-Attention
        attn2 = self.cross_attention(query=x, key=encoder_output, value=encoder_output)
        x = self.norm2(x + attn2)
        
        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x
```

### 8.6 从Decoder输出到Token预测

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=512):
        super().__init__()
        self.decoder = Decoder(...)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, encoder_input, decoder_input):
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)
        
        logits = self.output_projection(decoder_output)
        probabilities = softmax(logits, dim=-1)
        
        return probabilities
```

**示例**:
```
Decoder输出(最后一个token的hidden state):
  h = [0.2, 0.5, -0.3, ..., 0.8]

经过Linear层:
  logits = [2.5, -1.2, 0.8, 3.1, ...]

经过Softmax:
  probs = [0.15, 0.02, 0.08, 0.25, ...]
  
  选择概率最高的 "public" (0.25) 作为下一个token
```

---

## 9. 第八章:完整的Transformer架构

### 9.1 Encoder-Decoder整合

```
┌──────────────────────────────────────────────────────┐
│                  Transformer 架构                     │
│                                                      │
│  输入序列: "创建一个用户服务类"                        │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │           Encoder (6层)                    │     │
│  │  Input Embedding + Positional Encoding     │     │
│  │         ↓                                  │     │
│  │  [Encoder Block × 6]                       │     │
│  │         ↓                                  │     │
│  │  Encoder Output                            │     │
│  └────────────────────────────────────────────┘     │
│                    ↓ 传递给Decoder                   │
│  ┌────────────────────────────────────────────┐     │
│  │           Decoder (6层)                    │     │
│  │  Target Embedding + Positional Encoding    │     │
│  │         ↓                                  │     │
│  │  [Decoder Block × 6]                       │     │
│  │         ↓                                  │     │
│  │  Decoder Output                            │     │
│  └────────────────────────────────────────────┘     │
│                    ↓                                 │
│  ┌────────────────────────────────────────────┐     │
│  │         Output Projection                  │     │
│  │    Linear + Softmax → 词汇表概率分布        │     │
│  └────────────────────────────────────────────┘     │
│                    ↓                                 │
│  输出序列: "public class UserService { ..."         │
└──────────────────────────────────────────────────────┘
```

### 9.2 数据流详细追踪

**任务**:将自然语言需求转换为Java代码

**输入**:`"创建一个用户服务类"`  
**期望输出**:`"public class UserService {"`

#### 步骤1:Tokenization
```python
source_ids = [BOS, 105, 23, 89, 234, 45, EOS]
target_ids = [BOS, 15, 23, 89, 234, 67, EOS]
```

#### 步骤2:Embedding + Positional Encoding
```python
encoder_input = source_embeddings + source_positions
decoder_input = target_embeddings + target_positions
```

#### 步骤3:Encoder前向传播
```python
x = encoder_input
for layer in encoder_layers:
    attn_out = multi_head_attention(x, x, x)
    x = layernorm(x + attn_out)
    ffn_out = feed_forward(x)
    x = layernorm(x + ffn_out)

encoder_output = x
```

#### 步骤4:Decoder自回归生成
```python
decoder_input = [BOS]
generated_tokens = []

for step in range(max_length):
    emb = embedding(decoder_input) + positional_encoding(len(decoder_input))
    dec_output = decoder(emb, encoder_output)
    
    last_hidden = dec_output[-1]
    logits = output_projection(last_hidden)
    probs = softmax(logits)
    
    next_token_id = argmax(probs)
    next_token = vocabulary[next_token_id]
    
    generated_tokens.append(next_token)
    decoder_input.append(next_token_id)
    
    if next_token == EOS:
        break
```

**每一步的详细过程**:
```
Step 1: 输入[BOS] → 生成 "public" (0.35)
Step 2: 输入[BOS, "public"] → 生成 "class" (0.42)
Step 3: 输入[BOS, "public", "class"] → 生成 "UserService" (0.28)
Step 4: 输入[BOS, "public", "class", "UserService"] → 生成 "{" (0.55)
Step 5: 输入[BOS, "public", "class", "UserService", "{"] → 生成 EOS (0.65)

停止!最终输出: ['public', 'class', 'User', 'Service', '{']
```

### 9.3 项目中的实现

查看项目中的[transformer.py](file:///E:/Project/llm-codegen-demo/scripts/core/transformer.py):

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_heads=8, num_layers=6):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output_proj = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(self.hidden_size)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.hidden_size)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory)
        
        logits = self.output_proj(output)
        return logits
```

### 9.4 关键设计要点

#### 为什么Encoder和Decoder都有多层?

```
第1层:捕捉局部语法关系(词性、短语结构)
第2层:捕捉句法结构(主谓宾关系)
第3层:捕捉语义关系(实体、意图)
第4层:捕捉篇章结构(逻辑、连贯性)
第5-6层:捕捉抽象概念(风格、领域知识)

实验表明:6-12层是性价比最好的选择
```

#### 为什么使用残差连接?

```
没有残差连接:
  x → Layer1 → Layer2 → ... → Layer12 → output
  梯度需要从Layer12传回Layer1,经过12次乘法
  如果每层梯度<1,最终梯度会趋近于0

有残差连接:
  梯度可以通过"捷径"直接传回
  即使某层梯度很小,其他路径仍然畅通
```

#### 为什么需要LayerNorm?

```
没有LayerNorm:
  第1层输出均值=0.5, 方差=1.0
  第2层输出均值=1.2, 方差=2.5
  第3层输出均值=3.8, 方差=8.2
  ... 数值不稳定,训练困难

有LayerNorm:
  每层输出都被归一化为均值≈0, 方差≈1
  训练更稳定,可以使用更大的学习率
```

---

## 10. 第九章:自回归生成

### 10.1 什么是自回归?

**定义**:自回归是指模型在生成时,每一步都基于之前生成的内容来预测下一步。

**通俗类比**:
```
就像写作文:
- 写完第一句后,第二句要接着第一句的意思
- 写完第二段后,第三段要承接前两段
- 每一句话都依赖于前面的所有内容
```

### 10.2 自回归 vs 非自回归

| 特性 | 自回归(AR) | 非自回归(NAR) |
|------|-----------|---------------|
| **生成方式** | 逐个token | 并行生成所有token |
| **速度** | 慢(串行) | 快(并行) |
| **质量** | 高(考虑上下文) | 较低(独立生成) |
| **应用场景** | GPT、翻译、对话 | 快速摘要 |

### 10.3 自回归生成的实现

```python
def generate_autoregressive(model, prompt, max_length=100):
    tokens = tokenize(prompt)
    token_ids = encode(tokens)
    
    for _ in range(max_length):
        input_tensor = torch.tensor([token_ids])
        
        with torch.no_grad():
            logits = model(input_tensor)
        
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        
        token_ids.append(next_token_id)
        
        if next_token_id == EOS_TOKEN:
            break
    
    return decode(token_ids)
```

### 10.4 自回归生成的挑战

#### 速度慢
```
生成100个token的代码:
  - 需要100次模型推理
  - 每次推理都要处理整个历史序列
  - 如果序列长度是1000,总计算量 = 100 × 1000 = 100,000次attention计算

解决方案:KV Cache(见第11章)
```

#### 错误累积
```
如果第2步错误:
  Step 1: "public" ✓
  Step 2: "interface" ✗ (应该是"class")
  Step 3: 基于"public interface"继续生成...
  最终: public interface UserService { ... }
  
  虽然语法正确,但语义错误(接口vs类)
```

**缓解方法**:
- ✅ Beam Search(束搜索):同时维护多个候选序列
- ✅ Sampling with Temperature:增加多样性
- ✅ Reinforcement Learning:通过奖励机制纠正错误

---

## 11. 第十章:采样策略

### 11.1 为什么需要采样?

**问题**:如果总是选择概率最高的token(贪心解码),会有什么问题?

```
贪心解码:
  "从前有一个" → "人" → "他" → "住" → "在" → "一个" → "小" → "村庄"
  输出: "从前有一个人他住在一个小村庄"
  
  问题:单调乏味,缺乏创意

采样解码:
  "从前有一个" → "国王" (概率0.15,被选中)
  "从前有一个国王" → "他" → "统治" → ...
  输出: "从前有一个国王他统治着一个遥远的王国"
  
  优势:更有创意,多样化,更像人类写作
```

### 11.2 Temperature(温度)

**Temperature** 控制概率分布的"平滑程度"。

```python
logits = [2.0, 1.0, 0.5, -0.5]

# Temperature = 1.0(标准)
probs = softmax(logits / 1.0) = [0.50, 0.31, 0.12, 0.07]

# Temperature = 0.2(低温度,更确定)
probs = softmax(logits / 0.2) = [0.86, 0.11, 0.02, 0.01]

# Temperature = 2.0(高温度,更随机)
probs = softmax(logits / 2.0) = [0.35, 0.26, 0.21, 0.18]
```

**可视化的影响**:
```
T=0.1 (非常确定):          T=1.0 (标准):
  ████                        ██
  █                           █
  █                           █
  A B C D                     A B C D

T=2.0 (较随机):             T=5.0 (非常随机):
  ██                          █
  ██                          █
  A B C D                     A B C D
```

**实际应用建议**:

| Temperature | 适用场景 | 特点 |
|------------|---------|------|
| **0.1-0.3** | 代码生成、数学计算 | 确定性高,错误少 |
| **0.4-0.7** | 技术文档、API调用 | 平衡准确性和多样性 |
| **0.8-1.2** | 创意写作、对话 | 有创意但不离谱 |
| **1.3-2.0** | 诗歌、头脑风暴 | 高度创意,可能不合理 |

### 11.3 Top-K Sampling

**Top-K** 只从概率最高的K个token中采样,忽略长尾的低概率token。

```python
# Top-K = 50
# 只保留概率最高的50个token
top_50_tokens = ["the", "a", "an", "hello", ..., "world"]
top_50_probs = [0.15, 0.12, 0.08, 0.05, ..., 0.001]

# 重新归一化概率
renormalized_probs = top_50_probs / sum(top_50_probs)

# 从这50个中采样
next_token = sample(top_50_tokens, renormalized_probs)
```

**为什么需要Top-K?**
```
Prompt: "Python是一种编程"

完整的概率分布:
  "语言": 0.35
  "框架": 0.15
  "工具": 0.10
  ...
  "苹果": 0.0001  ← 不合理
  "飞机": 0.00005 ← 更不合理

使用Top-K=10:
  只从前10个合理选项中选择
  完全排除了"苹果"、"飞机"等荒谬选项
```

### 11.4 Top-P Sampling(核采样)

**Top-P** 选择累积概率达到阈值P的最小token集合。

```python
# 排序后的概率分布
tokens = ["A", "B", "C", "D", "E", "F", ...]
probs = [0.30, 0.25, 0.20, 0.10, 0.08, 0.05, ...]

# Top-P = 0.9
# 累积概率:
#   A: 0.30
#   A+B: 0.55
#   A+B+C: 0.75
#   A+B+C+D: 0.85
#   A+B+C+D+E: 0.93 ≥ 0.9 ✓

# 选择前5个token: ["A", "B", "C", "D", "E"]
```

**Top-P vs Top-K**:

| 特性 | Top-K | Top-P |
|------|-------|-------|
| **候选数量** | 固定K个 | 动态变化 |
| **适应性** | 不考虑分布形状 | 根据分布自适应 |
| **优点** | 简单直观 | 更灵活 |

### 11.5 综合示例

```python
def advanced_sampling(logits, temperature=1.0, top_k=50, top_p=0.9):
    # 1. 应用温度
    logits = logits / temperature
    
    # 2. Top-K过滤
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # 3. 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # 4. Top-P过滤
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        probs[indices_to_remove] = 0
    
    # 5. 重新归一化
    probs = probs / probs.sum()
    
    # 6. 采样
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token
```

### 11.6 实战建议

**代码生成推荐配置**:
```python
config = {
    "temperature": 0.2,
    "top_k": 20,
    "top_p": 0.95
}
```

**创意写作推荐配置**:
```python
config = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.9
}
```

---

*继续阅读第三部分文档了解KV Cache优化和后处理技术...*
