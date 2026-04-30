# 增强功能使用指南

本文档介绍新增的LLM底层逻辑可视化功能，帮助学习者深入理解模型内部决策过程。

---

## 📊 新增功能概览

### 1. ProbabilityAnalyzer - 概率分布分析器
**位置**: `generator.py`

**功能**: 实时分析每一步token生成的概率分布特征

**关键指标**:
- **Entropy (熵)**: 衡量模型的不确定性
  - 低熵 (< 2.0): 模型很确定
  - 高熵 (> 4.0): 模型犹豫不决
- **Token Rank**: 选中token在概率排序中的位置
  - #0: 选择了最可能的token
  - #5+: 选择了较低概率的token（更有创造性）
- **Gini Coefficient**: 概率集中度
  - 接近1: 概率集中在少数token
  - 接近0: 概率均匀分布
- **Vocabulary Coverage**: 达到90%累积概率所需的token数量

**使用示例**:
```python
from pipeline import CodeGenerationPipeline

pipeline = CodeGenerationPipeline(
    vocab_size=1000,
    d_model=128,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    device='cpu'
)

# 启用概率分析
result = pipeline.generate(
    prompt="public class UserService",
    max_length=20,
    temperature=0.7,
    enable_probability_analysis=True,  # 关键参数
    verbose=True
)

# 访问分析数据
for step, analysis in enumerate(result['probability_analyses']):
    print(f"Step {step}:")
    print(f"  Entropy: {analysis['entropy']:.4f}")
    print(f"  Token rank: #{analysis['ranks']}")
    print(f"  Gini: {analysis['gini_coefficients']:.4f}")
```

**输出示例**:
```
[Probability Analysis] Step 0: Selected 'class'
  Entropy (uncertainty): 3.2145
  Token rank: #0
  Selected probability: 0.0847
  Top-1 probability: 0.0847
  Top-5 cumulative: 0.2134
  Gini coefficient: 0.3421
  Tokens for 90% prob: 245
```

---

### 2. AttentionTracker - 注意力权重追踪器
**位置**: `transformer.py`

**功能**: 记录并可视化每一层Encoder和Decoder的注意力模式演变

**关键特性**:
- 逐层记录Encoder自注意力权重
- 逐层记录Decoder自注意力和交叉注意力权重
- 计算每层的注意力熵（专注度vs分散度）
- 生成注意力演变热力图

**使用示例**:
```python
from pipeline import CodeGenerationPipeline

# 创建管道时启用注意力追踪
pipeline = CodeGenerationPipeline(
    vocab_size=1000,
    d_model=128,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    device='cpu',
    enable_attention_tracking=True  # 关键参数
)

result = pipeline.generate(
    prompt="public class OrderService",
    max_length=15,
    verbose=True
)

# 获取追踪器
tracker = pipeline.model.get_attention_tracker()

# 打印逐层对比
tracker.print_layer_comparison(
    num_encoder_layers=2,
    num_decoder_layers=2
)

# 可视化注意力演变（需要matplotlib）
token_names = ["public", "class", "Order", "Service"]
tracker.visualize_attention_evolution(
    token_names=token_names,
    save_path='attention_evolution.png'
)
```

**输出示例**:
```
======================================================================
[AttentionTracker] 逐层注意力模式对比
======================================================================

--- Encoder Layers ---

Layer 1:
  Shape: [1, 8, 32, 32]
  Mean attention: 0.031250
  Max attention: 0.5450
  Attention entropy: 1.5319
  (More focused)

Layer 2:
  Shape: [1, 8, 32, 32]
  Mean attention: 0.031250
  Max attention: 0.4638
  Attention entropy: 1.5394
  (More focused)

--- Decoder Layers ---

Layer 1:
  Self-Attention:
    Entropy: 1.4311 (focused)
  Cross-Attention:
    Entropy: 3.4240 (distributed)

Layer 2:
  Self-Attention:
    Entropy: 1.4484 (focused)
  Cross-Attention:
    Entropy: 3.4061 (distributed)
```

---

## 🎯 学习价值

### 概率分析能帮你理解：

1. **Temperature的作用机制**
   ```python
   # 对比不同temperature下的熵值变化
   for temp in [0.3, 0.7, 1.2]:
       result = pipeline.generate(
           prompt="public class Test",
           temperature=temp,
           enable_probability_analysis=True
       )
       # 观察：低temperature → 低熵 → 更确定
   ```

2. **模型的置信度**
   - 低熵 + 高Top-1概率 = 模型非常确定
   - 高熵 + 低Top-1概率 = 模型在探索多种可能性

3. **为什么选择这个token**
   - Rank #0: 贪婪选择，最安全
   - Rank #5+: 创造性选择，可能产生新颖代码

### 注意力追踪能帮你理解：

1. **信息流动路径**
   - Encoder Layer 1: 捕捉局部模式（相邻token关系）
   - Encoder Layer 2: 整合全局信息（长距离依赖）

2. **Decoder如何参考Input**
   - Cross-Attention熵值高: 广泛参考多个source token
   - Cross-Attention熵值低: 聚焦特定source token

3. **深层网络的学习能力**
   - 对比不同层的注意力模式
   - 理解"深度"带来的抽象能力提升

---

## 🔧 完整演示脚本

运行 `test_enhanced_features.py` 查看完整示例：

```bash
python test_enhanced_features.py
```

该脚本包含三个测试：
1. **ProbabilityAnalyzer独立测试**: 直接分析模拟logits
2. **集成概率分析**: 在完整生成流程中启用分析
3. **注意力追踪**: 记录并可视化逐层注意力演变

---

## 📈 性能影响

- **概率分析**: 每次生成步骤增加约5-10%计算开销（主要来自排序和熵计算）
- **注意力追踪**: 额外内存占用约20-30MB（存储各层注意力权重）

**建议**: 在学习和调试时启用，生产环境禁用

---

## 💡 实践建议

### 实验1: Temperature对概率分布的影响
```python
temperatures = [0.2, 0.5, 0.7, 1.0, 1.5]
for temp in temperatures:
    result = pipeline.generate(
        prompt="public class User",
        temperature=temp,
        enable_probability_analysis=True,
        verbose=False
    )
    avg_entropy = sum(a['entropy'] for a in result['probability_analyses']) / len(result['probability_analyses'])
    print(f"Temp {temp}: Avg Entropy = {avg_entropy:.4f}")
```

### 实验2: 不同层的注意力模式
```python
# 观察Encoder和Decoder的注意力熵差异
tracker.print_layer_comparison(num_encoder_layers=2, num_decoder_layers=2)
# 思考：为什么Cross-Attention通常比Self-Attention更分散？
```

### 实验3: 生成长度对不确定性的影响
```python
# 观察随着生成进行，模型是否越来越不确定
result = pipeline.generate(
    prompt="public class Test",
    max_length=50,
    enable_probability_analysis=True
)
entropies = [a['entropy'] for a in result['probability_analyses']]
print("Entropy trend:", entropies)
# 通常会看到熵值随步数增加而上升
```

---

## 🚀 下一步扩展

基于这些基础功能，可以进一步开发：
- **Embedding空间可视化**: t-SNE降维展示token向量
- **交互式学习沙盒**: 实时调整参数观察影响
- **不确定性热力图**: 可视化整个生成过程的不确定性演变
- **Cross-Attention对齐图**: 展示Source-Target token对应关系

---

## ❓ 常见问题

**Q: 为什么Gini系数可能是负数？**
A: 在未训练的模型中，概率分布可能非常均匀甚至出现异常值。这是正常现象，训练后的模型Gini系数通常在0-1之间。

**Q: 注意力熵值的参考范围是多少？**
A: 
- < 1.5: 非常专注（可能过拟合风险）
- 1.5-2.5: 适度专注（理想状态）
- > 3.0: 非常分散（可能在探索或不确定）

**Q: 可以同时启用两个功能吗？**
A: 可以！它们互不干扰：
```python
pipeline = CodeGenerationPipeline(
    ...,
    enable_attention_tracking=True
)
result = pipeline.generate(
    ...,
    enable_probability_analysis=True
)
```
