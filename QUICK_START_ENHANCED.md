# 增强功能快速开始

## 🚀 5分钟上手

### 1. 体验概率分析 (2分钟)

```python
from pipeline import CodeGenerationPipeline

# 创建管道
pipeline = CodeGenerationPipeline(
    vocab_size=1000, d_model=128, nhead=8,
    num_encoder_layers=2, num_decoder_layers=2, device='cpu'
)

# 生成代码并启用概率分析
result = pipeline.generate(
    prompt="public class UserService",
    max_length=15,
    enable_probability_analysis=True,  # ← 关键参数
    verbose=True
)

# 查看分析结果
print(f"\n平均熵: {sum(a['entropy'] for a in result['probability_analyses']) / len(result['probability_analyses']):.4f}")
```

**你会看到**:
```
[Probability Analysis] Step 0: Selected 'class'
  Entropy (uncertainty): 3.2145
  Token rank: #0
  Selected probability: 0.0847
  Top-5 cumulative: 0.2134
  Gini coefficient: 0.3421
```

---

### 2. 体验注意力追踪 (2分钟)

```python
# 创建管道并启用注意力追踪
pipeline = CodeGenerationPipeline(
    vocab_size=1000, d_model=128, nhead=8,
    num_encoder_layers=2, num_decoder_layers=2, device='cpu',
    enable_attention_tracking=True  # ← 关键参数
)

# 生成代码
result = pipeline.generate(
    prompt="public class OrderService",
    max_length=10,
    verbose=False
)

# 查看逐层对比
tracker = pipeline.model.get_attention_tracker()
tracker.print_layer_comparison(
    num_encoder_layers=2,
    num_decoder_layers=2
)
```

**你会看到**:
```
--- Encoder Layers ---
Layer 1: Attention entropy: 1.5319 (More focused)
Layer 2: Attention entropy: 1.5394 (More focused)

--- Decoder Layers ---
Layer 1:
  Self-Attention: Entropy: 1.4311 (focused)
  Cross-Attention: Entropy: 3.4240 (distributed)
```

---

### 3. 运行完整演示 (1分钟)

```bash
# 运行测试脚本
python test_enhanced_features.py
```

这会展示:
- ProbabilityAnalyzer独立测试
- 集成概率分析示例
- 注意力追踪可视化

---

## 🎯 核心概念速查

### 熵 (Entropy)
- **含义**: 模型的不确定性
- **范围**: 0 - ln(vocab_size) ≈ 6.9 (对于vocab=1000)
- **解读**:
  - < 2.0: 非常确定
  - 2.0-4.0: 适度不确定
  - > 4.0: 高度不确定

### Token排名 (Rank)
- **含义**: 选中token在概率排序中的位置
- **解读**:
  - #0: 选择了最可能的token（保守）
  - #1-5: 选择了次优token（平衡）
  - #5+: 选择了低概率token（创造性）

### Gini系数
- **含义**: 概率分布的集中度
- **范围**: -1到1（理想情况0到1）
- **解读**:
  - > 0.5: 概率集中（少数token主导）
  - 0.2-0.5: 适度分散
  - < 0.2: 非常均匀（模型犹豫）

### 注意力熵
- **含义**: 注意力权重的专注度
- **解读**:
  - < 1.5: 专注（关注少数token）
  - 1.5-2.5: 适度分散
  - > 3.0: 广泛分散（参考多个token）

---

## 💡 立即尝试的小实验

### 实验1: Temperature对比
```python
for temp in [0.3, 0.7, 1.2]:
    result = pipeline.generate(
        prompt="public class Test",
        temperature=temp,
        enable_probability_analysis=True,
        verbose=False
    )
    avg_entropy = sum(a['entropy'] for a in result['probability_analyses']) / len(result['probability_analyses'])
    print(f"Temp {temp}: Avg Entropy = {avg_entropy:.4f}")
```

### 实验2: 观察熵值趋势
```python
result = pipeline.generate(
    prompt="public class LongTest",
    max_length=30,
    enable_probability_analysis=True,
    verbose=False
)

entropies = [a['entropy'] for a in result['probability_analyses']]
print("熵值演变:", [f"{e:.2f}" for e in entropies])
# 通常会看到熵值随步数增加而上升
```

### 实验3: 注意力模式对比
```python
# 对比Encoder和Decoder的注意力
tracker.print_layer_comparison(
    num_encoder_layers=2,
    num_decoder_layers=2
)
# 思考：为什么Cross-Attention通常更分散？
```

---

## 📖 深入学习

- **详细文档**: `ENHANCED_FEATURES_GUIDE.md`
- **优化总结**: `OPTIMIZATION_SUMMARY.md`
- **源代码**:
  - `generator.py` - ProbabilityAnalyzer类
  - `transformer.py` - AttentionTracker类

---

## ❓ 常见问题

**Q: 启用这些功能会影响性能吗？**
A: 轻微影响。概率分析增加~5-10%计算时间，注意力追踪增加~20-30MB内存。学习时启用，生产时禁用。

**Q: 为什么有些指标看起来异常？**
A: 这是未训练模型的正常现象。训练后的模型会有更合理的分布。重点在于理解指标的相对变化。

**Q: 可以同时启用两个功能吗？**
A: 可以！它们互不干扰：
```python
pipeline = CodeGenerationPipeline(..., enable_attention_tracking=True)
result = pipeline.generate(..., enable_probability_analysis=True)
```

---

## 🎉 下一步

1. ✅ 完成上面的3个快速实验
2. 📖 阅读 `ENHANCED_FEATURES_GUIDE.md` 深入了解
3. 🔬 尝试设计自己的实验
4. 💭 思考：这些指标如何帮助你理解LLM的工作原理？

祝你学习愉快！
