# 快速启动指南

## ⚡ 5分钟快速开始

### 1. 安装依赖（约2-3分钟）

```bash
cd llm-codegen-demo
pip install torch numpy matplotlib seaborn
```

> **提示**: PyTorch较大，首次安装可能需要几分钟

### 2. 运行测试（约1分钟）

```bash
python test_all.py
```

预期输出：
```
总计: 7/7 测试通过
🎉 所有测试通过！系统工作正常。
```

### 3. 运行演示（交互式）

```bash
python main.py
```

选择选项 `6` 进入交互模式，然后输入：
```
>>> public class UserService
```

观察代码生成过程！

## 🎯 核心概念速览

### Token化是什么？
```
"public class User" → [15, 23, 156]
   文本              数字ID序列
```

### 注意力机制做什么？
```
每个token关注其他相关token：
"class" 关注 → "public" (0.45), "User" (0.35)
```

### 生成过程如何进行？
```
Step 0: "public class" 
Step 1: 预测 "User" (概率0.65)
Step 2: 预测 "{" (概率0.78)
...
直到遇到 <EOS> 或达到最大长度
```

## 🔍 快速调试

### 查看Token化细节
```python
from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()
ids, mask = tokenizer.encode("public class")
print(f"IDs: {ids}")
print(f"Mask: {mask}")
```

### 测试不同采样策略
```python
from generator import TemperatureSampling, TopKSampling

# 更确定
strategy1 = TemperatureSampling(temperature=0.3)

# 更多样
strategy2 = TemperatureSampling(temperature=1.0)

# 平衡
strategy3 = TopKSampling(top_k=50, temperature=0.7)
```

### 可视化注意力
```python
from visualizer import AttentionVisualizer

visualizer = AttentionVisualizer()
visualizer.visualize_model_architecture()
```

## 📊 关键参数调优

### 提高生成质量
```python
pipeline = CodeGenerationPipeline(
    d_model=256,          # 增加维度（默认128）
    nhead=16,             # 增加头数（默认8）
    num_encoder_layers=4, # 增加层数（默认2）
    num_decoder_layers=4
)
```

### 控制生成多样性
```python
# 更确定性（适合代码生成）
result = pipeline.generate(prompt, temperature=0.5)

# 更多样性（适合创意写作）
result = pipeline.generate(prompt, temperature=1.2)
```

### 加速生成
```python
# 使用GPU（如果有）
pipeline = CodeGenerationPipeline(device='cuda')

# 减少生成长度
result = pipeline.generate(prompt, max_length=50)

# 启用缓存
result = pipeline.generate(prompt, use_cache=True)
```

## 🐛 常见问题快速解决

### 问题1: ModuleNotFoundError: No module named 'torch'
**解决**: 
```bash
pip install torch
```

### 问题2: 生成结果全是<UNK>
**原因**: 词汇表太小  
**解决**: 
```python
tokenizer = SimpleTokenizer(vocab_size=2000)  # 增大词汇表
```

### 问题3: 生成速度太慢
**解决**:
```python
# 1. 减小模型
pipeline = CodeGenerationPipeline(
    num_encoder_layers=1,
    num_decoder_layers=1
)

# 2. 使用GPU
pipeline = CodeGenerationPipeline(device='cuda')
```

### 问题4: 内存不足
**解决**:
```python
# 减小序列长度
result = pipeline.generate(prompt, max_length=50)

# 减小batch size（当前实现batch_size=1）
```

## 📖 深入学习路径

### 第1天：理解基础
1. ✅ 运行 `test_all.py`
2. ✅ 阅读 `tokenizer.py`
3. ✅ 尝试交互模式

### 第2天：深入模型
1. ✅ 学习 `attention.py`
2. ✅ 理解 `transformer.py`
3. ✅ 查看可视化

### 第3天：实验和优化
1. ✅ 调整采样参数
2. ✅ 修改模型配置
3. ✅ 分析生成结果

## 💻 实用命令

```bash
# 运行完整测试
python test_all.py

# 运行主程序
python main.py

# 单独测试模块
python tokenizer.py
python attention.py
python transformer.py

# 查看文件结构
ls -la

# 查看帮助
cat README.md
cat GUIDE.md
```

## 🎓 关键知识点

### 必须理解的概念
1. **Token化**: 文本→数字
2. **Embedding**: 数字→向量
3. **位置编码**: 添加顺序信息
4. **Self-Attention**: token间关系
5. **Multi-Head**: 多角度关注
6. **Encoder-Decoder**: 编解码架构
7. **Sampling**: 从概率中选择
8. **Temperature**: 控制随机性

### 重要公式
```
Attention(Q,K,V) = softmax(Q@K^T/√d_k)@V

Positional Encoding:
  PE(pos, 2i) = sin(pos/10000^(2i/d))
  PE(pos, 2i+1) = cos(pos/10000^(2i/d))
```

## 🚀 下一步

1. **阅读文档**: `README.md`, `GUIDE.md`
2. **查看源码**: 每个文件都有详细注释
3. **动手实验**: 修改参数观察效果
4. **扩展功能**: 添加新的采样策略或后处理规则

---

**有问题？查看详细文档或源代码注释！**
