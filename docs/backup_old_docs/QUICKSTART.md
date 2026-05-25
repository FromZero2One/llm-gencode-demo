# 🚀 快速入门 (5分钟)

> **目标**: 在5分钟内运行第一个代码生成示例

---

## 📋 前置要求

- Python 3.7+
- PyTorch (已安装)

---

## ⚡ 快速开始

### 步骤1: 运行演示 (1分钟)

```bash
python scripts/main.py
```

选择选项 `1` 查看基本代码生成流程。

### 步骤2: 编写你的第一个脚本 (2分钟)

创建文件 `my_first_demo.py`:

```python
from scripts.pipeline import CodeGenerationPipeline

# 创建Pipeline (使用small预设)
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

运行:
```bash
python my_first_demo.py
```

### 步骤3: 尝试不同的预设 (2分钟)

```python
# Tiny preset - 最快,适合快速测试
pipeline_tiny = CodeGenerationPipeline(preset='tiny')

# Small preset - 平衡性能和速度 (默认)
pipeline_small = CodeGenerationPipeline(preset='small')

# Medium preset - 更强表达能力
pipeline_medium = CodeGenerationPipeline(preset='medium')
```

---

## 🎯 核心概念

### Preset配置系统

| Preset | 适用场景 | 参数量 | 速度 |
|--------|---------|--------|------|
| **tiny** | 快速测试、教学演示 | 247K | ⚡⚡⚡ 最快 |
| **small** | 常规实验、课堂演示 | 720K | ⚡⚡ 平衡 |
| **medium** | 深入研究、性能测试 | ~2M | ⚡ 更强 |

### 基本用法

```python
# 最简单的用法 - 只需1行!
pipeline = CodeGenerationPipeline(preset='small')

# 生成代码 - 只需2行!
result = pipeline.generate("public class Test")
print(result['code'])
```

---

## 📚 下一步

根据你的需求选择:

### 🎓 我是学生/初学者
→ 阅读 [详细教程](TUTORIAL.md) 了解Transformer原理

### 🔧 我是开发者/研究者
→ 阅读 [进阶指南](ADVANCED.md) 学习自定义配置

### 📖 我想了解项目全貌
→ 阅读 [完整文档](README.md)

---

## ❓ 常见问题

**Q: 为什么生成的代码看起来不真实?**  
A: 这是一个教学演示项目,模型是随机初始化的,没有经过训练。重点是理解LLM的工作原理,而非生成可用代码。

**Q: 如何加快生成速度?**  
A: 使用 `preset='tiny'` 或启用GPU: `CodeGenerationPipeline(preset='small', device='cuda')`

**Q: 可以自定义模型参数吗?**  
A: 可以!覆盖preset参数:
```python
pipeline = CodeGenerationPipeline(
    preset='small',
    d_model=256,  # 覆盖模型维度
    nhead=16      # 覆盖注意力头数
)
```

---

## 🎉 恭喜!

你已经完成了快速入门!现在你可以:
- ✅ 运行代码生成演示
- ✅ 使用不同的preset配置
- ✅ 继续深入学习

**建议**: 花10分钟阅读 [详细教程](TUTORIAL.md) 了解背后的原理。
