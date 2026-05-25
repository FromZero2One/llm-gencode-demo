# LLM代码生成演示项目

一个从零实现的完整LLM代码生成系统，深入展示大模型从输入到输出的每个步骤。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

---

## 📚 文档导航

**根据你的需求选择合适的文档:**

| 文档 | 适合人群 | 阅读时间 | 内容 |
|------|---------|---------|------|
| [🚀 快速入门](QUICKSTART.md) | 所有人 | 5分钟 | 快速运行第一个示例 |
| [📖 详细教程](TUTORIAL.md) | 学生/初学者 | 30分钟 | Transformer原理详解 |
| [🔧 进阶指南](ADVANCED.md) | 开发者/研究者 | 60分钟 | 自定义配置和扩展 |
| [📘 完整文档](README_FULL.md) | 需要全面了解 | 按需查阅 | 所有技术细节 |

> **建议**: 先阅读 [快速入门](QUICKSTART.md)，然后根据需要选择其他文档。

---

## 🎯 项目简介

这是一个**教育性质的LLM代码生成演示系统**，从零实现了Transformer架构的所有核心组件。

### 核心价值

- **透明化**: 每个步骤都可见，没有黑盒
- **模块化**: 8个核心模块，可单独学习
- **易用性**: Preset配置系统，1行代码即可开始
- **可扩展**: 易于添加新功能或修改现有逻辑

### 简化成果 (v2.0)

| 指标 | 简化前 | 简化后 | 改进 |
|------|--------|--------|------|
| 配置参数 | 10+个 | 1个preset | ⬇️ 90% |
| 入门时间 | 30分钟 | 5分钟 | ⬇️ 83% |
| 代码行数 | ~4,500 | ~3,300 | ⬇️ 27% |
| 文件数 | 11个 | 10个 | ⬇️ 9% |
| README长度 | 1,053行 | ~200行 | ⬇️ 81% |

---

## 🚀 快速开始

### 安装

```bash
pip install torch numpy matplotlib seaborn
```

### 运行演示

```bash
python scripts/main.py
```

### 最简示例

```python
from scripts.pipeline import CodeGenerationPipeline

# 只需2行！
pipeline = CodeGenerationPipeline(preset='small')
result = pipeline.generate("public class UserService")
print(result['processed_code'])
```

**详细教程**: 查看 [QUICKSTART.md](QUICKSTART.md)

---

## 📦 项目结构

```
llm-codegen-demo/
├── docs/                    # 文档目录
│   ├── README.md           # 本文档
│   ├── QUICKSTART.md       # 快速入门
│   ├── TUTORIAL.md         # 详细教程
│   ├── ADVANCED.md         # 进阶指南
│   └── README_FULL.md      # 完整文档
├── scripts/                # 核心代码
│   ├── main.py             # 主程序入口
│   ├── pipeline.py         # Pipeline整合
│   ├── core/               # 核心组件
│   │   ├── tokenizer.py    # Tokenizer
│   │   ├── attention.py    # Attention机制
│   │   └── transformer.py  # Transformer
│   ├── generation/         # 代码生成
│   │   └── code_generator.py  # 生成器+后处理器
│   ├── optimization/       # 性能优化
│   │   ├── cache.py        # 结果缓存
│   │   └── kv_cache.py     # KV Cache
│   ├── optional/           # 可选功能
│   │   └── training/       # 训练系统(可选)
│   ├── config/             # 配置系统
│   │   └── presets.py      # Preset定义
│   ├── utils/              # 工具模块
│   └── tests/              # 测试套件
└── logs/                   # 日志文件
```

---

## 🧪 测试

```bash
# 运行所有测试
python scripts/tests/test_all.py

# 预期输出:
# ✅ 所有8项测试全部通过
```

---

## 🔍 核心特性

### 1. Preset配置系统

```python
# 三种预设配置
tiny   # 超小型，用于快速测试
small  # 小型，默认推荐
medium # 中型，更高质量

# 使用示例
pipeline = CodeGenerationPipeline(preset='small')
```

### 2. 智能缓存

- **KV Cache**: 加速单次生成(10-50倍)
- **Result Cache**: 避免重复生成

### 3. 多种采样策略

- Greedy (贪婪解码)
- Temperature (温度采样)
- Top-K (前K个采样)
- Top-P (核采样)

### 4. 代码后处理

- 括号匹配检查
- 语法验证
- 代码格式化

---

## 📖 学习路径

### 新手入门

1. **第一步**: 阅读 [QUICKSTART.md](QUICKSTART.md) (5分钟)
2. **第二步**: 运行 `python scripts/main.py` 体验功能
3. **第三步**: 阅读 [TUTORIAL.md](TUTORIAL.md) 深入学习

### 进阶学习

1. **第四步**: 阅读 [ADVANCED.md](ADVANCED.md) 掌握高级技巧
2. **第五步**: 阅读源码，理解实现细节
3. **第六步**: 尝试扩展开发，添加新功能

### 深入研究

- 阅读 [README_FULL.md](README_FULL.md) 了解所有技术细节
- 研究Transformer相关论文
- 参与开源贡献

---

## ❓ 常见问题

### Q: 如何提高生成质量?

**短期方案**:
- 增大preset: `tiny → small → medium`
- 降低temperature: `0.7 → 0.5`

**长期方案**:
- 在真实数据上训练
- 使用BPE分词
- 增加模型规模

### Q: 可以用GPU吗?

```python
pipeline = CodeGenerationPipeline(preset='small', device='cuda')
```

加速效果: 5-10x

### Q: 为什么生成的代码全是`<UNK>`?

词汇表太小，使用更大的preset:
```python
pipeline = CodeGenerationPipeline(preset='medium')  # vocab_size=2000
```

---

## 🛠️ 技术栈

- **Python 3.7+**: 编程语言
- **PyTorch 2.0+**: 深度学习框架
- **NumPy**: 数值计算
- **Matplotlib/Seaborn**: 可视化(可选)

---

## 📄 许可证

本项目仅用于教育和学习目的。

---

## 🙏 致谢

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化讲解
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 代码详解

---

**祝您学习愉快！**

*最后更新: 2026-05-25 | 版本: v2.0-simplified*
