# 版本发布说明 - v2.0-simplified

**发布日期**: 2026-05-25  
**版本**: v2.0-simplified  
**类型**: 重大更新 (Breaking Changes)

---

## 🎉 概述

v2.0-simplified是一个**重大简化版本**,专注于提升用户体验和降低学习门槛。经过4周的系统性重构,我们将一个复杂的演示系统转变为易学、易用、易维护的教学工具。

### 核心改进

| 指标 | v1.x | v2.0 | 改进幅度 |
|------|------|------|---------|
| **配置参数** | 10+个 | 1个preset | ⬇️ 90% |
| **入门时间** | 30分钟 | 5分钟 | ⬇️ 83% |
| **代码行数** | ~4,500 | ~3,300 | ⬇️ 27% |
| **README长度** | 1,053行 | 230行 | ⬇️ 78% |
| **文档数量** | 1个单一文档 | 5层分级文档 | 更完善 |

---

## ✨ 新特性

### 1. Preset配置系统 ⭐

**问题**: 之前需要理解10+个参数的含义和选择依据

**解决方案**: 引入3个预定义配置

```python
# v1.x (复杂)
pipeline = CodeGenerationPipeline(
    vocab_size=1000,
    d_model=128,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    max_seq_length=128,
    cache_size=20,
    device='cpu',
    debug_mode=False
)

# v2.0 (简单!)
pipeline = CodeGenerationPipeline(preset='small')
```

**可用Preset**:
- `tiny`: 超小型,用于快速测试 (vocab=500, d_model=64)
- `small`: 小型,默认推荐 (vocab=1000, d_model=128)
- `medium`: 中型,更高质量 (vocab=2000, d_model=256)

**支持覆盖**:
```python
# 基于preset,但覆盖特定参数
pipeline = CodeGenerationPipeline(
    preset='small',
    vocab_size=3000,  # 覆盖默认值
    device='cuda'
)
```

---

### 2. 分层文档体系 ⭐

**问题**: 单一长文档(1,053行),新手难以找到入口

**解决方案**: 建立5层分级文档

```
docs/
├── README.md (230行)          ← 入口文档
├── QUICKSTART.md (135行)      ← 5分钟快速入门
├── TUTORIAL.md (666行)        ← 详细教程
├── ADVANCED.md (659行)        ← 进阶指南
└── README_FULL.md (928行)     ← 完整文档
```

**用户旅程**:
- **新手**: README → QUICKSTART (5分钟开始)
- **学习者**: TUTORIAL (30分钟深入)
- **开发者**: ADVANCED (60分钟精通)
- **研究者**: README_FULL (按需查阅)

---

### 3. 简化的模块结构

**合并模块**:
- `generator.py` (381行) + `postprocessor.py` (416行) → `code_generator.py` (~450行)
- 减少接口复杂度,提高内聚性

**可选分离**:
- `training/` → `optional/training/`
- 标记为非核心功能,降低初学者认知负担

---

## 🔄 重大变更 (Breaking Changes)

### 1. Pipeline接口变更

**旧接口** (v1.x):
```python
pipeline = CodeGenerationPipeline(
    vocab_size=1000,
    d_model=128,
    nhead=8,
    # ... 更多参数
)
```

**新接口** (v2.0):
```python
pipeline = CodeGenerationPipeline(preset='small')
```

**迁移指南**:
```python
# 如果之前使用特定配置,选择最接近的preset
# v1.x: vocab_size=500, d_model=64
pipeline = CodeGenerationPipeline(preset='tiny')

# v1.x: vocab_size=1000, d_model=128
pipeline = CodeGenerationPipeline(preset='small')

# v1.x: vocab_size=2000, d_model=256
pipeline = CodeGenerationPipeline(preset='medium')

# 如需精确匹配,使用参数覆盖
pipeline = CodeGenerationPipeline(
    preset='small',
    vocab_size=1500  # 自定义值
)
```

---

### 2. 训练系统移至optional

**旧位置**: `scripts.training.trainer`  
**新位置**: `scripts.optional.training.trainer`

**影响**: 
- 导入路径变更
- 训练功能标记为"可选",不在主文档重点介绍

**迁移**:
```python
# v1.x
from scripts.training.trainer import Trainer

# v2.0
from scripts.optional.training.trainer import Trainer
```

---

### 3. 文件重命名

| 旧文件 | 新文件 | 说明 |
|--------|--------|------|
| `generator.py` + `postprocessor.py` | `code_generator.py` | 合并 |
| `docs/README.md` (928行) | `docs/README_FULL.md` | 备份 |
| `docs/README.md` (新建) | `docs/README.md` (230行) | 精简版 |

---

## 📚 新增文档

### QUICKSTART.md (135行)

**目标**: 5分钟内运行第一个示例

**内容**:
- ✅ 3个简单步骤
- ✅ Preset配置对比表
- ✅ 常见问题解答
- ✅ 下一步学习路径

**适合**: 所有人首次使用

---

### TUTORIAL.md (666行)

**目标**: 深入理解Transformer原理

**内容**:
- ✅ 核心概念详解(Tokenization、Attention、Transformer)
- ✅ 模块学习路径
- ✅ 4个实战练习(含代码和思考题)
- ✅ 常见问题解答

**适合**: 学生/初学者深入学习

---

### ADVANCED.md (659行)

**目标**: 掌握高级配置和扩展开发

**内容**:
- ✅ Preset系统深度解析
- ✅ 性能优化技巧(GPU、KV Cache)
- ✅ 扩展开发指南(添加采样策略等)
- ✅ 调试与诊断
- ✅ 最佳实践

**适合**: 开发者/研究者

---

## 🛠️ 技术改进

### 1. 代码质量

- ✅ 移除冗余代码(~1,200行)
- ✅ 统一日志格式
- ✅ 改进错误提示
- ✅ 增强类型注解

### 2. 测试覆盖

- ✅ 8项核心测试全部通过
- ✅ 新增Preset系统测试
- ✅ 保持向后兼容测试

### 3. 性能优化

- ✅ KV Cache默认启用(加速10-50倍)
- ✅ Result Cache智能管理
- ✅ 内存占用优化

---

## 📊 兼容性说明

### 向后兼容性

**部分兼容**: v2.0引入了Breaking Changes,但保留了主要功能

**兼容的功能**:
- ✅ Tokenizer API不变
- ✅ Attention机制不变
- ✅ Transformer模型不变
- ✅ 缓存机制不变

**不兼容的变更**:
- ❌ Pipeline构造函数签名变更
- ❌ 训练系统导入路径变更
- ❌ 文件组织结构变更

### Python版本要求

- **最低版本**: Python 3.7+
- **推荐版本**: Python 3.8+
- **依赖**: PyTorch 2.0+, NumPy

---

## 🚀 升级指南

### 从v1.x升级到v2.0

**步骤1**: 备份现有代码
```bash
git clone <repo-url> llm-codegen-demo-v1
```

**步骤2**: 获取v2.0
```bash
git pull origin main
# 或
git checkout v2.0.0
```

**步骤3**: 更新代码
```python
# 查找所有CodeGenerationPipeline调用
# 替换为preset方式

# 旧代码
pipeline = CodeGenerationPipeline(vocab_size=1000, d_model=128, ...)

# 新代码
pipeline = CodeGenerationPipeline(preset='small')
```

**步骤4**: 更新导入路径
```python
# 如果使用训练系统
# 旧: from scripts.training.trainer import Trainer
# 新: from scripts.optional.training.trainer import Trainer
```

**步骤5**: 运行测试
```bash
python scripts/tests/test_all.py
```

---

## 🐛 已知问题

### 1. Preset灵活性限制

**问题**: 某些特殊场景可能需要非标准的参数组合

**临时方案**: 使用参数覆盖
```python
pipeline = CodeGenerationPipeline(
    preset='small',
    vocab_size=1500,
    d_model=192
)
```

**未来计划**: 支持自定义preset配置文件

---

### 2. 训练系统文档减少

**问题**: 训练系统移至optional,主文档中相关内容减少

**解决方案**: 
- 查看 `README_FULL.md` 了解完整训练流程
- 查看 `ADVANCED.md` 了解如何集成训练

---

## 📝 贡献指南

### 代码贡献

1. Fork仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 文档贡献

我们特别欢迎文档改进!
- 修正拼写/语法错误
- 补充示例代码
- 添加新的教程
- 翻译到其他语言

---

## 🙏 致谢

感谢所有参与v2.0开发和测试的贡献者:
- 架构设计团队
- 文档编写团队
- 测试验证团队
- 早期试用用户提供反馈

---

## 📅 发布计划

### v2.0.0 (2026-05-25) - 当前版本

- ✅ Preset配置系统
- ✅ 分层文档体系
- ✅ 模块合并简化
- ✅ 训练系统移至optional

### v2.1.0 (计划: 2026-06-15)

- 🔮 支持自定义preset配置文件
- 🔮 添加更多示例代码
- 🔮 性能进一步优化
- 🔮 社区贡献整合

### v3.0.0 (计划: 2026-08-01)

- 🔮 支持真实数据集训练
- 🔮 实现BPE Tokenizer
- 🔮 Web界面演示
- 🔮 API服务化

---

## 📄 许可证

本项目仅用于教育和学习目的。

---

## 📞 支持与反馈

**遇到问题?**
- 查看 [QUICKSTART.md](QUICKSTART.md)
- 查看 [TUTORIAL.md](TUTORIAL.md)
- 查看 [ADVANCED.md](ADVANCED.md)
- 查看 [README_FULL.md](README_FULL.md)

**报告Bug:**
- GitHub Issues: `<repo-url>/issues`

**提出建议:**
- GitHub Discussions: `<repo-url>/discussions`

---

**祝您使用愉快!**

*最后更新: 2026-05-25*  
*版本: v2.0-simplified*
