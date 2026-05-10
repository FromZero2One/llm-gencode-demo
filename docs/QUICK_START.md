# 🚀 快速开始指南

## 1. 安装与验证

```bash
# 安装依赖
pip install torch numpy matplotlib seaborn

# 验证安装
python scripts/tests/test_all.py
```

## 2. 运行演示

### 交互式主菜单（推荐）

```bash
python scripts/main.py
```

**可用选项**:
- `1` - 基本代码生成流程
- `2` - 不同采样策略对比
- `3` - 缓存机制演示
- `4` - 多样本生成
- `5` - 可视化功能
- `6` - 交互模式（自己输入prompt）
- `7` - 运行所有演示

### 直接运行特定演示

```bash
# 基本代码生成流程
echo "1" | python scripts/main.py

# 采样策略对比
echo "2" | python scripts/main.py

# 缓存机制演示
echo "3" | python scripts/main.py
```

## 3. Tokenizer调试工具

```bash
# 完整分析
python scripts/tests/debug_tokenizer.py

# 交互式调试
python scripts/tests/tokenizer_interactive.py
```

## 4. 性能测试

```bash
python scripts/tests/test_performance.py
```

## 5. 使用统一入口脚本

```bash
python run_demo.py
```

提供菜单式选择，包括：
- 运行完整测试
- 运行主程序演示
- 运行性能测试
- 运行Tokenizer调试
- 运行交互式Tokenizer

## 6. 第一个实验

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
    verbose=True
)

print(result['processed_code'])
```

## 7. 常用命令速查

```bash
# 运行所有测试
python scripts/tests/test_all.py

# 运行主程序
python scripts/main.py

# Tokenizer调试
python scripts/tests/debug_tokenizer.py

# 交互式Tokenizer
python scripts/tests/tokenizer_interactive.py

# 性能测试
python scripts/tests/test_performance.py

# 新功能测试
python scripts/tests/test_new_features.py

# 数学公式验证
python scripts/tests/verify_math.py

# 训练和KV Cache演示
python scripts/test_training_and_cache.py
```

## 8. 下一步

- 📖 阅读 [README.md](README.md) 了解详细用法
- 🔍 查看 [ADVANCED_GUIDE.md](ADVANCED_GUIDE.md) 学习高级功能
- 🛠️ 使用 [DEBUG_GUIDE.md](DEBUG_GUIDE.md) 进行调试
- 📋 查看 [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) 了解项目计划
