# 🔧 进阶指南 - 自定义配置与扩展开发

> **阅读时间**: 60分钟  
> **目标**: 掌握高级配置、性能优化和扩展开发技巧

---

## 📋 目录

1. [Preset系统深度解析](#1-preset系统深度解析)
2. [自定义配置](#2-自定义配置)
3. [性能优化技巧](#3-性能优化技巧)
4. [扩展开发](#4-扩展开发)
5. [调试与诊断](#5-调试与诊断)
6. [最佳实践](#6-最佳实践)

---

## 1. Preset系统深度解析

### 1.1 Preset设计原理

Preset系统的核心思想是:**预定义常用配置,提供简单接口**。

**优势**:
- ✅ 降低认知负担(1个参数 vs 10+个参数)
- ✅ 避免配置错误(经过验证的配置组合)
- ✅ 快速上手(无需理解每个参数的含义)
- ✅ 易于分享("使用small preset"比列出所有参数更清晰)

### 1.2 Preset结构详解

```python
PRESETS = {
    'tiny': {
        # 元数据(不会传递给Pipeline)
        'name': 'Tiny (超小型)',
        'description': '用于快速测试和理解基本原理',
        'use_case': '教学演示、快速原型验证',
        
        # 实际配置参数
        'vocab_size': 500,          # 词汇表大小
        'd_model': 64,              # 模型维度
        'nhead': 4,                 # 注意力头数
        'num_encoder_layers': 1,    # Encoder层数
        'num_decoder_layers': 1,    # Decoder层数
        'dim_feedforward': 256,     # FeedForward维度
        'max_seq_length': 64,       # 最大序列长度
        'cache_size': 20,           # 结果缓存大小
    },
    # ... small, medium
}
```

### 1.3 如何选择Preset?

| Preset | 参数量 | 显存占用 | 生成速度 | 适用场景 |
|--------|--------|---------|---------|----------|
| **tiny** | ~50K | <100MB | 极快 | 教学演示、快速测试 |
| **small** | ~200K | ~500MB | 快 | 日常使用、学习研究 |
| **medium** | ~800K | ~2GB | 中等 | 质量要求较高的场景 |

**决策流程**:
```
需要最快运行? → tiny
    ↓ 否
内存有限(<1GB)? → tiny
    ↓ 否
需要平衡质量和速度? → small (默认推荐)
    ↓ 否
追求更高生成质量? → medium
```

---

## 2. 自定义配置

### 2.1 覆盖Preset参数

你可以在preset基础上覆盖特定参数:

```python
from scripts.pipeline import CodeGenerationPipeline

# 基于small preset,但增大词汇表
pipeline = CodeGenerationPipeline(
    preset='small',
    vocab_size=3000,  # 覆盖默认的1000
    device='cpu'
)

# 基于tiny preset,但增加层数
pipeline = CodeGenerationPipeline(
    preset='tiny',
    num_encoder_layers=2,  # 覆盖默认的1
    num_decoder_layers=2   # 覆盖默认的1
)
```

### 2.2 创建自定义Preset

**方法1: 直接修改presets.py**

```python
# scripts/config/presets.py

PRESETS['my_custom'] = {
    'name': 'My Custom Config',
    'description': '针对特定任务的配置',
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
```

**方法2: 运行时动态创建**

```python
from scripts.config.presets import get_preset_config

# 获取基础配置
config = get_preset_config('small')

# 自定义修改
config['vocab_size'] = 2500
config['d_model'] = 192
config['nhead'] = 8

# 手动创建Pipeline
from scripts.pipeline import CodeGenerationPipeline

pipeline = CodeGenerationPipeline.__new__(CodeGenerationPipeline)
# ... 手动初始化(不推荐,建议使用方法1)
```

### 2.3 参数关系与约束

**关键约束**:
1. `d_model` 必须能被 `nhead` 整除
   ```python
   # ✅ 正确
   d_model=128, nhead=8  # 128/8=16
   
   # ❌ 错误
   d_model=128, nhead=7  # 128/7不是整数
   ```

2. `dim_feedforward` 通常是 `d_model` 的2-4倍
   ```python
   # 推荐配置
   d_model=128, dim_feedforward=256  # 2x
   d_model=256, dim_feedforward=1024 # 4x
   ```

3. `max_seq_length` 影响显存占用(O(n²))
   ```python
   # 谨慎设置
   max_seq_length=64   # 安全
   max_seq_length=512  # 需要大量显存
   ```

---

## 3. 性能优化技巧

### 3.1 GPU加速

**启用GPU**:
```python
pipeline = CodeGenerationPipeline(preset='small', device='cuda')
```

**多GPU支持**(实验性):
```python
import torch.nn as nn

model = pipeline.model
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**性能对比**:
| 设备 | 生成速度(tokens/s) | 加速比 |
|------|-------------------|--------|
| CPU (i7) | ~10 | 1x |
| GPU (RTX 3060) | ~50 | 5x |
| GPU (RTX 4090) | ~100 | 10x |

### 3.2 KV Cache优化

**自动启用**(默认开启):
```python
# KV Cache在TransformerModel内部自动管理
result = pipeline.generate(prompt="...", max_length=100)
```

**性能提升**:
- 短序列(<50 tokens): 2-5x加速
- 中等序列(50-200 tokens): 10-30x加速
- 长序列(>200 tokens): 30-50x加速

**显存权衡**:
```python
# KV Cache占用 ≈ batch_size × seq_len × d_model × num_layers × 2
# 示例: 1 × 100 × 128 × 4 × 2 = 102,400 floats ≈ 400KB
```

### 3.3 批量生成优化

**并行生成多个样本**:
```python
results = pipeline.generate_multiple_samples(
    prompt="public class UserService",
    num_samples=5,
    temperatures=[0.5, 0.6, 0.7, 0.8, 0.9],
    batch_size=2  # 每次并行生成2个
)
```

**注意**: 真正的批量化需要修改代码以支持batch处理。

### 3.4 内存优化

**减少显存占用**:
```python
# 1. 使用更小的preset
pipeline = CodeGenerationPipeline(preset='tiny')

# 2. 限制序列长度
result = pipeline.generate(prompt="...", max_length=50)

# 3. 减小缓存大小
pipeline = CodeGenerationPipeline(preset='small', cache_size=10)

# 4. 清除缓存
pipeline.cache.clear()
torch.cuda.empty_cache()  # 如果使用GPU
```

---

## 4. 扩展开发

### 4.1 添加新的采样策略

**步骤1: 在Generator中添加新方法**

```python
# scripts/generation/code_generator.py

class CodeGenerator:
    # ... 现有代码
    
    def beam_search(self, logits, beam_width=5):
        """
        Beam Search采样策略
        
        Args:
            logits: 模型输出的logits
            beam_width: beam宽度
            
        Returns:
            selected token ID
        """
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, beam_width)
        
        # 简单的beam search实现(简化版)
        selected_idx = torch.multinomial(top_probs, 1)
        return top_indices[selected_idx]
```

**步骤2: 在generate方法中支持新策略**

```python
def generate(self, prompt, max_length=50, strategy='greedy', **kwargs):
    # ... 现有代码
    
    if strategy == 'beam_search':
        next_token = self.beam_search(logits, beam_width=kwargs.get('beam_width', 5))
    elif strategy == 'greedy':
        next_token = self.greedy_decode(logits)
    # ... 其他策略
```

**步骤3: 使用新策略**

```python
result = pipeline.generate(
    prompt="public class User",
    strategy='beam_search',
    beam_width=5
)
```

### 4.2 自定义Tokenizer

**扩展现有Tokenizer**:

```python
from scripts.core.tokenizer import SimpleTokenizer

class AdvancedTokenizer(SimpleTokenizer):
    def __init__(self, vocab_size=1000, use_bpe=False):
        super().__init__(vocab_size)
        self.use_bpe = use_bpe
        
    def encode(self, text):
        if self.use_bpe:
            return self._bpe_encode(text)
        else:
            return super().encode(text)
    
    def _bpe_encode(self, text):
        # 实现BPE算法
        # ...
        pass
```

### 4.3 添加新的后处理规则

**扩展现有PostProcessor**:

```python
from scripts.generation.postprocessor import CodePostProcessor

class JavaCodePostProcessor(CodePostProcessor):
    def __init__(self):
        super().__init__()
        self.java_keywords = ['public', 'private', 'protected', 'static', ...]
    
    def validate_java_syntax(self, code):
        """Java特定的语法验证"""
        errors = []
        
        # 检查类名是否以大写字母开头
        import re
        class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.finditer(class_pattern, code)
        for match in matches:
            class_name = match.group(1)
            if not class_name[0].isupper():
                errors.append(f"Class name '{class_name}' should start with uppercase")
        
        return errors
    
    def process(self, code):
        result = super().process(code)
        
        # 添加Java特定验证
        java_errors = self.validate_java_syntax(code)
        result['errors'].extend(java_errors)
        result['is_valid'] = result['is_valid'] and len(java_errors) == 0
        
        return result
```

### 4.4 集成外部工具

**集成代码格式化器**:

```python
import subprocess

class FormattedPostProcessor(CodePostProcessor):
    def format_with_external_tool(self, code, language='java'):
        """使用外部工具格式化代码"""
        
        if language == 'java':
            # 使用google-java-format
            try:
                result = subprocess.run(
                    ['google-java-format', '-'],
                    input=code,
                    capture_output=True,
                    text=True
                )
                return result.stdout
            except FileNotFoundError:
                print("Warning: google-java-format not found")
                return code
        
        return code
```

---

## 5. 调试与诊断

### 5.1 启用Debug模式

**全局Debug**:
```python
pipeline = CodeGenerationPipeline(preset='small', debug_mode=True)
```

**模块级Debug**:
```python
from scripts.core.tokenizer import SimpleTokenizer
from scripts.core.attention import MultiHeadAttention

tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)
attention = MultiHeadAttention(d_model=128, nhead=8, debug_mode=True)
```

**Verbose输出**:
```python
result = pipeline.generate(
    prompt="public class User",
    max_length=50,
    verbose=True  # 打印详细信息
)
```

### 5.2 日志分析

**查看日志文件**:
```bash
# 日志文件位于 logs/ 目录
ls logs/

# 查看最新日志
cat logs/main_*.log | tail -100
```

**自定义日志级别**:
```python
from scripts.utils.logger import setup_logger

logger = setup_logger(__name__, level='DEBUG')  # 或 'INFO', 'WARNING', 'ERROR'
```

### 5.3 性能分析

**使用Python profiler**:
```python
import cProfile
import pstats

def benchmark_generation():
    pipeline = CodeGenerationPipeline(preset='small')
    result = pipeline.generate(prompt="public class User", max_length=50)
    return result

# 运行profiler
profiler = cProfile.Profile()
profiler.enable()
benchmark_generation()
profiler.disable()

# 打印统计信息
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 显示前20个最耗时的函数
```

**手动计时**:
```python
import time

start = time.perf_counter()
result = pipeline.generate(prompt="...", max_length=50)
elapsed = time.perf_counter() - start

print(f"Generation time: {elapsed:.4f}s")
print(f"Tokens/sec: {result['token_count']/elapsed:.2f}")
```

### 5.4 常见问题诊断

**问题1: 生成速度慢**

诊断步骤:
```python
# 1. 检查是否在使用CPU
print(f"Device: {pipeline.device}")

# 2. 检查序列长度
print(f"Max sequence length: {pipeline.model.max_seq_length}")

# 3. 检查KV Cache是否启用
# (查看TransformerModel源码中的use_cache参数)

# 解决方案:
# - 切换到GPU
# - 减小max_length
# - 使用更小的preset
```

**问题2: 内存不足**

诊断步骤:
```python
import psutil
import torch

# 检查系统内存
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")

# 检查GPU显存(如果使用GPU)
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")

# 解决方案:
# - 减小batch_size
# - 减小max_seq_length
# - 使用更小的preset
# - 清除缓存
```

**问题3: 生成质量差**

诊断步骤:
```python
# 1. 检查UNK比例
tokens = tokenizer.encode(prompt)[0]
unk_ratio = sum(1 for t in tokens if t == tokenizer.unk_id) / len(tokens)
print(f"UNK ratio: {unk_ratio:.2%}")

# 2. 检查temperature
# temperature过高会导致随机性过大

# 解决方案:
# - 增大词汇表
# - 降低temperature
# - 使用更大的preset
```

---

## 6. 最佳实践

### 6.1 配置选择指南

**场景1: 教学演示**
```python
pipeline = CodeGenerationPipeline(
    preset='tiny',      # 快速运行
    device='cpu',       # 无需GPU
    debug_mode=True     # 显示详细信息
)
```

**场景2: 日常开发**
```python
pipeline = CodeGenerationPipeline(
    preset='small',     # 平衡性能
    device='cuda' if torch.cuda.is_available() else 'cpu',
    cache_size=50       # 较大的缓存
)
```

**场景3: 高质量生成**
```python
pipeline = CodeGenerationPipeline(
    preset='medium',    # 更大模型
    device='cuda',      # 必须GPU
    cache_size=100      # 大缓存
)

result = pipeline.generate(
    prompt="...",
    temperature=0.5,    # 较低temperature
    max_length=200      # 较长序列
)
```

### 6.2 代码组织建议

**项目结构**:
```
my_project/
├── configs/
│   └── presets.yaml      # 自定义preset配置
├── models/
│   └── checkpoints/      # 保存的模型
├── scripts/
│   ├── generate.py       # 生成脚本
│   └── evaluate.py       # 评估脚本
├── outputs/
│   └── generated_code/   # 生成的代码
└── README.md
```

### 6.3 版本控制

**推荐的.gitignore**:
```gitignore
# Python
__pycache__/
*.pyc
*.pyo

# Logs
logs/

# Outputs
outputs/
visualizations/

# Models
models/checkpoints/

# IDE
.vscode/
.idea/
```

### 6.4 测试策略

**单元测试示例**:
```python
import unittest
from scripts.pipeline import CodeGenerationPipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = CodeGenerationPipeline(preset='tiny')
    
    def test_basic_generation(self):
        result = self.pipeline.generate(prompt="public class User")
        self.assertTrue(result['success'])
        self.assertGreater(len(result['processed_code']), 0)
    
    def test_cache_hit(self):
        # 第一次生成
        result1 = self.pipeline.generate(prompt="test")
        # 第二次应该命中缓存
        result2 = self.pipeline.generate(prompt="test")
        self.assertEqual(result2['source'], 'cache')

if __name__ == '__main__':
    unittest.main()
```

---

## 🎓 下一步

完成本进阶指南后,你可以:

1. **阅读完整文档**: [README_FULL.md](README_FULL.md) - 所有技术细节
2. **贡献代码**: 提交PR添加新功能或改进
3. **深入研究**: 阅读Transformer相关论文
4. **实际应用**: 将系统集成到你的项目中

---

**祝您开发愉快!**

*最后更新: 2026-05-25*
