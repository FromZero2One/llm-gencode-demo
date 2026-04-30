# Tokenizer Debug 指南

本文档介绍如何调试和可视化 tokenizer 的工作过程。

## 📁 可用的 Debug 工具

### 1. **debug_tokenizer.py** - 完整的 Tokenizer 分析工具

提供详细的 tokenization 过程分析和可视化。

#### 使用方法

```bash
source venv/bin/activate
python debug_tokenizer.py
```

#### 功能特性

- ✅ **Step-by-step 分析**: 显示每个处理步骤的详细信息
- ✅ **Token 可视化**: 彩色显示已知/未知 tokens
- ✅ **词汇表分析**: 查看词汇表组成和统计
- ✅ **多文本对比**: 比较多个文本的 tokenization 结果
- ✅ **边界测试**: 测试空字符串、中文等特殊情况
- ✅ **UNK 检测**: 识别不在词汇表中的 tokens

#### 输出示例

```
[Step 1] 原始输入文本
  文本: "public class UserService"
  长度: 24 字符
  词汇表大小: 248

[Step 2] 分词过程 (tokenize)
  分词数量: 3 tokens
  Tokens 列表:
    [ 1] 'public'
    [ 2] 'class'
    [ 3] 'UserService'

[Step 4] Token → ID 映射
    [ 0] '<BOS>'               → ID:    2 ✓
    [ 1] 'public'              → ID:    4 ✓
    [ 2] 'class'               → ID:    7 ✓
    [ 3] 'UserService'         → ID:    3 ✗ UNK

[Step 7] 统计分析
  总 token 数: 3
  已知 tokens: 2
  未知 tokens (UNK): 1
  词汇表覆盖率: 66.7%
```

---

### 2. **tokenizer_interactive.py** - 交互式 Debug 工具

实时输入文本并查看 tokenization 结果。

#### 使用方法

```bash
source venv/bin/activate
python tokenizer_interactive.py
```

或直接分析指定文本:

```bash
python tokenizer_interactive.py "public class UserService"
```

#### 交互命令

- `<文本>` - 分析输入的文本
- `vocab` - 查看词汇表统计
- `test` - 运行快速测试
- `help` - 显示帮助
- `quit/exit` - 退出程序

#### 使用示例

```
>>> public class UserService

分析文本: "public class UserService"
───────────────────────────────────────────────────────────────────────────────
Token 可视化
───────────────────────────────────────────────────────────────────────────────

Token 序列:
  [ 0] '<BOS>'               → ID:    2 ✓
  [ 1] 'public'              → ID:    4 ✓
  [ 2] 'class'               → ID:    7 ✓
  [ 3] 'UserService'         → ID:    3 ✗
  [ 4] '<EOS>'               → ID:    1 ✓

统计信息:
  总 tokens: 3
  已知 tokens: 2
  未知 tokens (UNK): 1
  词汇表覆盖率: 66.7%

⚠  未知 Tokens (建议添加到词汇表):
    - 'UserService'
```

---

### 3. **tokenizer_pdb_debug.py** - Python Debugger (pdb) 示例

演示如何使用 Python 内置调试器进行单步调试。

#### 使用方法

```bash
source venv/bin/activate
python tokenizer_pdb_debug.py
```

#### 常用 pdb 命令

| 命令 | 说明 |
|------|------|
| `n` (next) | 执行下一行 |
| `s` (step) | 进入函数内部 |
| `c` (continue) | 继续执行到下一个断点 |
| `p <变量>` | 打印变量值 |
| `l` (list) | 显示当前代码 |
| `h` (help) | 显示帮助 |
| `q` (quit) | 退出调试器 |
| `w` (where) | 显示调用栈 |
| `b <行号>` | 设置断点 |

#### 调试示例

```python
import pdb
from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)

# 设置断点
pdb.set_trace()

# 在调试器中可以:
# p tokenizer.vocab        # 查看词汇表
# p tokens                 # 查看 tokens
# n                        # 执行下一行
# s                        # 进入函数
```

---

### 4. **test_debug.py** - 完整流程 Debug

测试整个代码生成 pipeline 的 debug 模式。

#### 使用方法

```bash
source venv/bin/activate
python test_debug.py
```

#### 输出内容

- Pipeline 初始化过程
- Tokenizer 详细日志
- Transformer Encoder/Decoder 执行过程
- Attention 机制的张量形状
- 代码生成每一步的详细信息

---

## 🔧 启用/禁用 Debug 模式

### 方法1: 在创建 Tokenizer 时设置

```python
from tokenizer import SimpleTokenizer

# 启用 debug 模式
tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)

# 禁用 debug 模式 (生产环境)
tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=False)
```

### 方法2: 配置日志级别

```python
import logging

# DEBUG - 最详细的日志
logging.basicConfig(level=logging.DEBUG)

# INFO - 一般信息
logging.basicConfig(level=logging.INFO)

# WARNING - 仅警告和错误
logging.basicConfig(level=logging.WARNING)
```

---

## 📊 理解 Debug 输出

### Token 状态符号

- `✓` - Token 在词汇表中 (已知)
- `✗` - Token 不在词汇表中 (UNK)

### 关键指标

1. **词汇表覆盖率**: 已知 tokens / 总 tokens
   - > 90%: 优秀
   - 70-90%: 良好
   - < 70%: 需要扩展词汇表

2. **UNK 比例**: 未知 tokens / 总 tokens
   - < 5%: 可接受
   - 5-15%: 需要注意
   - > 15%: 建议扩展词汇表

3. **Attention Mask**: 
   - `1` = 有效 token
   - `0` = padding token

---

## 🎯 常见调试场景

### 场景1: 检查为什么某些 tokens 被识别为 UNK

```bash
python debug_tokenizer.py
```

查看 "[Step 8] ⚠ 未知 Tokens" 部分，了解哪些 tokens 需要添加到词汇表。

### 场景2: 验证 tokenization 是否正确

```bash
python tokenizer_interactive.py
>>> your_code_here
```

检查解码结果是否与原始文本匹配。

### 场景3: 分析词汇表覆盖情况

```bash
python tokenizer_interactive.py
>>> vocab
```

查看词汇表组成和统计信息。

### 场景4: 单步调试找出问题

```bash
python tokenizer_pdb_debug.py
```

使用 pdb 命令逐步执行，检查每一步的变量值。

---

## 💡 最佳实践

1. **开发阶段**: 启用 `debug_mode=True` 和 `logging.DEBUG`
2. **测试阶段**: 使用 `debug_tokenizer.py` 进行全面测试
3. **生产环境**: 禁用 debug 模式以提高性能
4. **问题排查**: 使用 `tokenizer_pdb_debug.py` 进行单步调试

---

## 🚀 扩展词汇表

如果发现很多 UNK tokens，可以扩展词汇表:

```python
# 在 tokenizer.py 的 _build_vocabulary 方法中添加新 tokens
common_tokens = [
    # ... 现有 tokens ...
    'UserService',  # 添加新 token
    'Long',         # 添加新 token
    'orElse',       # 添加新 token
]
```

---

## 📝 示例输出

完整的 debug 输出示例请运行:

```bash
python debug_tokenizer.py 2>&1 | tee debug_output.log
```

这将保存所有输出到 `debug_output.log` 文件供后续分析。
