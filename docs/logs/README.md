# 日志系统使用指南

## 🎯 核心功能

**解决Windows控制台300行缓冲区限制**，所有测试脚本输出自动保存到 `logs/` 目录。

### 主要特性
- ✅ **双重输出**：控制台实时显示 + 文件完整保存
- ✅ **自动命名**：`{脚本名}_{YYYYMMDD_HHMMSS}.log`
- ✅ **UTF-8编码**：完美支持中文
- ✅ **零配置**：运行脚本即可自动生成日志

---

## 💡 使用方法

### 在脚本中集成（3行代码）

```python
from logger import logging_context

if __name__ == '__main__':
    with logging_context(__file__):
        print("这条消息会同时显示在控制台和日志文件中")
        # 你的代码...
```

### 查看日志

```powershell
# 列出所有日志
ls logs/

# 用记事本打开最新日志
notepad (Get-ChildItem logs/*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

# 搜索错误
Select-String -Path "logs/*.log" -Pattern "ERROR"
```

---

## 📝 已集成的脚本（19个）

### 主脚本
- `scripts/pipeline.py`, `scripts/transformer.py`, `scripts/main.py`

### 核心模块
- `scripts/tokenizer.py`, `scripts/generator.py`, `scripts/attention.py`
- `scripts/cache.py`, `scripts/postprocessor.py`, `scripts/visualizer.py`, `scripts/kv_cache.py`

### 测试脚本
- `scripts/tests/test_all.py`, `scripts/tests/test_new_features.py`
- `scripts/tests/test_performance.py`, `scripts/tests/debug_tokenizer.py`
- `scripts/tests/tokenizer_interactive.py`, `scripts/tests/tokenizer_pdb_debug.py`
- `scripts/tests/verify_math.py`, `scripts/test_training_and_cache.py`

**运行以上任何脚本都会自动生成日志文件。**

---

## 📊 日志文件格式

```
================================================================================
日志文件: transformer_20260509_160258.log
创建时间: 2026-05-09 16:02:58
Python版本: 3.13.11
工作目录: E:\Project\llm-codegen-demo
================================================================================

[INFO] 日志系统已启动
[INFO] 日志文件: E:\Project\llm-codegen-demo\logs\transformer_20260509_160258.log

... (完整的测试输出) ...

================================================================================
日志结束时间: 2026-05-09 16:02:58
================================================================================
```

---

## ⚙️ 高级用法

### 自定义日志目录
```python
with logging_context(__file__, log_dir='my_logs'):
    print("日志保存到 my_logs/ 目录")
```

### 临时禁用日志
```python
with logging_context(__file__, enable_logging=False):
    print("只输出到控制台")
```

---

## ❓ 常见问题

**Q: 为什么需要日志文件？**  
A: Windows控制台有300行缓冲区限制，超出部分会被覆盖。日志文件保存完整输出。

**Q: 日志文件会很大吗？**  
A: 一般测试脚本2-15 KB，启用debug_mode时可能达到50-100 KB。

**Q: 可以同时运行多个脚本吗？**  
A: 可以！每个日志文件名包含时间戳，不会冲突。

**Q: 如何清理旧日志？**  
```powershell
# 删除7天前的日志
Get-ChildItem logs/*.log | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item
```

---

## 📌 最佳实践

1. ✅ 重要测试保留日志，用于问题追溯
2. ✅ 定期清理旧日志，避免占用磁盘空间
3. ✅ 不要将 `logs/` 提交到Git（已在.gitignore中）
4. ✅ 使用VS Code或记事本查看日志（PowerShell需用 `-Encoding UTF8`）

---

## 🚀 快速开始

```bash
# 1. 运行测试脚本
python scripts\transformer.py

# 2. 查看生成的日志
ls logs/

# 3. 用记事本打开
notepad logs\transformer_*.log
```

**就是这么简单！** 🎉
