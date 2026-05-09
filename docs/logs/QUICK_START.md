# 日志系统快速入门

## 🎯 一句话总结

**运行任何测试脚本，输出自动保存到 `logs/` 目录，再也不怕控制台显示不全！**

---

## ⚡ 30秒开始

```bash
# 1. 运行脚本
python scripts\transformer.py

# 2. 查看日志
ls logs/

# 3. 打开日志
notepad logs\transformer_*.log
```

完成！日志文件已自动生成。

---

## 💡 核心优势

| 之前的问题 | 现在的解决方案 |
|-----------|--------------|
| Windows控制台300行限制 | 文件完整保存所有输出 |
| 重要输出被覆盖 | 随时查看历史日志 |
| 无法追溯问题 | 完整的日志记录 |

### 特性
- ✅ **双重输出**：控制台 + 文件
- ✅ **自动命名**：`{脚本名}_{时间戳}.log`
- ✅ **UTF-8编码**：支持中文
- ✅ **零配置**：开箱即用

---

## 📋 已集成的脚本

**无需修改代码，直接运行即可：**

### 主脚本（3个）
- `scripts/pipeline.py`
- `scripts/transformer.py`
- `scripts/main.py`

### 核心模块（7个）
- `tokenizer.py`, `generator.py`, `attention.py`
- `cache.py`, `postprocessor.py`, `visualizer.py`, `kv_cache.py`

### 测试脚本（9个）
- `tests/test_all.py`, `tests/test_new_features.py`
- `tests/test_performance.py`, `tests/debug_tokenizer.py`
- `tests/tokenizer_interactive.py`, `tests/tokenizer_pdb_debug.py`
- `tests/verify_math.py`, `test_training_and_cache.py`

**共19个脚本已集成日志系统。**

---

## 🔍 使用示例

### 示例1：Transformer测试
```bash
python scripts\transformer.py
# 生成: logs/transformer_20260509_160258.log
```

### 示例2：完整测试套件
```bash
python scripts\tests\test_all.py
# 生成: logs/test_all_20260509_160345.log
```

### 示例3：搜索错误
```powershell
Select-String -Path "logs/*.log" -Pattern "ERROR"
```

---

## 🛠️ 在自己的脚本中使用

只需3行代码：

```python
from logger import logging_context

if __name__ == '__main__':
    with logging_context(__file__):
        print("消息会同时显示在控制台和日志文件中")
```

---

## 📊 日志文件大小参考

| 类型 | 大小 | 示例 |
|------|------|------|
| 简单测试 | 2-5 KB | transformer.py |
| 完整测试 | 10-15 KB | test_all.py |
| 性能测试 | 5-10 KB | test_performance.py |
| 调试模式 | 50-100 KB | debug_mode=True |

---

## ❓ 常见问题

**Q: 日志文件在哪里？**  
A: 项目根目录的 `logs/` 文件夹。

**Q: 可以同时运行多个脚本吗？**  
A: 可以！文件名包含时间戳，不会冲突。

**Q: 如何禁用日志？**  
```python
with logging_context(__file__, enable_logging=False):
    print("只输出到控制台")
```

**Q: 会影响性能吗？**  
A: 影响极小（<1%）。

**Q: 如何清理旧日志？**  
```powershell
# 删除7天前的日志
Get-ChildItem logs/*.log | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item
```

---

## 📌 最佳实践

1. ✅ 重要测试保留日志
2. ✅ 定期清理旧日志
3. ✅ 不要提交 `logs/` 到Git
4. ✅ 用VS Code或记事本查看日志

---

## ✨ 总结

**现在你可以：**
- ✅ 运行脚本，不用担心输出丢失
- ✅ 随时查看完整日志
- ✅ 对比不同运行结果
- ✅ 分享日志协助调试

**就是这么简单！** 🎉

---

## 📚 下一步

### 想深入了解？
- 📖 [README.md](./README.md) - 完整使用指南
- 🔧 [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - 技术实现细节
- 📊 [TEST_REPORT.md](./TEST_REPORT.md) - 测试报告

### 想在自己的项目中使用？
1. 复制 `scripts/logger.py` 到你的项目
2. 在脚本中导入：`from logger import logging_context`
3. 使用上下文管理器包裹代码
4. 完成！

### 遇到问题？
- 查看 [README.md](./README.md) 的“故障排查”章节
- 检查日志文件是否正确生成
- 确认使用的是UTF-8兼容的编辑器
