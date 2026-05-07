# 🚀 完整 LLM 流程演示 - 快速开始指南

## 📋 可用的演示方式

### **1. 交互式主菜单 (推荐)** ⭐

运行 [`scripts/main.py`](file:///home/wsm/codes/llm-gencode-demo/scripts/main.py) 进入交互式菜单：

```bash
source venv/bin/activate
python scripts/main.py
```

**可用选项**:
- `1` - 基本代码生成流程（展示完整 pipeline）
- `2` - 不同采样策略对比
- `3` - 缓存机制演示
- `4` - 多样本生成
- `5` - 可视化功能
- `6` - 交互模式（自己输入 prompt）
- `7` - 运行所有演示
- `0` - 退出

---

### **2. 直接运行特定演示**

#### **演示1: 基本代码生成流程**
```bash
source venv/bin/activate
echo "1" | python scripts/main.py
```

**展示内容**:
- ✅ Tokenization 过程
- ✅ Transformer Encoder/Decoder
- ✅ Attention 机制
- ✅ 代码生成
- ✅ 后处理
- ✅ 缓存机制

**输出示例**:
```
================================================================================
                    演示1: 基本代码生成流程
================================================================================

################################################################################
# 测试用例 1: public class UserService
################################################################################

============================================================
[Pipeline] Step 1: 检查缓存
============================================================
[Cache] [MISS] Cache MISS

============================================================
[Pipeline] Step 2: 创建采样策略
============================================================
[Sampling] 使用Temperature采样, temperature=0.7

============================================================
[Pipeline] Step 3: 生成代码
============================================================
[CodeGenerator] Prompt tokenized: 32 tokens
[Transformer] 开始Encoder过程
...
```

---

### **3. Tokenizer Debug 工具**

#### **完整分析**
```bash
python scripts/debug_tokenizer.py
```

**功能**:
- 9步详细分析
- Token → ID 映射
- 词汇表统计
- 多文本对比
- UNK tokens 检测

#### **交互式调试**
```bash
python scripts/tokenizer_interactive.py
```

**使用**:
```
>>> public class User
>>> vocab
>>> test
>>> quit
```

或直接分析:
```bash
python scripts/tokenizer_interactive.py "public class UserService"
```

---

### **4. 完整 Pipeline Debug**

```bash
python scripts/test_debug.py
```

**展示**:
- Pipeline 初始化
- 每个组件的 debug 日志
- 完整的执行流程
- 性能统计

---

## 🎯 快速体验完整流程

### **最简单的方式**

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行基本演示（自动选择选项1）
python scripts/main.py <<< "1"
```

### **查看详细的 Tokenizer 分析**

```bash
# 运行 tokenizer 完整分析
python scripts/debug_tokenizer.py 2>&1 | head -100
```

### **交互式体验**

```bash
# 进入交互模式
python scripts/main.py <<< "6"

# 然后输入您的 prompt
>>> public class UserController
>>> @GetMapping("/api/users")
>>> quit
```

---

## 📊 理解输出

### **Debug 日志解读**

#### **Tokenizer 日志**
```
2026-04-30 22:11:20,460 - tokenizer - DEBUG - 原始文本长度: 24 字符
2026-04-30 22:11:20,460 - tokenizer - DEBUG - 分词结果数量: 3 tokens
2026-04-30 22:11:20,460 - tokenizer - DEBUG - Tokens: ['public', 'class', 'UserService']
```
- ✅ 显示输入文本被分成 3 个 tokens
- ✅ 可以看到哪些 tokens 在词汇表中

#### **Attention 日志**
```
2026-04-30 22:11:20,461 - attention - DEBUG -   - Batch size: 1
2026-04-30 22:11:20,461 - attention - DEBUG -   - Query sequence length: 32
2026-04-30 22:11:20,461 - attention - DEBUG -   - Input shape: torch.Size([1, 32, 128])
2026-04-30 22:11:20,462 - attention - DEBUG -   - Attention scores shape: torch.Size([1, 8, 32, 32])
```
- ✅ 显示张量形状变化
- ✅ 验证 attention weights 是否正确 (sum = 1.0)

#### **Pipeline 日志**
```
[Pipeline] Step 1: 检查缓存
[Cache] [MISS] Cache MISS

[Pipeline] Step 2: 创建采样策略
[Sampling] 使用Temperature采样

[Pipeline] Step 3: 生成代码
[CodeGenerator] 开始代码生成

[Pipeline] Step 4: 后处理
[PostProcessor] 开始后处理

[Pipeline] Step 5: 缓存结果
[Cache] Added new entry
```
- ✅ 展示完整的 5 步流程
- ✅ 每个步骤都有详细说明

---

## 🔧 自定义演示

### **修改测试 prompts**

编辑 [`scripts/main.py`](file:///home/wsm/codes/llm-gencode-demo/scripts/main.py:51-L55):

```python
test_prompts = [
    "public class UserService",
    "public User findById",
    "private List<User>",
    # 添加您自己的 prompts
    "@RestController",
    "public void processRequest",
]
```

### **调整生成参数**

```python
result = pipeline.generate(
    prompt=prompt,
    max_length=50,        # 增加生成长度
    temperature=0.9,      # 更高温度 = 更随机
    top_k=100,            # 更大的候选集
    use_cache=True,       # 启用缓存
    do_post_process=True, # 启用后处理
    verbose=True          # 显示详细日志
)
```

### **关闭 Debug 模式**

```python
pipeline = CodeGenerationPipeline(
    ...,
    debug_mode=False  # 关闭 debug，提高性能
)
```

---

## 📚 相关文档

- [`TOKENIZER_DEBUG_GUIDE.md`](file:///home/wsm/codes/llm-gencode-demo/TOKENIZER_DEBUG_GUIDE.md) - Tokenizer Debug 详细指南
- [`README.md`](file:///home/wsm/codes/llm-gencode-demo/README.md) - 项目完整文档
- [`TEST_REPORT.md`](file:///home/wsm/codes/llm-gencode-demo/TEST_REPORT.md) - 测试报告

---

## 💡 提示

1. **首次运行**: 建议使用 `echo "1" | python scripts/main.py` 查看基本流程
2. **深入学习**: 运行 `python scripts/debug_tokenizer.py` 了解 tokenization 细节
3. **实验**: 使用交互模式 `python scripts/main.py <<< "6"` 尝试自己的 prompts
4. **性能**: 生产环境设置 `debug_mode=False`

---

## 🎓 学习路径

1. **Step 1**: 运行基本演示，观察完整流程
2. **Step 2**: 阅读 debug 输出，理解每个步骤
3. **Step 3**: 使用 tokenizer debug 工具深入了解分词
4. **Step 4**: 修改参数，观察不同效果
5. **Step 5**: 阅读源代码，理解实现细节

祝您学习愉快！🚀
