# LLM 代码生成演示项目 - 学习进度记录

**最后更新**: 2026-05-18  
**学习者角色**: 大模型开发架构师视角

---

## 📅 第一阶段：基础与输入层（已完成）

### 1. Tokenizer (分词器)
*   **核心文件**: `scripts/tokenizer.py`
*   **掌握内容**:
    *   理解了文本如何被正则表达式切分为 Tokens。
    *   掌握了 `vocab`（词汇表）与 `ID` 的映射关系。
    *   深入理解了 `Attention Mask` 的作用：区分有效内容与 Padding，防止模型关注无意义的填充位。
    *   了解了 `BOS` (开始)、`EOS` (结束)、`UNK` (未知) 等特殊标记的意义。

### 2. 交互工具优化
*   **核心文件**: `scripts/tests/tokenizer_interactive.py`
*   **完成工作**:
    *   将原有的自由文本命令模式重构为**数字菜单选择模式**。
    *   提升了工具的易用性，降低了认知负荷，符合架构师对 UX 的要求。

---

## 🔄 第二阶段：嵌入与位置编码（进行中）

### 1. Embedding & Positional Encoding
*   **核心文件**: `scripts/attention.py`
*   **当前进度**:
    *   已初步运行脚本，观察了位置编码的数值分布。
    *   正在解析 `PositionalEncoding` 类中正弦/余弦函数的实现逻辑。
    *   **待深入**: 理解为什么使用三角函数来注入位置信息，以及它如何帮助模型捕捉相对位置关系。

---

## ⏳ 第三阶段：核心引擎——Transformer（待办）

### 1. Attention Mechanism (注意力机制)
*   **核心文件**: `scripts/attention.py` (`MultiHeadAttention` 类)
*   **学习目标**:
    *   彻底搞懂 $Q, K, V$ 矩阵的物理意义及变换过程。
    *   理解多头（Multi-Head）并行计算的实现细节（Transpose 与 Reshape）。
    *   分析 Softmax 缩放因子 $\sqrt{d_k}$ 的作用。

### 2. Encoder-Decoder 架构
*   **核心文件**: `scripts/transformer.py`
*   **学习目标**:
    *   对比 Encoder 和 Decoder 的结构差异。
    *   理解 Causal Mask（因果掩码）在 Decoder 中防止“偷看”未来的原理。
    *   掌握 Residual Connection（残差连接）和 LayerNorm 如何解决梯度消失问题。

---

## ⏳ 第四阶段：推理与工程化（待办）

### 1. Sampling & KV Cache
*   **核心文件**: `scripts/generator.py`, `scripts/kv_cache.py`
*   **学习目标**:
    *   对比 Greedy、Temperature、Top-K、Top-P 采样策略。
    *   理解 KV Cache 如何将推理复杂度从 $O(n^2)$ 降为 $O(n)$。

### 2. Pipeline & Training
*   **核心文件**: `scripts/pipeline.py`, `scripts/trainer.py`
*   **学习目标**:
    *   串联全链路流程。
    *   研究 CrossEntropyLoss 与 AdamW 优化器的实现。

---

## 💡 架构师学习笔记
*   **调试技巧**: 善用 `debug_mode=True` 观察 Tensor 的形状变化（Shape），这是理解神经网络数据流的捷径。
*   **可视化驱动**: 多利用 `visualizer.py` 生成的热力图来辅助理解抽象的注意力权重。
*   **实验精神**: 通过修改 `d_model` 或 `temperature` 等参数，直观感受超参数对生成质量的影响。
