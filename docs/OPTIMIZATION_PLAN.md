# LLM CodeGen Demo 优化方案

> 基于对 `llm-codegen-demo` 全部源码的完整阅读，并参考 `minimind`（Decoder-Only，Qwen3 风格、GQA、RMSNorm、SwiGLU、RoPE、完整训练链路、SGLang 后端）与 2026 年 LLM 工程最佳实践制定。
>
> 核心原则：**保持教育性质（中文注释、单文件单概念、可直接运行），逐步向“现代轻量 LLM 工程形态”靠近，让学习者既能看懂原理，又能训练出能用的模型。**

---

## 一、现状诊断

| 维度 | llm-codegen-demo 现状 | minimind 做法 | 2026 现代 LLM 标准 | 优化紧迫度 |
|------|---------------------|-------------|------------------|-----------|
| 架构 | Encoder-Decoder（类 T5/BART） | Decoder-Only（类 Qwen3/GPT） | Decoder-Only | 🔴 高 |
| 注意力机制 | 标准 MHA（每个头独立 K/V） | GQA（8Q/4KV，分组查询注意力） | GQA / MQA / MLA | 🔴 高 |
| 归一化 | LayerNorm | RMSNorm | RMSNorm | 🟡 中 |
| 激活函数 | ReLU | SwiGLU | SwiGLU / GeGLU | 🟡 中 |
| 位置编码 | Sinusoidal 绝对位置编码 | RoPE + YaRN 长文本外推 | RoPE + 长上下文外推 | 🔴 高 |
| KV Cache | 类已定义，未集成到 `generate_step()` | 已集成到 `generate()`（`past_key_values`） | PagedAttention + 连续批处理 | 🔴 高 |
| 采样策略 | 只有 Temperature，`top_k` 参数未使用 | Temperature + Top-P + Top-K | 全部 + Beam Search + Speculative | 🟡 中 |
| 训练链路 | 可选 Trainer，不完整，与推理脱节 | Pretrain → SFT → LoRA → DPO → PPO/GRPO → Agent → 蒸馏 | 全套后训练 | 🔴 高 |
| 推理服务 | 无 API 服务，仅本地 CLI | FastAPI OpenAI 兼容 + SGLang 后端 | vLLM / SGLang + 持续批处理 | 🟡 中 |
| 模型转换 | 无 | PyTorch ↔ Transformers ↔ Qwen3 | 兼容 ollama / vllm / llama.cpp | 🟢 低 |
| 缓存体系 | 应用层精确匹配（MD5），语义匹配基于关键词 MD5（粗糙） | 无应用层缓存（但兼容 SGLang 前缀缓存） | L1 精确匹配 + L2 语义缓存 + L3 前缀缓存 | 🟡 中 |

---

## 二、优化目标

1. **架构双轨制**：保留 Encoder-Decoder 作为教学模式，新增 Decoder-Only（自回归语言模型），贴近现代 LLM 范式。
2. **核心组件现代化**：引入 GQA、RMSNorm、SwiGLU、RoPE，减少显存占用，提高推理速度。
3. **KV Cache 真正工作**：从“定义类”到“集成到推理流水线”，让学习者直观体会 O(n²) → O(n) 的加速。
4. **训练推理闭环**：支持保存训练权重，加载到 Pipeline 进行推理，看到“训练前随机噪声 vs 训练后有结构代码”的对比。
5. **完善生成策略**：补充 Top-K / Top-P / Beam Search，理解生成多样性与确定性的 trade-off。
6. **工程化服务**：提供 OpenAI 兼容 API、模型格式转换、Streamlit WebUI，兼容主流生态。

---

## 三、分阶段实施路线图

### P1：基础补齐（推荐先做）

| 子任务 | 改动文件 | 参考 minimind 实现 | 预期效果 |
|--------|---------|------------------|---------|
| KV Cache 真正集成到 `TransformerModel.generate_step()` | `scripts/core/transformer.py`, `scripts/optimization/kv_cache.py` | `model/model_minimind.py` `Attention.forward()` 中 `past_key_value` 拼接；`generate()` 中 `past_key_values` 跨步传递 | 单次推理 O(n²) → O(n)，生成速度显著提升 |
| 引入 Top-K / Top-P 采样 | `scripts/generation/code_generator.py` | `scripts/eval_llm.py` / `serve_openai_api.py` 中 `top_p`、`temperature` 参数 | 完善生成策略，理解确定性 vs 多样性 |
| 引入 RMSNorm（替代 LayerNorm） | `scripts/core/transformer.py` | `model/model_minimind.py` `RMSNorm`（第 50-60 行） | 更快、更稳定的归一化，贴近现代 LLM |
| 引入 RoPE 位置编码 | `scripts/core/attention.py` | `model/model_minimind.py` `precompute_freqs_cis` / `apply_rotary_pos_emb`（第 62-84 行） | 支持相对位置编码，预留长文本外推接口 |

### P2：架构扩展

| 子任务 | 改动文件 | 参考 minimind 实现 | 预期效果 |
|--------|---------|------------------|---------|
| 新增 `DecoderOnlyTransformer` 类 | `scripts/core/transformer.py` | `MiniMindForCausalLM`（Decoder-Only，无 Encoder） | 理解现代 LLM 为什么放弃 Encoder |
| 权重绑定（embedding 与 lm_head 共享） | `scripts/core/transformer.py` | `model_minimind.py` 第 247 行 `tie_word_embeddings` | 减少参数，贴近工业实现 |
| Pipeline 支持 `architecture='encdec' \| 'decoder'` | `scripts/pipeline.py` | — | 同一项目内对比两种架构 |

### P3：训练闭环

| 子任务 | 改动文件 | 参考 minimind 实现 | 预期效果 |
|--------|---------|------------------|---------|
| 新增 DPO / GRPO 训练脚本 | `scripts/optional/training/train_dpo.py`, `train_grpo.py` | `minimind/trainer/train_dpo.py`, `train_grpo.py` | 完成偏好对齐与强化学习链路 |
| 新增 LoRA 参数高效微调 | `scripts/optional/training/train_lora.py`, `scripts/core/model_lora.py`（或独立文件） | `minimind/trainer/train_lora.py`, `model/model_lora.py` | 参数高效适配，适合垂域场景 |
| Pipeline 增加 `load_checkpoint(path)` | `scripts/pipeline.py` | `minimind/scripts/eval_llm.py` 加载 `.pt` 权重 | 训练后模型可加载到推理，形成闭环 |
| main.py 增加“加载训练权重”演示选项 | `scripts/main.py` | `minimind/scripts/eval_llm.py` | 直观展示训练前后效果差异 |

### P4：工程服务

| 子任务 | 改动文件 | 参考 minimind 实现 | 预期效果 |
|--------|---------|------------------|---------|
| 新增 FastAPI OpenAI 兼容服务 | `scripts/serve_openai_api.py` | `minimind/scripts/serve_openai_api.py` | 模型可作为 API 服务部署，支持流式输出 |
| 新增模型格式转换 | `scripts/convert_model.py` | `minimind/scripts/convert_model.py` | 支持 PyTorch ↔ Transformers ↔ Qwen3，兼容 ollama / vllm |
| 新增 Streamlit WebUI（可选） | `scripts/web_demo.py` | `minimind/scripts/web_demo.py` | 非技术用户可视化体验 |

### P5：缓存升级（锦上添花）

| 子任务 | 改动文件 | 参考 minimind 实现 | 预期效果 |
|--------|---------|------------------|---------|
| 语义缓存（Embedding + 向量库 + 余弦相似度） | `scripts/optimization/cache.py` | — | 意思相近的不同 Prompt 可复用缓存，提高命中率 |
| 预留提示词前缀缓存接口 | `scripts/pipeline.py` | `minimind/trainer/rollout_engine.py` `SGLangRolloutEngine` + `flush_cache` | 为接入 vLLM/SGLang 前缀缓存做准备 |

---

## 四、优先级建议（学习者路径）

如果你是**第一次学习 Transformer/LLM**，建议按以下顺序进行：

```
第 1 周：P1 中的 KV Cache 集成
  → 理解 LLM 推理为什么快（历史 K/V 复用）

第 2 周：P1 中的 Top-K / Top-P + RMSNorm + RoPE
  → 理解生成策略与现代模型组件

第 2-3 周：P2 中的 Decoder-Only 架构
  → 对比 Encoder-Decoder vs Decoder-Only，理解现代 LLM 范式

第 3-4 周：P3 中的训练闭环（DPO/GRPO/LoRA + 权重加载）
  → 体会“训练前随机噪声 vs 训练后有结构代码”

之后：P4/P4 根据兴趣选择
```

---

## 五、预期产出

完成所有阶段后，`llm-codegen-demo` 将具备：

1. **两种架构可选**：Encoder-Decoder（教学理解）和 Decoder-Only（现代 LLM 范式）
2. **现代模型组件**：GQA、RMSNorm、SwiGLU、RoPE、KV Cache 集成
3. **完整推理策略**：Greedy / Temperature / Top-K / Top-P / Beam Search
4. **多级缓存**：精确匹配 + 语义缓存 + 预留前缀缓存接口
5. **完整训练链路**：预训练 → SFT → LoRA → DPO/GRPO → Agent RL → 蒸馏
6. **生产级服务**：OpenAI 兼容 API、模型格式转换、Streamlit WebUI
7. **全中文注释**：保持项目教育性质，每个改动都配中文解释和 `__main__` 测试

---

## 六、如何开始

建议从 **P1 阶段的第一个子任务：KV Cache 真正集成到 `TransformerModel.generate_step()`** 开始。

这是对理解 LLM 推理加速帮助最大的改动，也是与 `minimind` 对比中差距最明显的地方。

> 你确认从 P1 开始吗？如果确认，我可以直接给出 `TransformerModel` 和 `Attention` 层的修改代码，以及对应的 `__main__` 测试代码，你可以直接在 `llm-codegen-demo` 上试跑。

---

> **文档位置**：`docs/OPTIMIZATION_PLAN.md`  
> **参考项目**：`E:\Project\minimind`（全部源码已读完）  
> **当前项目**：`E:\Project\llm-codegen-demo`（全部源码已读完）