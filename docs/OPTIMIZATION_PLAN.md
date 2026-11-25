# LLM CodeGen Demo 优化方案

> 基于对 `llm-codegen-demo` 全部源码的完整阅读，并参考 `minimind`（Decoder-Only，Qwen3 风格、GQA、RMSNorm、SwiGLU、RoPE、完整训练链路、SGLang 后端）与 2026 年 LLM 工程最佳实践制定。
>
> 核心原则：**保持教育性质（中文注释、单文件单概念、可直接运行），逐步向"现代轻量 LLM 工程形态"靠近，让学习者既能看懂原理，又能训练出能用的模型。**

---

## ✅ 实施进度总览（2026-11 更新）

| 阶段 | 内容 | 状态 |
|------|------|------|
| P1 | KV Cache 集成 / Top-K·Top-P 采样 / RMSNorm / RoPE | ✅ 完成 |
| P2 | DecoderOnlyTransformer + 权重绑定 + 双架构 Pipeline | ✅ 完成（顺带修复因果掩码语义反转 bug） |
| P3 | Pipeline.load_checkpoint / save_model + Trainer 双架构分发 | ✅ 完成（顺带修复 TextDataset 与模块导入 bug） |
| P4 | OpenAI 兼容 API（流式 SSE）+ HF 格式模型转换 | ✅ 完成 |
| P5 | SemanticCache（n-gram 向量 + 余弦相似度）双级缓存 | ✅ 完成 |
| P6 | BPE 分词器（对标 minimind train_tokenizer） | ✅ 完成 |
| P7 | 真实数据管线 + 预训练闭环 | ✅ 完成（PretrainDataset + train_pretrain，梯度累积） |
| P8 | SFT 指令微调 + Chat Template + 真多轮 API | ✅ 完成（chat_template + SFTDataset + train_full_sft） |
| P9 | LoRA 微调（apply / load / merge） | ✅ 完成（lora.py + train_lora.py） |
| P10 | 架构补强（GQA / SwiGLU / SDPA）+ DPO 对齐训练 | ✅ 完成（attention GQA/SDPA + SwiGLU + train_dpo） |
| 远期待做 | MoE / GRPO·PPO / Agent / 蒸馏 / Beam Search / Streamlit WebUI | ⬜ 暂不实施 |

### 后期待做清单（2026-11 收官时整理）

P1–P10 已全部完成并提交。以下为当时主动排除或做了简化、留待后期再做的项，按实施成本从低到高排列：

**快速可补（1 天内量级）**
| 项 | 说明 | 建议起点 |
|----|------|---------|
| API 加载 LoRA 插件 | `serve_openai_api.py` 加 `--lora_path`，启动时 `apply_lora + load_lora`，一个底座挂多个行为插件 | `scripts/core/lora.py` 已有现成函数，纯接线 |
| INT8 动态量化 | 推理提速 + 体积减半，`torch.ao.quantization.quantize_dynamic` 一行核心 | 可加 `scripts/optimization/quantize.py` |
| Streamlit 聊天 WebUI | 把 SFT 模型包成可交互对话界面，演示效果最直观 | `scripts/webui.py`，复用 `chat_template` + `generate_step` |

**中等成本（对齐 minimind 完整实现）**
| 项 | 现状差距 | 做法 |
|----|---------|------|
| Beam Search 解码 | 只有 temperature/top-k/top-p 贪心与采样 | `code_generator.py` 加 beam 维度展开 |
| YaRN/NTK 长度外推 | RoPE 有，但超过预生成长度即失效 | RoPE 频率缩放公式 |
| MQA / MLA | 只有 GQA | MQA=1 个 KV 头的 GQA 特例，改动极小；MLA 较复杂 |
| PagedAttention 式缓存 | 教学版预分配连续张量 | 分页块管理，参考 vLLM 论文 |

**大工程（当初明确排除）**
| 项 | 排除原因 |
|----|---------|
| MoE（混合专家） | 路由+专家并行，demo 规模收益小 |
| GRPO / PPO 强化学习对齐 | DPO 已覆盖对齐主线；PPO 需 4 模型在线，重且不稳 |
| Agent / 工具调用 | 超出"LLM 本体教学"范围 |
| 知识蒸馏 | 需要教师模型 |
| 多机分布式训练 | 保持零依赖单机可跑 |
| vLLM / ollama / llama.cpp 部署导出 | 引入外部重依赖，破坏零依赖原则 |
| 真实规模指令数据 | demo 仅 400 条重复语料，真实效果需更大更多样的 SFT/DPO 数据 |

### 已知测试盲区（2026-11 收官审计，后期开发前先补）

P7–P10 的验证大多为一次性脚本跑通，以下项**未充分验证**，动相关代码前建议先补验证：

| # | 盲区 | 风险 | 补验证做法 |
|---|------|------|-----------|
| 1 | `serve_openai_api.py` 的 `--chat_template` 多轮链路只做了语法检查，未起服务实测 | API 返回的 prompt 格式可能错，多轮历史可能没进模板 | 起服务后用 curl/requests 发带历史的 messages 实测；顺带补 `messages_to_prompt` 单元测试 |
| 2 | GQA / SwiGLU / SDPA / DPO / LoRA 验证均为一次性脚本，未固化进 `test_all.py` | 改 `attention.py`/`transformer.py` 时回归只能兜住"SDPA 权重为 None"，架构回归靠人工记忆 | 把下列断言固化为测试：GQA n_kv_heads=nhead 等价 MHA、SDPA 与手写输出差 <1e-5、LoRA merge 前后输出一致、DPO margin 训练后 >0 |
| 3 | bug 修复（调度器 clamp、PE 跳过加载）只验证了正确路径，无负路径回归用例 | 将来重构可能悄悄退化回老 bug | 用例：调度器步数超 total_steps 2 倍后 lr 仍 >= min_lr；checkpoint 的 pe 形状不一致时加载不报错 |
| 4 | SFT/LoRA demo 生成质量弱（输出 `.println("invalid")` 类片段） | 数据问题非代码 bug：SFT 语料仅 400 条重复样本 | 换更大更多样的指令数据重训；代码侧无需修改 |

> 注：P1–P5 及 P6 的既有测试（`test_all.py` 8 模块 + 各文件 `__main__` 自验证区）均已全绿；#2 的"SDPA 权重为 None"断言已固化进 `test_all.py`。

### P6–P10 详细方案（对标 minimind，2026-11 制定）

> 目标：从"理解 LLM 原理的教学项目"升级为"**能从零练出一个会对话的小模型**"的完整项目。
> 依赖链：P6 → P7 → P8 → P9 → P10（严格串行，每阶段产出下一阶段的输入）。

**P6 · BPE 分词器**
- 现状问题：`SimpleTokenizer` 正则 + 固定词表，OOV 全靠 `<UNK>`，是模型输出大量 `<UNK>` 的根因
- 新建 `scripts/core/bpe_tokenizer.py`：`tokenizers` 库 ByteLevel BPE，教学词表 6400 + 36 特殊位（chat 角色位为 P8 铺路）
- 兼容层：与 `SimpleTokenizer` 相同的 `encode/decode/vocab` 接口，Pipeline 加 `tokenizer_type='regex'|'bpe'` 零改动切换
- 验收：任意输入（生造词/混合中英）无损编解码 roundtrip

**P7 · 真实数据管线 + 预训练**
- 新建 `scripts/data/lm_dataset.py`：`PretrainDataset`（txt/jsonl → BPE → 滑窗自回归样本，支持流式读取）+ demo 语料生成脚本
- 新建 `scripts/training/train_pretrain.py`：对接 Trainer，支持梯度累积，loss 曲线对接 visualizer
- 验收：训练后模型能续写语料中的代码模式，不再纯噪声

**P8 · SFT 指令微调 + Chat Template**
- 新建 `scripts/core/chat_template.py`：ChatML 风格模板 + loss mask（仅 assistant 部分计 loss）
- 新建 `scripts/data/sft_dataset.py`：jsonl → `(input_ids, labels)`，prompt 部分 label=-100
- 新建 `scripts/training/train_full_sft.py` + demo 问答数据
- `serve_openai_api.py` 接模板，支持真多轮对话
- 验收：API 问答按 SFT 格式回答

**P9 · LoRA 微调**
- 新建 `scripts/core/lora.py`：`LoRALinear`（W' = W + (α/r)·BA）、`apply_lora / load_lora / merge_lora`
- 新建 `scripts/training/train_lora.py`：只训低秩参数，打印参数量对比
- `serve_openai_api.py` 加 `--lora_path`
- 验收：主模型权重 hash 不变，行为改变

**P10 · 架构补强 + DPO**
- GQA：`n_kv_heads < n_q_heads`，KV Cache 显存 ÷4，与已有 KV Cache 教学串联
- SwiGLU：`ffn_type='relu'|'swiglu'` 开关
- SDPA：`F.scaled_dot_product_attention` 快速路径（保留手写路径教学）
- DPO：`train_dpo.py` + 偏好对数据，sigmoid loss
- 验收：GQA 显存对比、DPO 后 chosen 偏好上升

**明确排除**：MoE、PPO/GRPO/Agent/蒸馏、多机训练、爬取数据集（保持零依赖可运行）。

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