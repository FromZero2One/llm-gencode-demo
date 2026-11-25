# 缓存机制学习笔记

> 基于 `llm-codegen-demo`（教学项目）与 `minimind`（轻量训练项目）两个代码库的缓存实现，整理企业级 AI / Agent 缓存体系。

---

## 一、核心结论

| 缓存层级 | 作用范围 | llm-codegen-demo | minimind | 企业级标准 |
|---------|---------|-----------------|---------|-----------|
| **L1 精确匹配缓存** | 应用层：同一 Prompt 不重复生成 | ✅ `GenerationCache`（MD5 精确匹配） | ❌ 无 | ✅ Redis + 参数哈希 |
| **L2 语义缓存** | 应用层：意思相近的问题复用答案 | ⚠️ 有，但基于关键词提取 + MD5，粗糙且易误命中 | ❌ 无 | ✅ Embedding + 向量库 + 余弦相似度 |
| **L3 提示词/前缀缓存** | API/模型服务层：复用长 System Prompt 的计算 | ❌ 无 | ❌ 无（但兼容 vLLM/SGLang，可切换） | ✅ API 厂商提供（如 Anthropic `cache_control`）或自建 Prefix Caching |
| **L4 模型内部 KV Cache** | 推理引擎层：单次生成内部复用历史 K/V | ⚠️ 类已定义，未集成到 `generate_step()` | ✅ 已集成到 `generate()`（`past_key_values`） | ✅ PagedAttention、连续批处理等 |
| **L5 Agent 轨迹缓存** | Agent 层：缓存多步 Thought/Action/Observation | ❌ 无 | ❌ 无 | ✅ 缓存整个推理轨迹 |

---

## 二、llm-codegen-demo 中的缓存实现

### 1. GenerationCache（结果缓存）—— 应用层

文件：`scripts/optimization/cache.py`

**解决问题**：用户反复问同样的问题（如“写个冒泡排序”），避免重新调用模型推理。

**核心数据结构**：
- `self.cache = OrderedDict()`：哈希 -> 缓存条目，保持插入顺序，便于实现 LRU 淘汰
- `self.access_count = {}`：记录每个条目的访问次数
- `self.stats`：命中/未命中/淘汰次数

**缓存条目结构**：
```python
{
    'prompt': str,              # 原始提示词
    'result': str,              # 生成结果
    'semantic_hash': str,       # 语义哈希
    'timestamp': float,
    'prompt_length': int,
    'result_length': int
}
```

**两种查找方式**：

| 方法 | 命中规则 | 问题 |
|------|---------|------|
| `get(prompt)` | `MD5(prompt)` 完全相同 | 对空格、大小写、标点极度敏感，命中率低 |
| `get_similar(prompt)` | 提取 Java 关键字（`public/class/Service` 等），排序后求 MD5 | 中文自然语言提取不到关键字，语义哈希为空，不同请求会得到相同空 MD5，**易误命中** |

**Pipeline 调用方式**（`scripts/pipeline.py`）：
```python
# 仅调用了精确匹配，未调用 get_similar
cached_result = self.cache.get(prompt)
if cached_result is not None:
    return result  # 直接返回，不碰模型
# 未命中 -> 调用模型生成 -> self.cache.put(prompt, raw_code)
```

---

### 2. KVCache / KVCacheManager —— 模型层（已定义，未集成）

文件：`scripts/optimization/kv_cache.py`

**解决问题**：Transformer 自回归生成时，每步只需计算新 Token 的 K/V，历史 Token 的 K/V 复用，将复杂度从 O(n²) 降至 O(n)。

**核心结构**：
- `key_cache` / `value_cache`：预分配的零张量，形状 `(num_layers, batch_size, num_heads, max_seq_len, head_dim)`
- `update(layer_idx, new_k, new_v, position)`：将新 K/V 写入缓存切片
- `get(layer_idx, length)`：读取缓存的 K/V

**关键缺口**：`TransformerModel.generate_step()` 中**没有使用 KVCache**。当前实现每次都传入完整 `generated` 序列，Decoder 重新对所有历史位置做 Self-Attention，未获得预期加速。

---

## 三、minimind 项目中的缓存实现

### 1. 模型内部 KV Cache —— 已集成

文件：`model/model_minimind.py`

- `Attention.forward()` 接收 `past_key_value` 和 `use_cache` 参数
- 当 `past_key_value` 不为空时，将新 K/V 与历史 K/V 拼接：`torch.cat([past_key_value[0], xk], dim=1)`
- `MiniMindForCausalLM.generate()` 中每步只输入新 token，`past_key_values` 跨步传递

**结论**：minimind 已在模型推理层面实现了 KV Cache，单次推理内部实现 O(n) 加速。

### 2. 应用层缓存 —— 缺失

文件：`scripts/serve_openai_api.py`

OpenAI 兼容的 API 服务端中**没有任何应用层缓存逻辑**。每次请求都会完整走一遍模型推理，没有精确匹配缓存，没有语义缓存。

### 3. SGLang 后端支持

文件：`trainer/rollout_engine.py`

`SGLangRolloutEngine` 提供 `flush_cache()` 接口（第 196 行），可配合 SGLang 的 RadixAttention（前缀缓存）实现 L3 缓存能力，但当前代码未主动调用该接口做缓存管理。

---

## 四、企业级缓存五层架构详解

```
┌─────────────────────────────────────────────────────────┐
│                   用户请求 (Prompt)                        │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ L1: 精确匹配缓存 (MD5/Redis) │  ← 应用层：同一个问题不重复生成
        │ 命中 → 直接返回              │
        └──────────────┬──────────────┘
                       │ 未命中
        ┌──────────────▼──────────────┐
        │ L2: 语义缓存 (Embedding+向量库) │  ← 应用层：意思相近的问题复用答案
        │ 余弦相似度 > 0.85 → 返回      │
        └──────────────┬──────────────┘
                       │ 未命中
        ┌──────────────▼──────────────┐
        │ L3: 提示词/前缀缓存 (Prefix KV)│ ← API/模型服务层：复用长 System Prompt 的计算
        │ (如 Anthropic cache_control) │
        └──────────────┬──────────────┘
                       │ 未命中
        ┌──────────────▼──────────────┐
        │ L4: 模型内部 KV Cache        │ ← 推理引擎层：单次生成内部复用历史 K/V
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ L5: Agent 轨迹缓存           │ ← Agent 层：缓存多步 Thought/Action/Observation
        └─────────────────────────────┘
```

### L1：精确匹配缓存（应用层）

**命中规则**：`hash(prompt + 模型参数 + system_prompt + temperature 等)` 完全相同。

- 优点：100% 准确，延迟极低（毫秒级查表）
- 缺点：哪怕只差一个标点符号，就完全不命中；命中率通常较低（5%–20%）

### L2：语义缓存（应用层）

**命中规则**：
1. 把 Prompt 通过 Embedding 模型（如 `bge-base-zh`, `text-embedding-3-small`）向量化
2. 存入向量数据库（Milvus、Pinecone、pgvector）
3. 新请求向量化后，计算与历史缓存向量的余弦相似度
4. `cos_sim >= 0.85` → 命中，返回历史回答

- 优点：能捕捉 paraphrase（同义改写），命中率可达 40%–70%
- 缺点：需部署 Embedding 模型和向量数据库；有向量化 + 检索延迟；可能出现“过度命中”（意思相近但需要不同回答的问题被误判）

### L3：提示词缓存 / Prompt Caching（API 层）

**核心对象**：System Prompt（系统提示词）

**为什么 System Prompt 特别适合缓存**：
1. **很长**：通常几百到几千 Token（角色设定、代码规范、业务文档、少样本示例）
2. **高度重复**：同一个应用里，成百上千个用户请求共用同一段 System Prompt
3. **很少变动**：同一版本应用里，System Prompt 几乎固定不变

**原理**：服务端把长前缀的 KV 计算结果缓存下来，下次请求只要前缀字节级完全一致，就直接复用，跳过前缀部分的 Attention 计算。

**示例（Anthropic 风格）**：
```python
response = client.messages.create(
    model="claude-3-opus",
    system=[{"type": "text", "text": long_system_prompt, "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": "写个冒泡排序"}],
)
```

**收益**：假设 System Prompt 2000 Token，User Prompt 100 Token，开启缓存后每次只需计算 100 Token，约节省 95% 的输入计算。

**限制**：前缀必须字节级完全一致；通常有最小长度要求（如 1024–2048 Token）；有 TTL 过期时间。

### L4：模型内部 KV Cache（推理引擎层）

单次请求内部，历史 Token 的 K/V 张量复用。

- minimind：已集成
- llm-codegen-demo：类已定义，未集成
- 生产级推理服务（vLLM、TensorRT-LLM）：支持 PagedAttention、连续批处理、动态分页等优化

### L5：Agent 轨迹缓存（Agent 层）

缓存整个推理轨迹：
```
Thought → Action → Observation → Thought → Action → Final Answer
```

第二次遇到相似任务，不需要重新走一遍多步循环，直接返回缓存的轨迹和最终答案。尤其在 RLAIF / Agentic RL 训练中，可避免重复调用昂贵的工具或环境。

---

## 五、System Prompt 在缓存中的关键作用

**是的，System Prompt 在缓存中起到了关键作用，但前提是模型服务支持“提示词缓存”。**

| 场景 | System Prompt 缓存作用 |
|------|----------------------|
| llm-codegen-demo 当前 | 无特殊缓存，System Prompt 概念缺失，每次从头计算 |
| 生产级大模型 API | System Prompt 是缓存核心对象，开启 `cache_control` 后可复用长前缀 KV 计算 |
| 本地推理服务（vLLM/SGLang） | System Prompt 作为公共前缀，自动被前缀缓存（Prefix Caching）复用 |

---

## 六、两个项目对比总结

| 维度 | llm-codegen-demo | minimind |
|------|-----------------|---------|
| 项目定位 | 教育性 Transformer 代码生成 demo | 轻量 LLM 全流程训练项目（64M 参数） |
| L1 精确匹配 | ✅ 有（GenerationCache） | ❌ 无 |
| L2 语义缓存 | ⚠️ 有，但基于关键词 MD5，粗糙 | ❌ 无 |
| L3 提示词缓存 | ❌ 无 | ❌ 无（兼容 SGLang，可切换） |
| L4 模型内部 KV Cache | ⚠️ 类已定义，未集成 | ✅ 已集成到 `generate()` |
| L5 Agent 轨迹缓存 | ❌ 无 | ❌ 无 |
| 推理服务 | 本地 Pipeline 直接调用模型 | FastAPI OpenAI 兼容服务端 |

---

## 七、针对 llm-codegen-demo 的改进建议

### 第一步：给 Pipeline 加上 L1 精确匹配缓存（已有，可优化）

当前已有 `GenerationCache`，但可优化：
- 将缓存 Key 扩展为 `hash(prompt + temperature + max_length + system_prompt)`，避免不同参数下误复用
- 支持 TTL 过期和内存上限控制

### 第二步：把语义匹配从“关键词 MD5”改为“基于 Embedding 的相似度”

```python
class SemanticCache:
    def __init__(self, embedding_model, vector_db, threshold=0.85):
        self.embed = embedding_model
        self.db = vector_db
        self.threshold = threshold

    def get(self, prompt: str):
        q_vec = self.embed.embed(prompt)
        results = self.db.search(vector=q_vec, top_k=3, metric="cosine")
        for hist_prompt, hist_response, sim in results:
            if sim >= self.threshold:
                return hist_response
        return None

    def put(self, prompt: str, response: str):
        vec = self.embed.embed(prompt)
        self.db.insert(vector=vec, payload={"prompt": prompt, "response": response})
```

### 第三步：把 KVCache 真正集成到 `TransformerModel.generate_step()`

当前 `generate_step()` 每次都传入完整 `generated` 序列，需修改为：
- 维护每层 Decoder 的 K/V 缓存张量
- 每步只输入新 token，拼接到缓存中
- 实现真正的 O(n) 推理加速

### 第四步：引入提示词缓存（L3）

如果接入第三方 API（如 Anthropic、Moonshot），在 `system` 参数中开启 `cache_control`；若自建推理服务，切换到支持前缀缓存的后端（vLLM / SGLang）。

---

## 八、快速验证：缓存是否命中

在 llm-codegen-demo 中验证：

```python
from scripts.pipeline import CodeGenerationPipeline

pipeline = CodeGenerationPipeline(preset='tiny')

# 第一次：缓存未命中，走模型生成
r1 = pipeline.generate("public class UserService", max_length=30, verbose=False)
print(r1['source'])  # 'generated'

# 第二次：prompt 完全相同，缓存命中
r2 = pipeline.generate("public class UserService", max_length=30, verbose=False)
print(r2['source'])  # 'cache'

# 不同 prompt：不会命中
r3 = pipeline.generate("帮我写一个冒泡排序java版本", max_length=30, verbose=False)
print(r3['source'])  # 'generated'
```

---

## 九、参考文件

- `llm-codegen-demo/scripts/optimization/cache.py`：GenerationCache 实现
- `llm-codegen-demo/scripts/optimization/kv_cache.py`：KVCache 定义
- `llm-codegen-demo/scripts/pipeline.py`：Pipeline 缓存调用逻辑
- `minimind/model/model_minimind.py`：KV Cache 集成实现
- `minimind/scripts/serve_openai_api.py`：API 服务端（当前无应用层缓存）
- `minimind/trainer/rollout_engine.py`：SGLang Rollout 引擎（含 flush_cache 接口）