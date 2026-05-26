2026年重构复活后的 **OpenAI Codex**（Rust 架构的本地智能体）在操作哲学上更强调“确定性规划”**和**“沙箱安全性”。它的命令行设计非常硬朗。

如果全盘使用 Codex 来从零开发这个带缓存的通知系统，你可以直接复制并保存下面这份 `CODEX_OPERATIONS_SOP.md` 规范。

---

# 📖 OpenAI Codex 新功能开发标准作业程序 (SOP)

> **导言：** 本指南规范了使用 OpenAI Codex CLI 独立构建新功能时的生命周期。Codex 的核心武器是 **Skills（技能树系统）**、**Sandbox（安全沙箱机制）** 以及 **Strict Planning（强规划约束）**。通过本流程，可以确保新功能在完全隔离的安全环境中编写并经过高强度自审。

---

## 🛑 阶段一：强制规划与意图锚定（Plan 阶段）

* **目标：** 激活 Codex 的 `create-plan` 技能，在动手前锁定所有文件变更范围。
* **最佳水位：** 初始状态（~5%）

### 1. 触发指令

在终端中执行 Codex 的强规划命令：

```bash
codex plan "我要开发一个带 Redis 缓存的用户通知系统。请启用 create-plan 技能，生成完整的设计清单与新建文件树。"

```

### 2. 核心原子动作

* **规划拦截（Plan Gate）：** Codex 收到指令后，不会立刻修改代码，而是在终端输出一个交互式清单（包含拟创建的文件路径和契约）。
* **开发者确认：** 检查清单无误后，在终端输入 `y`。Codex 会将该设计方案写入本地缓存，作为后续步骤的最高行为准则，防止中途频繁跑偏。

---

## 💻 阶段二：沙箱隔离实现与代码填肉（Build 阶段）

* **目标：** 利用 Codex 极速的 Rust 引擎，在隔离的本地沙箱中批量生成骨架。
* **最佳水位：** ~15%

### 1. 触发指令

要求 Codex 挂载沙箱并开始开荒：

```bash
codex run --sandbox "根据刚才的 plan，在沙箱环境中批量创建并实现 src/types/notification.ts、src/services/cache.ts 和 src/services/storage.ts。"

```

### 2. 核心原子动作

* **`--sandbox`（沙箱环境）：** Codex 会在本地拉起一个轻量级的安全隔离层，所有的文件写入和初始运行都在沙箱内进行，绝不污染你当前的主分支代码。
* **高速生成：** 得益于底层的 Rust 运行时，Codex 的文件批量生成和模板填充速度极快。

---

## 🧪 阶段三：技能联动验证与自愈（Test 阶段）

* **目标：** 调用本地工具链跑通测试，让 Codex 自动吞掉报错并完成修复。
* **最佳水位：** ~35%

### 1. 触发指令

让 Codex 在沙箱内直接调用你的本地测试指令：

```bash
codex exec "npm run test && npx tsc --noEmit"

```

### 2. 核心原子动作

* **`codex exec`：** 该命令允许 Codex 读取本地终端的完整输出。如果检测到 `FAIL` 或类型报错，Codex 的自动修复引擎（Self-Healing）会就地启动，重新编辑沙箱内对应的文件，直到整个测试套件完全变绿（PASS）。

---

## ✂️ 阶段四：设置检查点与上下文剪枝（Prune 阶段）

* **目标：** 丢弃长对话中累积的冗余编译日志，复位 Token 水位。
* **触发时机：** **当终端交互多轮，感觉模型开始出现重复话术或速度变慢时（约 60% 水位）。**

### 1. 触发指令

执行带参数的剪枝与检查点同步命令：

```bash
codex prune --keep "notificationManager.ts, cache.ts" --checkpoint "通知系统基础逻辑通过编译"

```

### 2. 核心原子动作

* **`--keep` 与 `prune`：** 清空当前会话中已经无用的终端报错历史，但强行在内存中置顶（Pin）核心文件的最新逻辑状态。
* **`--checkpoint`：** 将当前的沙箱代码阶段性固化。此时 Codex 会对上下文进行大幅度脱水压缩，可用 Token 瞬间复位到极佳状态。

---

## 🕵️ 阶段五：缺陷高压审计与安全防护（Audit 阶段）

* **目标：** 压榨 Codex 对工业级并发、安全漏洞的审计能力，封死高并发死角。
* **最佳水位：** ~20%（剪枝复位后）

### 1. 触发指令

调用高级审计技能：

```bash
codex audit src/services/notificationManager.ts --level high --focus "concurrency, cache-stampede"

```

### 2. 核心原子动作

* **`codex audit`：** 这是 2026 版 Codex 的王牌技能之一。它会站在彻头彻尾的黑客视角对代码实施静态与动态模拟审计。
* *结果展示：* 它会非常敏锐地指出在高并发用户涌入时，缓存失效可能导致的“缓存击穿（Cache Stampede）”，并强行阻断交付，在代码中注入互斥锁（Promise Lock / Singleflight 机制）补丁。

---

## 📦 阶段六：沙箱合并与 Git 归档（Ship 阶段）

* **目标：** 将完美的沙箱成果合并回主项目，安全收工。

### 1. 触发指令

```bash
codex ship --commit "feat: implement notification system with redis cache and mutex protection"

```

### 2. 核心原子动作

* **`codex ship`：** 将全绿、通过审计的沙箱代码正式写回你的物理本地磁盘，同步结束沙箱进程，并自动为你撰写并提交一条完美的 Git Commit。

---

## 🛠️ OpenAI Codex 核心命令速查表

| 指令 (Command) | 核心功能 | 开发者心法 |
| --- | --- | --- |
| **`codex plan --create-plan`** | 架构前置锁定 | 绝不盲目开干。先生成变更清单，按图索骥。 |
| **`codex run --sandbox`** | 隔离沙箱开荒 | 脏活累活全进沙箱，编译失败也绝不污染主干代码。 |
| **`codex exec "<cmd>"`** | 终端工具链注入 | 授权 Codex 听取终端回显，利用自愈引擎死磕 Bug。 |
| **`codex prune --checkpoint`** | **手动内存复位** | **水位过半必用。斩断冗余日志，用 Checkpoint 锁死阶段性胜利。** |
| **`codex audit`** | 工业级漏洞找茬 | 封死并发漏洞的终极大闸，专抓缓存击穿与死锁。 |