#!/usr/bin/env python3
"""
第1天学习：LLM代码生成完整流程（教学版）

用途：系统性学习LLM从输入到输出的完整过程
特点：
  - 结构化的章节设计，循序渐进
  - ASCII流程图直观展示整体架构
  - 对比实验表格展示参数影响
  - 源码链接提示引导深入探索
  - 学习要点、核心概念、思考题、实践建议

适用场景：
  - 初学者系统学习LLM原理
  - 理解每个模块的作用和关系
  - 掌握关键概念和工作流程
  - 获得实践指导和下一步建议

运行方式：
  python day1_demo.py

注意：此脚本关闭了debug模式，输出简洁明了，适合学习
"""

import sys
import os

# 将scripts目录添加到Python路径
script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, script_dir)

from pipeline import CodeGenerationPipeline
from logger import logging_context


def print_section(title, level=1):
    """打印带分隔线的章节标题"""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    elif level == 2:
        print("\n" + "-" * 80)
        print(f"  {title}")
        print("-" * 80)
    else:
        print(f"\n  ▶ {title}")


def print_learning_point(point, explanation=None):
    """打印学习要点"""
    print(f"\n  💡 学习要点: {point}")
    if explanation:
        print(f"     {explanation}")


def print_key_concept(concept, description):
    """打印核心概念"""
    print(f"\n  📚 核心概念: {concept}")
    print(f"     {description}")


def print_code_link(module, line_range, description=""):
    """打印源码链接提示"""
    print(f"     📖 查看源码: scripts/{module}.py (第{line_range}行)")
    if description:
        print(f"        {description}")


def main():
    """主函数"""
    
    # ==================== 介绍部分 ====================
    print_section("第1天学习：LLM代码生成完整流程")
    
    print("""
    本演示将展示大语言模型从输入到输出的完整过程。
    
    学习目标：
      ✓ 理解Token化的作用和过程
      ✓ 掌握Transformer架构的基本组件
      ✓ 了解自回归生成的工作原理
      ✓ 认识后处理的重要性
      ✓ 学会使用缓存机制提升性能
    
    让我们开始吧！🚀
    """)
    
    # 添加流程图
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    LLM代码生成完整流程                            │
    └─────────────────────────────────────────────────────────────────┘
    
    输入文本 → Token化 → Transformer推理 → 采样 → 后处理 → 输出代码
       │          │           │              │         │
       │          ├─ 分割文本  ├─ Encoder     ├─ 选择   ├─ 语法验证
       │          ├─ 转ID     ├─ Decoder     ├─ Top-K  ├─ 格式化
       │          └─ Mask     └─ 注意力      └─ Temp   ├─ Import
       │                                              └─ 优化
       └──────────────────────────────────────────────┘
    """)
    
    # ==================== Step 1: 初始化 ====================
    print_section("Step 1: 初始化模型管道", level=2)
    
    print_learning_point(
        "模块化设计",
        "整个系统由多个独立模块组成：Tokenizer、Transformer、Generator等"
    )
    print_code_link("tokenizer", "1-50", "SimpleTokenizer类定义")
    print_code_link("pipeline", "16-70", "CodeGenerationPipeline类定义")
    
    print("\n  正在创建管道...")
    pipeline = CodeGenerationPipeline(
        vocab_size=1000,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        debug_mode=False  # 关闭debug模式，减少冗余输出
    )
    
    print("\n  ✅ 管道初始化完成")
    print(f"     - 词汇表大小: 1000")
    print(f"     - 模型维度: 128")
    print(f"     - 注意力头数: 8")
    print(f"     - Encoder层数: 2")
    print(f"     - Decoder层数: 2")
    print(f"     - 参数量: ~1.18M")
    
    print_key_concept(
        "模型参数",
        "参数量决定了模型的表达能力。1.18M参数对于演示来说足够小，便于理解。"
    )
    
    # ==================== Step 2: Token化 ====================
    print_section("Step 2: Token化 - 将文本转换为数字", level=2)
    
    prompt = "public class UserService"
    print(f"\n  输入文本: '{prompt}'")
    
    print_learning_point(
        "Token化是什么？",
        "LLM不能直接处理文本，需要将文本分割成小单元（tokens），然后转换为数字ID"
    )
    
    print("\n  正在进行Token化...")
    from tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    token_ids, mask = tokenizer.encode(prompt, max_length=32)
    
    print(f"\n  ✅ Token化完成")
    print(f"     - Token IDs: {token_ids[:5]}... (共{len(token_ids)}个)")
    print(f"     - Attention Mask: {mask[:5]}...")
    
    decoded = tokenizer.decode(token_ids)
    print(f"     - 解码结果: '{decoded}'")
    
    print_key_concept(
        "Attention Mask",
        "Mask用于告诉模型哪些位置是真实token，哪些是padding。1表示真实，0表示填充。"
    )
    print_code_link("tokenizer", "150-200", "encode方法中的mask生成逻辑")
    
    print_learning_point(
        "UNK Token",
        f"注意：'{prompt}' 中可能有词汇表中没有的词，会被替换为 <UNK>"
    )
    
    # ==================== Step 3: Transformer推理 ====================
    print_section("Step 3: Transformer推理 - 核心计算", level=2)
    
    print("""
    Transformer是LLM的核心架构，包含两个主要部分：
    
    Encoder（编码器）:
      - 理解输入序列的含义
      - 捕捉词与词之间的关系
      - 生成上下文表示
    
    Decoder（解码器）:
      - 基于Encoder的输出逐步生成新token
      - 每次只生成一个token（自回归）
      - 使用因果掩码防止看到未来
    """)
    
    print_learning_point(
        "多头注意力",
        "8个注意力头从不同角度理解序列关系，就像用8种不同的方式阅读同一篇文章"
    )
    print_code_link("attention", "50-150", "MultiHeadAttention实现")
    print_code_link("transformer", "200-300", "EncoderLayer和DecoderLayer")
    
    print("\n  开始生成代码...")
    print(f"  Prompt: '{prompt}'")
    print(f"  最大长度: 50 tokens")
    print(f"  Temperature: 0.7\n")
    
    result = pipeline.generate(
        prompt=prompt,
        max_length=50,
        temperature=0.7,
        verbose=False  # 关闭verbose，我们自己控制输出
    )
    
    print("\n  ✅ 生成完成")
    print(f"     - 生成Token数: {result['token_count']}")
    print(f"     - 耗时: {result['processing_time']:.4f}秒")
    print(f"     - 速度: {result['token_count']/result['processing_time']:.2f} tokens/秒")
    
    # ==================== Step 4: 分析生成结果 ====================
    print_section("Step 4: 分析生成结果", level=2)
    
    print("\n  原始生成的代码:")
    print(f"  {result['raw_code'][:200]}...")
    
    print_learning_point(
        "为什么有很多<UNK>？",
        "因为模型未经过训练，权重是随机的，无法准确预测下一个token。\n"
        "实际应用中，模型需要在大量数据上训练才能生成有意义的代码。"
    )
    
    print_key_concept(
        "自回归生成",
        "模型一次只生成一个token，然后将新生成的token加入输入，继续生成下一个。\n"
        "这就像写文章时一个字一个字地写，写完前面的才能决定后面的。"
    )
    print_code_link("generator", "100-200", "CodeGenerator的generate方法")
    
    # ==================== Step 5: 后处理 ====================
    print_section("Step 5: 后处理 - 提高代码质量", level=2)
    
    print("\n  后处理包括以下步骤：")
    print("     1. 语法验证 - 检查括号匹配、基本语法")
    print("     2. 代码格式化 - 自动缩进、换行")
    print("     3. Import管理 - 添加必要的import语句")
    print("     4. 代码优化 - 改进代码结构")
    
    print(f"\n  处理后的代码:")
    processed = result['processed_code']
    lines = processed.split('\n')
    for i, line in enumerate(lines[:15], 1):  # 只显示前15行
        print(f"     {i:2d}: {line}")
    if len(lines) > 15:
        print(f"     ... (还有{len(lines)-15}行)")
    
    print_learning_point(
        "后处理的重要性",
        "即使模型生成了不完美的代码，后处理器也能修复一些常见问题，\n"
        "如添加缺失的import、格式化代码等。"
    )
    print_code_link("postprocessor", "1-100", "CodePostProcessor类定义")
    
    # ==================== Step 6: 缓存机制 ====================
    print_section("Step 6: 缓存机制 - 性能优化", level=2)
    
    print("\n  缓存可以避免重复计算相同的prompt")
    print(f"  当前缓存状态: {len(pipeline.cache.cache)}/{pipeline.cache.max_size}")
    
    print("\n  尝试再次生成相同的内容...")
    result2 = pipeline.generate(
        prompt=prompt,
        max_length=50,
        temperature=0.7,
        verbose=False
    )
    
    print(f"\n  ✅ 第二次生成完成")
    print(f"     - 来源: {result2['source']}")
    print(f"     - 耗时: {result2['processing_time']:.4f}秒")
    
    if result2['source'] == 'cache':
        speedup = result['processing_time'] / result2['processing_time']
        print(f"     - 加速比: {speedup:.1f}x ⚡")
        
        print_learning_point(
            "缓存的威力",
            f"对于相同的请求，缓存可以将速度提升{speedup:.0f}倍！\n"
            f"这在生产环境中非常重要，可以大幅降低计算成本。"
        )
    
    # ==================== Step 7: 对比实验 ====================
    print_section("Step 7: 对比实验 - 参数影响分析", level=2)
    
    print("\n  让我们看看不同参数对生成的影响...")
    
    # 实验1: 不同vocab_size的UNK比例
    print("\n  📊 实验1: 词汇表大小对UNK比例的影响")
    print("     " + "-" * 70)
    
    test_texts = ["public class UserService", "def hello_world():", "import numpy as np"]
    vocab_sizes = [100, 500, 1000, 2000]
    
    print(f"     {'词汇表大小':<12} {'UNK数量':<10} {'UNK比例':<10} {'说明'}")
    print(f"     {'-'*12} {'-'*10} {'-'*10} {'-'*30}")
    
    for vocab_size in vocab_sizes:
        test_tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        total_unk = 0
        total_tokens = 0
        
        for text in test_texts:
            ids, _ = test_tokenizer.encode(text, max_length=32)
            unk_count = sum(1 for token_id in ids if test_tokenizer.get_token_info(token_id)['token'] == '<UNK>')
            total_unk += unk_count
            total_tokens += len([tid for tid in ids if tid != 0])  # 排除PAD
        
        unk_ratio = total_unk / max(total_tokens, 1) * 100
        note = "较小" if vocab_size <= 100 else "适中" if vocab_size <= 1000 else "较大"
        print(f"     {vocab_size:<12} {total_unk:<10} {unk_ratio:<9.1f}% {note}")
    
    print_learning_point(
        "词汇表大小的权衡",
        "词汇表越大，UNK越少，但模型参数量会增加。需要在覆盖率和效率之间平衡。"
    )
    print_code_link("tokenizer", "30-80", "词汇表构建逻辑")
    
    # 实验2: 不同temperature的生成多样性
    print("\n  📊 实验2: Temperature对生成多样性的影响")
    print("     " + "-" * 70)
    
    temperatures = [0.1, 0.5, 0.7, 1.0, 2.0]
    print(f"     {'Temperature':<12} {'确定性':<10} {'多样性':<10} {'适用场景'}")
    print(f"     {'-'*12} {'-'*10} {'-'*10} {'-'*25}")
    
    temp_descriptions = [
        (0.1, "极高", "极低", "代码生成、翻译（需要准确）"),
        (0.5, "高", "低", "事实性问答"),
        (0.7, "中等", "中等", "通用场景（推荐）"),
        (1.0, "标准", "标准", "原始概率分布"),
        (2.0, "低", "高", "创意写作、头脑风暴"),
    ]
    
    for temp, determinism, diversity, scenario in temp_descriptions:
        marker = " ← 当前" if temp == 0.7 else ""
        print(f"     {temp:<12.1f} {determinism:<10} {diversity:<10} {scenario}{marker}")
    
    print_learning_point(
        "Temperature的作用原理",
        "Temperature通过缩放logits来改变概率分布：\n"
        "  - T < 1: 放大高概率，抑制低概率 → 更确定\n"
        "  - T > 1: 平滑概率分布 → 更随机"
    )
    print_code_link("generator", "150-250", "Temperature采样实现")
    
    # ==================== 总结 ====================
    print_section("📝 今日学习总结", level=1)
    
    print("""
    今天我们学习了LLM代码生成的完整流程：
    
    1️⃣  Token化: 文本 → Tokens → Token IDs
    2️⃣  Transformer: Encoder理解输入，Decoder逐步生成
    3️⃣  自回归生成: 一次一个token，循环直到达到最大长度
    4️⃣  后处理: 修复语法、格式化、添加import
    5️⃣  缓存: 避免重复计算，提升性能
    
    关键收获：
      ✓ 理解了每个模块的作用和相互关系
      ✓ 知道了为什么需要后处理
      ✓ 学会了使用缓存优化性能
      ✓ 认识到训练对模型质量的重要性
    
    思考题：
      1. 如果增大vocab_size，<UNK>的数量会如何变化？
      2. Temperature参数对生成结果有什么影响？
      3. 为什么Decoder需要使用因果掩码？
      4. 除了缓存，还有哪些方法可以提升生成速度？
    
    明日预告：
      第2天我们将深入探索Token化和注意力机制的实现细节！
    """)
    
    print_section("🎯 实践建议", level=1)
    
    print("""
    建议你尝试以下实验：
    
    1. 修改vocab_size观察UNK比例的变化
       python -c "from scripts.tokenizer import SimpleTokenizer; t=SimpleTokenizer(500); print(t.encode('test'))"
    
    2. 调整temperature观察生成多样性的变化
       在day1_demo.py中修改temperature参数（0.1, 0.5, 1.0, 2.0）
    
    3. 查看源代码理解实现细节
       - scripts/tokenizer.py: Token化实现
       - scripts/attention.py: 注意力机制
       - scripts/transformer.py: Transformer架构
    
    4. 运行其他演示
       python scripts/main.py  # 交互式菜单
       python scripts/tests/debug_tokenizer.py  # Tokenizer调试
    """)


if __name__ == "__main__":
    # 使用日志上下文管理器，自动创建日志文件
    with logging_context(__file__):
        main()
