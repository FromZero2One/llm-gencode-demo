"""
Tokenizer Debug 工具
提供详细的 tokenization 过程可视化和调试功能

主要用途：
- 逐步展示文本如何被转换为 token IDs
- 可视化 Attention Mask 的生成过程
- 分析词汇表覆盖率和 UNK token 比例
- 对比不同文本的 tokenization 效果
- 测试边界情况（空字符串、中文、纯符号等）

适用场景：
- 学习 Transformer 模型的输入处理流程
- 调试 tokenization 相关问题
- 优化词汇表设计
- 教学演示
"""

import sys
import os
import logging
from typing import List, Dict

# 添加父目录（scripts）到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import SimpleTokenizer
from logger import logging_context

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class TokenizerDebugger:
    """
    Tokenizer 调试器 - 提供详细的分词过程分析
    
    功能特性：
    - 9步详细分析流程：从原始文本到最终的 token IDs 和 attention mask
    - 词汇表统计分析：分类统计特殊token、关键字、类型等
    - 多文本对比：批量比较不同代码片段的 tokenization 效果
    - 边界测试：验证极端情况下的行为
    
    使用示例：
        >>> debugger = TokenizerDebugger(vocab_size=1000)
        >>> result = debugger.debug_tokenize("public class User")
        >>> debugger.debug_vocab()
        >>> debugger.compare_texts(["text1", "text2"])
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        初始化调试器
        
        Args:
            vocab_size (int): 词汇表大小，默认为1000。与SimpleTokenizer保持一致。
        """
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size, debug_mode=True)
        
    def debug_tokenize(self, text: str, show_details: bool = True) -> Dict:
        """
        详细调试 tokenization 过程
        
        执行9个步骤的完整分析：
        1. 显示原始输入文本和基本信息
        2. 分词过程（tokenize）
        3. 添加特殊 tokens（BOS/EOS）
        4. Token → ID 映射（识别已知和未知tokens）
        5. 序列处理（截断或填充到max_length）
        6. 创建 Attention Mask（标记有效内容和padding）
        7. 统计分析（词汇表覆盖率、UNK比例等）
        8. 显示未知 tokens（需要添加到词汇表）
        9. 解码验证（检查编码-解码是否一致）
        
        Args:
            text (str): 待分析的输入文本
            show_details (bool): 是否显示详细信息（如每个token的列表）
                                设为False时只显示摘要，适合批量对比
            
        Returns:
            Dict: 包含所有调试信息的字典，包括：
                - input_text: 原始输入文本
                - input_length: 输入文本长度（字符数）
                - tokens: 分词后的token列表
                - token_ids: 编码后的token ID序列（含padding）
                - attention_mask: 注意力掩码（1=有效，0=padding）
                - unk_tokens: 未知token列表（不在词汇表中）
                - known_tokens: 已知token列表（在词汇表中）
                - statistics: 统计信息字典
                    * total_tokens: 总token数
                    * known_count: 已知token数
                    * unk_count: 未知token数
                    * unk_rate: UNK比例（%）
                    * vocab_coverage: 词汇表覆盖率（%）
        """
        print("\n" + "="*80)
        print(" " * 25 + "TOKENIZER DEBUG 分析")
        print("="*80)
        
        result = {
            'input_text': text,
            'input_length': len(text),
            'tokens': [],
            'token_ids': [],
            'attention_mask': [],
            'unk_tokens': [],
            'known_tokens': [],
            'statistics': {}
        }
        
        # Step 1: 显示原始文本
        print(f"\n[Step 1] 原始输入文本")
        print(f"  文本: \"{text}\"")
        print(f"  长度: {len(text)} 字符")
        print(f"  词汇表大小: {len(self.tokenizer.vocab)}")
        
        # Step 2: Tokenize
        print(f"\n[Step 2] 分词过程 (tokenize)")
        tokens = self.tokenizer.tokenize(text)
        result['tokens'] = tokens
        
        print(f"  分词数量: {len(tokens)} tokens")
        if show_details:
            print(f"  Tokens 列表:")
            for i, token in enumerate(tokens, 1):
                print(f"    [{i:2d}] '{token}'")
        
        # Step 3: 添加特殊 tokens
        print(f"\n[Step 3] 添加特殊 Tokens")
        special_tokens = [self.tokenizer.BOS_TOKEN] + tokens + [self.tokenizer.EOS_TOKEN]
        print(f"  BOS (开始): {self.tokenizer.BOS_TOKEN}")
        print(f"  EOS (结束): {self.tokenizer.EOS_TOKEN}")
        print(f"  总 tokens (含特殊): {len(special_tokens)}")
        
        # Step 4: 转换为 IDs
        print(f"\n[Step 4] Token → ID 映射")
        token_ids = []
        unk_tokens = []
        known_tokens = []
        
        for i, token in enumerate(special_tokens):
            if token in self.tokenizer.vocab:
                token_id = self.tokenizer.vocab[token]
                known_tokens.append(token)
                status = "✓"
            else:
                token_id = self.tokenizer.vocab[self.tokenizer.UNK_TOKEN]
                unk_tokens.append(token)
                status = "✗ UNK"
            
            token_ids.append(token_id)
            
            if show_details and i < 30:  # 只显示前30个
                print(f"    [{i:2d}] '{token:20s}' → ID: {token_id:4d} {status}")
        
        if len(special_tokens) > 30:
            print(f"    ... ({len(special_tokens) - 30} more tokens)")
        
        result['token_ids'] = token_ids
        result['unk_tokens'] = unk_tokens
        result['known_tokens'] = known_tokens
        
        # Step 5: 截断或填充（序列长度标准化）
        # 
        # 💡 max_length 说明：
        # - 这里硬编码为64，是教学演示用的简化值
        # - 本质上就是大模型的"上下文长度"概念
        # - 真实模型：GPT-3(2K), GPT-4(8K-128K), Claude(100K)
        # - 本代码使用较小值是为了便于观察和调试
        max_length = 64
        print(f"\n[Step 5] 序列处理 (max_length={max_length})")
        original_length = len(token_ids)
        
        if len(token_ids) > max_length:
            # 截断：超过max_length时丢弃多余tokens
            # 注意：保留BOS在开头，确保EOS在末尾
            print(f"  ⚠  需要截断: {original_length} → {max_length}")
            token_ids = token_ids[:max_length-1] + [self.tokenizer.vocab[self.tokenizer.EOS_TOKEN]]
        elif len(token_ids) < max_length:
            # 填充：不足max_length时用PAD tokens补齐
            # 原因：批量处理时需要统一长度的序列
            padding_count = max_length - len(token_ids)
            print(f"  ℹ  需要填充: {original_length} + {padding_count} PAD tokens")
            pad_id = self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]
            token_ids.extend([pad_id] * padding_count)
        else:
            print(f"  ✓ 长度正好: {len(token_ids)}")
        
        # Step 6: 创建 Attention Mask（注意力掩码）
        #
        # 💡 Attention Mask 核心概念：
        # - 作用：标记哪些位置是有效内容(1)，哪些是padding(0)
        # - 原理：在Transformer的Self-Attention中，mask=0的位置会被加上极大负数(-1e9)
        #         softmax后这些位置的权重趋近于0，模型不会关注padding
        # - 必要性：批量处理不同长度文本时，确保模型只关注有效tokens
        #
        # 计算公式：
        #   attention_mask = [1] * 有效token数 + [0] * padding数
        #
        # 示例：
        #   text = "return" → 3个tokens (BOS + return + EOS)
        #   max_length = 10
        #   token_ids =      [2, 19, 1, 0, 0, 0, 0, 0, 0, 0]
        #                    ↑^^^有效^^^↑↑^^^^padding^^^^↑
        #   attention_mask = [1, 1,  1, 0, 0, 0, 0, 0, 0, 0]
        #                    ↑^^^有效=1^↑↑^^padding=0^^^↑
        #
        # 为什么需要 Attention Mask？
        #   1. 批量处理：同时处理多个不同长度的文本
        #   2. 提高效率：避免模型浪费计算资源在padding上
        #   3. 保证准确性：防止padding影响模型的注意力分布
        attention_mask = [1] * min(original_length, max_length) + [0] * max(0, max_length - min(original_length, max_length))
        #                ↑^^^^^有效tokens（mask=1）^^^^^↑↑^^^^^padding（mask=0）^^^^^↑
        result['attention_mask'] = attention_mask
        
        print(f"\n[Step 6] Attention Mask")
        print(f"  有效 tokens: {sum(attention_mask)}")  # mask中1的数量 = 有效token数
        print(f"  Padding tokens: {len(attention_mask) - sum(attention_mask)}")  # mask中0的数量 = padding数
        if show_details:
            # 可视化显示：1表示有效，0表示padding
            mask_str = ''.join(['1' if m == 1 else '0' for m in attention_mask[:64]])
            print(f"  Mask (前64位): {mask_str}")
            # 示例输出：111110000000000000... (前5个有效，后面都是padding)
        
        # Step 7: 统计分析
        # 计算词汇表覆盖率和UNK比例，评估tokenizer的质量
        print(f"\n[Step 7] 统计分析")
        stats = {
            'total_tokens': len(tokens),  # 原始分词数量（不含BOS/EOS）
            'known_count': len(known_tokens) - 2,  # 排除BOS和EOS后的已知token数
            'unk_count': len(unk_tokens),  # 未知token数量（会被替换为<UNK>）
            'unk_rate': len(unk_tokens) / max(len(special_tokens), 1) * 100,  # UNK占比
            'vocab_coverage': len(known_tokens) / max(len(special_tokens), 1) * 100  # 词汇表覆盖率
        }
        result['statistics'] = stats
        
        print(f"  总 token 数: {stats['total_tokens']}")
        print(f"  已知 tokens: {stats['known_count']}")
        print(f"  未知 tokens (UNK): {stats['unk_count']}")
        print(f"  词汇表覆盖率: {stats['vocab_coverage']:.1f}%")
        print(f"  UNK 比例: {stats['unk_rate']:.1f}%")
        # 💡 提示：如果UNK比例过高，说明需要扩展词汇表
        
        # Step 8: 显示 UNK tokens
        if unk_tokens:
            print(f"\n[Step 8] ⚠  未知 Tokens (需要添加到词汇表)")
            for i, token in enumerate(unk_tokens, 1):
                print(f"    [{i}] '{token}'")
        
        # Step 9: 解码验证
        print(f"\n[Step 9] 解码验证 (decode)")
        decoded = self.tokenizer.decode(token_ids)
        print(f"  解码结果: \"{decoded}\"")
        print(f"  原始 vs 解码: {'✓ 匹配' if decoded.replace(' ', '') == text.replace(' ', '') else '✗ 不匹配'}")
        
        print("\n" + "="*80)
        print(" " * 30 + "DEBUG 完成")
        print("="*80 + "\n")
        
        return result
    
    def debug_vocab(self, show_sample: bool = True) -> Dict:
        """
        调试词汇表 - 分析词汇表的组成和结构
        
        功能：
        - 统计词汇表大小和ID范围
        - 分类统计：特殊tokens、Java关键字、Java类型等
        - 显示前50个tokens示例及其分类
        
        Args:
            show_sample (bool): 是否显示词汇表示例，默认为True
            
        Returns:
            Dict: 包含词汇表信息的字典
                - vocab_size: 词汇表大小
                - special_tokens: 特殊token列表（如<PAD>, <BOS>等）
                - sample: 前50个token的示例字典（如果show_sample=True）
        
        用途：
        - 检查词汇表是否包含必要的tokens
        - 评估词汇表的覆盖范围
        - 发现缺失的重要tokens
        """
        print("\n" + "="*80)
        print(" " * 25 + "词汇表 (VOCABULARY) 分析")
        print("="*80)
        
        vocab = self.tokenizer.vocab
        reverse_vocab = self.tokenizer.reverse_vocab
        
        print(f"\n词汇表统计:")
        print(f"  总大小: {len(vocab)} tokens")
        print(f"  最大 ID: {max(vocab.values())}")
        print(f"  最小 ID: {min(vocab.values())}")
        
        # 分类统计
        special_tokens = [t for t in vocab.keys() if t.startswith('<')]
        keywords = [t for t in vocab.keys() if t in ['public', 'private', 'class', 'return', 'if', 'else']]
        java_types = [t for t in vocab.keys() if t in ['int', 'String', 'boolean', 'void', 'List', 'Map']]
        
        print(f"\n词汇表组成:")
        print(f"  特殊 tokens: {len(special_tokens)}")
        print(f"  Java 关键字: {len(keywords)}")
        print(f"  Java 类型: {len(java_types)}")
        
        if show_sample:
            print(f"\n词汇表示例 (前50个):")
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            for i, (token, token_id) in enumerate(sorted_vocab[:50]):
                category = "SPECIAL" if token.startswith('<') else "KEYWORD" if token in keywords else "TYPE" if token in java_types else "OTHER"
                print(f"    [{token_id:3d}] '{token:20s}' ({category})")
        
        print("\n" + "="*80 + "\n")
        
        return {
            'vocab_size': len(vocab),
            'special_tokens': special_tokens,
            'sample': dict(sorted_vocab[:50]) if show_sample else {}
        }
    
    def compare_texts(self, texts: List[str]):
        """
        比较多个文本的 tokenization 结果
        
        功能：
        - 批量处理多个文本，使用debug_tokenize（简化模式）
        - 生成对比表格，显示每个文本的关键指标
        - 便于评估不同代码片段的tokenization效果
        
        Args:
            texts (List[str]): 待对比的文本列表
            
        Returns:
            List[Dict]: 每个文本的调试结果列表
            
        输出示例：
            文本                                       Tokens   UNK      覆盖率     
            --------------------------------------------------------------------------------
            public class User                          3        0        100.0     %
            private void process                       3        1        66.7      %
            
        用途：
        - 比较不同代码风格的tokenization效率
        - 发现哪些类型的代码产生更多UNK tokens
        - 优化词汇表以提高覆盖率
        """
        print("\n" + "="*80)
        print(" " * 20 + "多文本对比分析")
        print("="*80)
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n{'─'*80}")
            print(f"文本 {i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            print(f"{'─'*80}")
            result = self.debug_tokenize(text, show_details=False)
            results.append(result)
        
        # 对比总结
        print(f"\n{'='*80}")
        print("对比总结")
        print(f"{'='*80}")
        print(f"{'文本':<40} {'Tokens':<8} {'UNK':<8} {'覆盖率':<10}")
        print(f"{'-'*80}")
        for i, result in enumerate(results, 1):
            text_preview = result['input_text'][:37] + '...' if len(result['input_text']) > 40 else result['input_text']
            print(f"{text_preview:<40} {result['statistics']['total_tokens']:<8} "
                  f"{result['statistics']['unk_count']:<8} {result['statistics']['vocab_coverage']:<10.1f}%")
        
        print("\n" + "="*80 + "\n")
        
        return results


def main():
    """
    主函数 - 运行所有 debug 测试
    
    执行4个完整的测试场景：
    
    测试1: 基本 Tokenization
    - 使用完整的Java方法作为输入
    - 展示9步详细分析流程（show_details=True）
    - 输出每个步骤的详细信息
    
    测试2: 词汇表分析
    - 显示词汇表的统计信息
    - 分类展示特殊tokens、关键字、类型等
    - 列出前50个tokens示例
    
    测试3: 多文本对比
    - 对比5种不同的Java代码片段
    - 生成对比表格，显示Tokens数、UNK数、覆盖率
    - 评估不同代码模式的tokenization效果
    
    测试4: 边界情况
    - 空字符串：测试tokenizer对空输入的处理
    - 单字符：测试最小输入
    - 中文：测试非ASCII字符（会产生UNK）
    - 变量名：测试数字和字母组合
    - 纯符号：测试特殊字符序列
    
    运行方式：
        python debug_tokenizer.py
    """
    
    # 创建 debugger 实例，使用1000大小的词汇表
    debugger = TokenizerDebugger(vocab_size=1000)
    
    # 测试1: 基本 tokenization - 完整展示9步分析流程
    print("\n" + "█"*80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "测试1: 基本 Tokenization" + " " * 30 + "█")
    print("█" + " " * 78 + "█")
    print("█"*80)
    
    test_text = "public class UserService { public User findById(Long id) { return userRepository.findById(id).orElse(null); } }"
    result = debugger.debug_tokenize(test_text, show_details=True)
    
    # 测试2: 词汇表分析 - 查看词汇表组成和结构
    print("\n" + "█"*80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "测试2: 词汇表分析" + " " * 35 + "█")
    print("█" + " " * 78 + "█")
    print("█"*80)
    
    debugger.debug_vocab(show_sample=True)
    
    # 测试3: 多文本对比 - 批量比较不同代码片段
    print("\n" + "█"*80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "测试3: 多文本对比" + " " * 35 + "█")
    print("█" + " " * 78 + "█")
    print("█"*80)
    
    test_texts = [
        "public class User",                    # 简单类声明
        "private void process",                 # 方法签名
        "@GetMapping(\"/api/users\")",          # Spring注解
        "userRepository.findById(id)",          # 方法调用
        "List<User> users = new ArrayList<>();",  # 泛型集合初始化
    ]
    debugger.compare_texts(test_texts)
    
    # 测试4: 边界情况 - 测试极端输入
    print("\n" + "█"*80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "测试4: 边界情况" + " " * 37 + "█")
    print("█" + " " * 78 + "█")
    print("█"*80)
    
    edge_cases = [
        "",              # 空字符串：测试空输入处理
        "a",             # 单字符：测试最小输入
        "你好世界",       # 中文：测试非ASCII字符（预期产生UNK）
        "var123 var456", # 变量名：测试字母数字组合
        "{[()]}",        # 纯符号：测试特殊字符序列
    ]
    
    for case in edge_cases:
        print(f"\n{'─'*80}")
        print(f"边界测试: \"{case}\"")
        print(f"{'─'*80}")
        debugger.debug_tokenize(case, show_details=False)  # 简化模式，只显示摘要


if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        main()
