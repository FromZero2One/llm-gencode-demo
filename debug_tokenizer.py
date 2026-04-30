"""
Tokenizer Debug 工具
提供详细的 tokenization 过程可视化和调试功能
"""

import sys
import logging
from typing import List, Dict
from tokenizer import SimpleTokenizer

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class TokenizerDebugger:
    """Tokenizer 调试器 - 提供详细的分词过程分析"""
    
    def __init__(self, vocab_size: int = 1000):
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size, debug_mode=True)
        
    def debug_tokenize(self, text: str, show_details: bool = True) -> Dict:
        """
        详细调试 tokenization 过程
        
        Args:
            text: 输入文本
            show_details: 是否显示详细信息
            
        Returns:
            包含所有调试信息的字典
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
        
        # Step 5: 截断或填充
        max_length = 64
        print(f"\n[Step 5] 序列处理 (max_length={max_length})")
        original_length = len(token_ids)
        
        if len(token_ids) > max_length:
            print(f"  ⚠  需要截断: {original_length} → {max_length}")
            token_ids = token_ids[:max_length-1] + [self.tokenizer.vocab[self.tokenizer.EOS_TOKEN]]
        elif len(token_ids) < max_length:
            padding_count = max_length - len(token_ids)
            print(f"  ℹ  需要填充: {original_length} + {padding_count} PAD tokens")
            pad_id = self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]
            token_ids.extend([pad_id] * padding_count)
        else:
            print(f"  ✓ 长度正好: {len(token_ids)}")
        
        # Step 6: 创建 attention mask
        attention_mask = [1] * min(original_length, max_length) + [0] * max(0, max_length - min(original_length, max_length))
        result['attention_mask'] = attention_mask
        
        print(f"\n[Step 6] Attention Mask")
        print(f"  有效 tokens: {sum(attention_mask)}")
        print(f"  Padding tokens: {len(attention_mask) - sum(attention_mask)}")
        if show_details:
            mask_str = ''.join(['1' if m == 1 else '0' for m in attention_mask[:64]])
            print(f"  Mask (前64位): {mask_str}")
        
        # Step 7: 统计分析
        print(f"\n[Step 7] 统计分析")
        stats = {
            'total_tokens': len(tokens),
            'known_count': len(known_tokens) - 2,  # 排除BOS和EOS
            'unk_count': len(unk_tokens),
            'unk_rate': len(unk_tokens) / max(len(special_tokens), 1) * 100,
            'vocab_coverage': len(known_tokens) / max(len(special_tokens), 1) * 100
        }
        result['statistics'] = stats
        
        print(f"  总 token 数: {stats['total_tokens']}")
        print(f"  已知 tokens: {stats['known_count']}")
        print(f"  未知 tokens (UNK): {stats['unk_count']}")
        print(f"  词汇表覆盖率: {stats['vocab_coverage']:.1f}%")
        print(f"  UNK 比例: {stats['unk_rate']:.1f}%")
        
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
        """调试词汇表"""
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
        """比较多个文本的 tokenization 结果"""
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
    """主函数 - 运行所有 debug 测试"""
    
    # 创建 debugger
    debugger = TokenizerDebugger(vocab_size=1000)
    
    # 测试1: 基本 tokenization
    print("\n" + "█"*80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "测试1: 基本 Tokenization" + " " * 30 + "█")
    print("█" + " " * 78 + "█")
    print("█"*80)
    
    test_text = "public class UserService { public User findById(Long id) { return userRepository.findById(id).orElse(null); } }"
    result = debugger.debug_tokenize(test_text, show_details=True)
    
    # 测试2: 词汇表分析
    print("\n" + "█"*80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "测试2: 词汇表分析" + " " * 35 + "█")
    print("█" + " " * 78 + "█")
    print("█"*80)
    
    debugger.debug_vocab(show_sample=True)
    
    # 测试3: 多文本对比
    print("\n" + "█"*80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "测试3: 多文本对比" + " " * 35 + "█")
    print("█" + " " * 78 + "█")
    print("█"*80)
    
    test_texts = [
        "public class User",
        "private void process",
        "@GetMapping(\"/api/users\")",
        "userRepository.findById(id)",
        "List<User> users = new ArrayList<>();",
    ]
    debugger.compare_texts(test_texts)
    
    # 测试4: 边界情况
    print("\n" + "█"*80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "测试4: 边界情况" + " " * 37 + "█")
    print("█" + " " * 78 + "█")
    print("█"*80)
    
    edge_cases = [
        "",  # 空字符串
        "a",  # 单字符
        "你好世界",  # 中文
        "var123 var456",  # 变量名
        "{[()]}",  # 纯符号
    ]
    
    for case in edge_cases:
        print(f"\n{'─'*80}")
        print(f"边界测试: \"{case}\"")
        print(f"{'─'*80}")
        debugger.debug_tokenize(case, show_details=False)


if __name__ == '__main__':
    main()
