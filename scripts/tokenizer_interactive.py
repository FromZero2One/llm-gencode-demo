"""
交互式 Tokenizer Debug 工具
可以实时输入文本并查看 tokenization 结果
"""

import sys
import logging
from tokenizer import SimpleTokenizer

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class InteractiveTokenizerDebugger:
    """交互式 Tokenizer 调试器"""
    
    def __init__(self, vocab_size: int = 1000):
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size, debug_mode=True)
        
    def visualize_tokens(self, text: str):
        """可视化显示 tokens"""
        tokens = self.tokenizer.tokenize(text)
        
        print("\n" + "─" * 80)
        print("Token 可视化")
        print("─" * 80)
        
        # 显示每个 token 及其 ID
        all_tokens = [self.tokenizer.BOS_TOKEN] + tokens + [self.tokenizer.EOS_TOKEN]
        
        print("\nToken 序列:")
        for i, token in enumerate(all_tokens):
            token_id = self.tokenizer.vocab.get(token, self.tokenizer.vocab[self.tokenizer.UNK_TOKEN])
            is_known = token in self.tokenizer.vocab
            
            if token == self.tokenizer.BOS_TOKEN:
                color_code = "\033[94m"  # Blue
            elif token == self.tokenizer.EOS_TOKEN:
                color_code = "\033[94m"  # Blue
            elif not is_known:
                color_code = "\033[91m"  # Red (UNK)
            else:
                color_code = "\033[92m"  # Green (Known)
            
            reset_code = "\033[0m"
            marker = "✓" if is_known else "✗"
            
            print(f"  [{i:2d}] {color_code}'{token:20s}'{reset_code} → ID: {token_id:4d} {marker}")
        
        print()
    
    def show_token_details(self, token: str):
        """显示单个 token 的详细信息"""
        print(f"\n{'─' * 80}")
        print(f"Token 详情: '{token}'")
        print(f"{'─' * 80}")
        
        if token in self.tokenizer.vocab:
            token_id = self.tokenizer.vocab[token]
            info = self.tokenizer.get_token_info(token_id)
            
            print(f"  Token ID: {token_id}")
            print(f"  是否在词汇表中: ✓ 是")
            print(f"  是否为特殊 token: {'是' if info['is_special'] else '否'}")
            print(f"  是否为关键字: {'是' if info['is_keyword'] else '否'}")
        else:
            print(f"  Token ID: N/A")
            print(f"  是否在词汇表中: ✗ 否 (会被映射为 UNK)")
            print(f"  UNK Token ID: {self.tokenizer.vocab[self.tokenizer.UNK_TOKEN]}")
        
        print()
    
    def interactive_mode(self):
        """进入交互模式"""
        print("\n" + "="*80)
        print(" " * 20 + "交互式 Tokenizer Debug 工具")
        print("="*80)
        print("\n使用说明:")
        print("  - 直接输入文本进行 tokenization")
        print("  - 输入 'vocab' 查看词汇表统计")
        print("  - 输入 'help' 查看帮助")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n再见!\n")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'vocab':
                    self.show_vocab_stats()
                    continue
                
                if user_input.lower() == 'test':
                    self.run_quick_tests()
                    continue
                
                # 执行 tokenization
                print(f"\n分析文本: \"{user_input}\"")
                result = self.analyze_text(user_input)
                
            except KeyboardInterrupt:
                print("\n\n再见!\n")
                break
            except Exception as e:
                print(f"\n错误: {e}\n")
    
    def analyze_text(self, text: str) -> dict:
        """分析文本并显示结果"""
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # 显示可视化
        self.visualize_tokens(text)
        
        # Encode
        token_ids, attention_mask = self.tokenizer.encode(text, max_length=64)
        
        print("─" * 80)
        print("编码结果")
        print("─" * 80)
        print(f"\nToken IDs (前20个):")
        print(f"  {token_ids[:20]}")
        
        print(f"\nAttention Mask (前20个):")
        print(f"  {attention_mask[:20]}")
        
        # Decode
        decoded = self.tokenizer.decode(token_ids)
        print(f"\n解码结果:")
        print(f"  \"{decoded}\"")
        
        # 统计
        unk_count = sum(1 for t in tokens if t not in self.tokenizer.vocab)
        known_count = len(tokens) - unk_count
        
        print(f"\n统计信息:")
        print(f"  总 tokens: {len(tokens)}")
        print(f"  已知 tokens: {known_count}")
        print(f"  未知 tokens (UNK): {unk_count}")
        print(f"  词汇表覆盖率: {known_count / max(len(tokens), 1) * 100:.1f}%")
        
        # 显示 UNK tokens
        if unk_count > 0:
            unk_tokens = [t for t in tokens if t not in self.tokenizer.vocab]
            print(f"\n⚠  未知 Tokens (建议添加到词汇表):")
            for token in set(unk_tokens):
                print(f"    - '{token}'")
        
        print()
        
        return {
            'tokens': tokens,
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'decoded': decoded
        }
    
    def show_vocab_stats(self):
        """显示词汇表统计"""
        vocab = self.tokenizer.vocab
        
        print("\n" + "="*80)
        print("词汇表统计")
        print("="*80)
        print(f"\n词汇表大小: {len(vocab)} tokens")
        
        # 分类统计
        special = [t for t in vocab.keys() if t.startswith('<')]
        keywords = [t for t in vocab.keys() if t in ['public', 'private', 'class', 'return', 'if', 'else', 'for', 'while']]
        
        print(f"\n分类:")
        print(f"  特殊 tokens: {len(special)}")
        print(f"  关键字: {len(keywords)}")
        print(f"  其他: {len(vocab) - len(special) - len(keywords)}")
        
        print(f"\n示例 tokens (前20个):")
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab[:20]:
            print(f"  [{token_id:3d}] '{token}'")
        
        print()
    
    def run_quick_tests(self):
        """运行快速测试"""
        print("\n" + "="*80)
        print("快速测试")
        print("="*80 + "\n")
        
        test_cases = [
            "public class User",
            "private void process",
            "@GetMapping(\"/api\")",
            "userRepository.findById(id)",
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n测试 {i}: \"{text}\"")
            tokens = self.tokenizer.tokenize(text)
            unk_count = sum(1 for t in tokens if t not in self.tokenizer.vocab)
            print(f"  Tokens: {len(tokens)}, UNK: {unk_count}")
        
        print()
    
    def show_help(self):
        """显示帮助信息"""
        print("\n" + "="*80)
        print("帮助信息")
        print("="*80)
        print("\n可用命令:")
        print("  <文本>      - 分析输入的文本")
        print("  vocab       - 查看词汇表统计")
        print("  test        - 运行快速测试")
        print("  help        - 显示此帮助")
        print("  quit/exit   - 退出程序")
        print("\n示例:")
        print('  >>> public class UserService')
        print('  >>> @GetMapping("/api/users")')
        print('  >>> userRepository.findById(id)')
        print()


def main():
    """主函数"""
    debugger = InteractiveTokenizerDebugger(vocab_size=1000)
    
    # 如果有命令行参数，直接分析
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        debugger.analyze_text(text)
    else:
        # 否则进入交互模式
        debugger.interactive_mode()


if __name__ == '__main__':
    main()
