"""
使用 Python Debugger (pdb) 调试 Tokenizer
演示如何使用断点和单步执行来调试 tokenizer
"""

import pdb
import sys
import os

# 添加父目录（scripts）到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import SimpleTokenizer
from logger import logging_context


def debug_with_breakpoints():
    """使用断点调试 tokenizer"""
    
    print("\n" + "="*80)
    print(" " * 20 + "Python Debugger (pdb) 示例")
    print("="*80)
    print("\n这个示例展示了如何使用 pdb 调试 tokenizer")
    print("常用 pdb 命令:")
    print("  n (next)     - 执行下一行")
    print("  s (step)     - 进入函数内部")
    print("  c (continue) - 继续执行到下一个断点")
    print("  p <变量>     - 打印变量值")
    print("  l (list)     - 显示当前代码")
    print("  h (help)     - 显示帮助")
    print("  q (quit)     - 退出调试器")
    print("="*80 + "\n")
    
    # 创建 tokenizer
    print("[1] 创建 tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)
    
    # 测试文本
    test_text = "public class UserService { public User findById(Long id) }"
    print(f"[2] 测试文本: {test_text}\n")
    
    # 设置断点 - 在这里可以检查变量
    print("[3] 即将进入调试模式...")
    print("提示: 输入 'c' 继续执行, 'q' 退出\n")
    
    # 断点1: tokenize 之前
    pdb.set_trace()
    
    # Tokenize
    print("\n[4] 执行 tokenize...")
    tokens = tokenizer.tokenize(test_text)
    
    # 断点2: tokenize 之后
    pdb.set_trace()
    
    # Encode
    print("\n[5] 执行 encode...")
    token_ids, attention_mask = tokenizer.encode(test_text, max_length=64)
    
    # 断点3: encode 之后
    pdb.set_trace()
    
    # Decode
    print("\n[6] 执行 decode...")
    decoded = tokenizer.decode(token_ids)
    
    print(f"\n[7] 最终结果:")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Token IDs: {len(token_ids)}")
    print(f"  Decoded: {decoded}")


def debug_specific_function():
    """调试特定函数"""
    
    print("\n" + "="*80)
    print(" " * 20 + "调试特定函数示例")
    print("="*80 + "\n")
    
    tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=False)
    
    test_text = "public class User"
    
    # 在 tokenize 函数中设置断点
    print(f"测试文本: {test_text}")
    print("即将进入 tokenize 函数内部...\n")
    
    # 使用 break 命令在特定函数设置断点
    pdb.run('tokenizer.tokenize(test_text)', globals(), locals())


def interactive_debug_example():
    """交互式调试示例"""
    
    print("\n" + "="*80)
    print(" " * 20 + "交互式调试示例")
    print("="*80 + "\n")
    
    tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)
    
    while True:
        try:
            text = input("\n输入要分析的文本 (或 'q' 退出): ").strip()
            
            if text.lower() == 'q':
                break
            
            if not text:
                continue
            
            # 在这里可以设置断点进行调试
            print(f"\n分析: \"{text}\"")
            
            # 取消下面这行的注释来启用断点
            # pdb.set_trace()
            
            tokens = tokenizer.tokenize(text)
            token_ids, mask = tokenizer.encode(text, max_length=32)
            
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {token_ids[:10]}...")
            print(f"Mask: {mask[:10]}...")
            
        except KeyboardInterrupt:
            print("\n\n退出...")
            break


if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        if len(sys.argv) > 1:
            mode = sys.argv[1]
            
            if mode == '1':
                debug_with_breakpoints()
            elif mode == '2':
                debug_specific_function()
            elif mode == '3':
                interactive_debug_example()
            else:
                print("用法: python tokenizer_pdb_debug.py [1|2|3]")
                print("  1 - 使用断点调试")
                print("  2 - 调试特定函数")
                print("  3 - 交互式调试")
        else:
            # 默认运行模式1
            print("选择调试模式:")
            print("  1. 使用断点调试 (推荐)")
            print("  2. 调试特定函数")
            print("  3. 交互式调试")
            
            choice = input("\n请选择 (1-3): ").strip()
            
            if choice == '1':
                debug_with_breakpoints()
            elif choice == '2':
                debug_specific_function()
            elif choice == '3':
                interactive_debug_example()
            else:
                print("无效选择")
