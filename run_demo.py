#!/usr/bin/env python3
"""
LLM代码生成演示项目入口脚本

此脚本简化了项目的运行命令，自动处理模块导入路径。
"""

import sys
import os

# 将scripts目录添加到Python路径
script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, script_dir)

def show_menu():
    """显示主菜单"""
    print("="*60)
    print("           LLM代码生成演示项目 - 主菜单")
    print("="*60)
    print("请选择要运行的演示:")
    print("  1. 运行完整测试")
    print("  2. 运行主程序演示")
    print("  3. 运行性能测试")
    print("  4. 运行Tokenizer调试")
    print("  5. 运行交互式Tokenizer")
    print("  6. 运行调试模式")
    print("  0. 退出")
    print("="*60)

def main():
    """主函数"""
    import sys
    
    # 如果提供了命令行参数，直接使用
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        execute_choice(choice)
        return
    
    # 否则进入交互模式
    while True:
        show_menu()
        try:
            choice = input("请输入选项 (0-6): ").strip()
            execute_choice(choice)
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            break
        except EOFError:
            print("\n\n输入结束。")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            break

def execute_choice(choice):
    """执行选择的选项"""
    if choice == "0":
        print("感谢使用！再见！")
        exit(0)
    elif choice == "1":
        import test_all
    elif choice == "2":
        # 直接运行main模块
        import subprocess
        import sys
        script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
        subprocess.run([sys.executable, os.path.join(script_dir, 'main.py')])
    elif choice == "3":
        import test_performance
    elif choice == "4":
        import debug_tokenizer
    elif choice == "5":
        import tokenizer_interactive
    elif choice == "6":
        # 检查test_debug是否存在
        debug_file = os.path.join(os.path.dirname(__file__), 'scripts', 'test_debug.py')
        if os.path.exists(debug_file):
            import subprocess
            import sys
            subprocess.run([sys.executable, debug_file])
        else:
            print("调试脚本不存在，请运行 python scripts/main.py 并选择详细模式")
    else:
        print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()