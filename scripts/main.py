"""
主程序入口 - 演示完整的LLM代码生成流程

这个程序展示了大模型从输入到输出的完整过程，包括：
1. Token化
2. 位置编码
3. Transformer推理
4. 注意力机制
5. 采样策略
6. 代码生成
7. 后处理
8. 缓存机制
9. 可视化
"""

import sys
import os
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.pipeline import CodeGenerationPipeline
from scripts.utils.visualizer import AttentionVisualizer
from scripts.utils.logger import logging_context

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def demo_basic_generation():
    """演示基本的代码生成流程"""
    print("\n" + "="*80)
    print(" " * 20 + "演示1: 基本代码生成流程")
    print("="*80)
    
    # 创建管道 (使用small preset,启用 debug_mode)
    pipeline = CodeGenerationPipeline(
        preset='small',
        device='cpu',
        debug_mode=True  # 启用调试模式
    )
    
    # 测试用例
    test_prompts = [
        "public class UserService",
        "public User findById",
        "private List<User>",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'#'*80}")
        print(f"# 测试用例 {i}: {prompt}")
        print(f"{'#'*80}")
        
        result = pipeline.generate(
            prompt=prompt,
            max_length=40,
            temperature=0.7,
            top_k=50,
            use_cache=True,
            do_post_process=True,
            verbose=True
        )
        
        if result['success']:
            print(f"\n[OK] 成功")
            print(f"  来源: {result['source']}")
            print(f"  耗时: {result['processing_time']:.4f}s")
            print(f"  Token数: {result['token_count']}")
        else:
            print(f"\n[ERROR] 失败: {result['error']}")
    
    # 显示统计
    pipeline.display_full_stats()


def demo_sampling_strategies():
    """演示不同的采样策略"""
    print("\n" + "="*80)
    print(" " * 20 + "演示2: 不同采样策略对比")
    print("="*80)
    
    pipeline = CodeGenerationPipeline(
        preset='small',
        device='cpu'
    )
    
    prompt = "public class UserController"
    
    temperatures = [0.3, 0.7, 1.2]
    
    print(f"\nPrompt: {prompt}\n")
    
    for temp in temperatures:
        print(f"\n{'-'*80}")
        print(f"Temperature = {temp}")
        print(f"{'-'*80}")
        
        result = pipeline.generate(
            prompt=prompt,
            max_length=30,
            temperature=temp,
            use_cache=False,
            verbose=False
        )
        
        if result['success']:
            print(f"生成的代码 ({result['token_count']} tokens):\n")
            print(result['processed_code'])
    
    print("\n说明:")
    print("  - 低温度 (0.3): 更确定性，倾向于高概率token")
    print("  - 中温度 (0.7): 平衡确定性和多样性")
    print("  - 高温度 (1.2): 更随机，更多样化但可能不太准确")


def demo_cache_mechanism():
    """演示缓存机制"""
    print("\n" + "="*80)
    print(" " * 20 + "演示3: 缓存机制")
    print("="*80)
    
    pipeline = CodeGenerationPipeline(
        preset='tiny',  # 使用tiny更快
        device='cpu'
    )
    
    prompt = "public class OrderService"
    
    # 第一次请求（缓存未命中）
    print("\n--- 第一次请求 ---")
    result1 = pipeline.generate(prompt, max_length=20, verbose=True)
    
    # 第二次相同请求（应该命中缓存）
    print("\n--- 第二次相同请求 ---")
    result2 = pipeline.generate(prompt, max_length=20, verbose=True)
    
    # 第三次相同请求（继续命中缓存）
    print("\n--- 第三次相同请求 ---")
    result3 = pipeline.generate(prompt, max_length=20, verbose=True)
    
    print(f"\n性能对比:")
    print(f"  第一次（生成）: {result1['processing_time']:.4f}s")
    print(f"  第二次（缓存）: {result2['processing_time']:.4f}s")
    print(f"  第三次（缓存）: {result3['processing_time']:.4f}s")
    
    if result2['source'] == 'cache':
        speedup = result1['processing_time'] / result2['processing_time'] if result2['processing_time'] > 0 else float('inf')
        print(f"\n[OK] 缓存加速比: {speedup:.1f}x")
    
    # 显示缓存统计
    pipeline.cache.display_stats()


def demo_multiple_samples():
    """演示多样本生成"""
    print("\n" + "="*80)
    print(" " * 20 + "演示4: 多样本生成")
    print("="*80)
    
    pipeline = CodeGenerationPipeline(
        preset='small',
        device='cpu'
    )
    
    prompt = "public class ProductService"
    
    samples = pipeline.generate_multiple_samples(
        prompt=prompt,
        num_samples=3,
        max_length=35,
        temperatures=[0.5, 0.7, 1.0]
    )
    
    print(f"\n生成了 {len(samples)} 个不同的样本:\n")
    
    for sample in samples:
        print(f"{'='*80}")
        print(f"Sample {sample['sample_id']} (temperature={sample['temperature']})")
        print(f"Tokens: {sample['token_count']}, Time: {sample['processing_time']:.4f}s")
        print(f"{'='*80}")
        print(sample['code'])
        print()


def demo_visualization():
    """演示可视化功能"""
    print("\n" + "="*80)
    print(" " * 20 + "演示5: 可视化功能")
    print("="*80)
    
    visualizer = AttentionVisualizer()
    
    if not visualizer.available:
        print("\n[WARNING] 跳过可视化演示：需要安装matplotlib和seaborn")
        print("  运行: pip install matplotlib seaborn")
        return
    
    # 可视化模型架构
    print("\n生成模型架构图...")
    visualizer.visualize_model_architecture(
        num_encoder_layers=2,
        num_decoder_layers=2,
        save_path='model_architecture.png'
    )
    
    print("\n[OK] 可视化完成，查看生成的PNG文件")


def interactive_mode():
    """交互模式 - 用户可以输入自己的prompt"""
    print("\n" + "="*80)
    print(" " * 20 + "交互模式: 输入您的代码提示")
    print("="*80)
    print("\n输入 'quit' 或 'exit' 退出")
    print("输入 'stats' 查看统计信息")
    print("输入 'help' 查看帮助\n")
    
    pipeline = CodeGenerationPipeline(
        preset='small',
        device='cpu'
    )
    
    while True:
        try:
            user_input = input("\n>>> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            
            if user_input.lower() == 'stats':
                pipeline.display_full_stats()
                continue
            
            if user_input.lower() == 'help':
                print("\n可用命令:")
                print("  <your prompt>  - 生成代码")
                print("  stats          - 显示统计信息")
                print("  quit/exit/q    - 退出程序")
                print("  help           - 显示此帮助")
                continue
            
            if not user_input:
                continue
            
            # 生成代码
            result = pipeline.generate(
                prompt=user_input,
                max_length=50,
                temperature=0.7,
                verbose=True
            )
            
        except KeyboardInterrupt:
            print("\n\n中断。再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("="*80)
    print(" "*20 + "LLM代码生成演示系统")
    print(" "*15 + "理解和调试大模型生成过程")
    print("="*80)
    
    print("\n请选择演示模式:")
    print("  1. 基本代码生成流程")
    print("  2. 不同采样策略对比")
    print("  3. 缓存机制演示")
    print("  4. 多样本生成")
    print("  5. 可视化功能")
    print("  6. 交互模式")
    print("  7. 运行所有演示")
    print("  0. 退出")
    
    try:
        choice = input("\n请输入选项 (0-7): ").strip()
    except EOFError:
        choice = '1'  # 默认运行演示1
    
    if choice == '1':
        demo_basic_generation()
    elif choice == '2':
        demo_sampling_strategies()
    elif choice == '3':
        demo_cache_mechanism()
    elif choice == '4':
        demo_multiple_samples()
    elif choice == '5':
        demo_visualization()
    elif choice == '6':
        interactive_mode()
    elif choice == '7':
        demo_basic_generation()
        demo_sampling_strategies()
        demo_cache_mechanism()
        demo_multiple_samples()
        demo_visualization()
    elif choice == '0':
        print("\n再见！")
        return
    else:
        print("\n无效选项，运行默认演示...")
        demo_basic_generation()
    
    print("\n" + "="*80)
    print("演示完成！")
    print("="*80)
    print("\n下一步:")
    print("  1. 查看各个模块的源代码了解实现细节")
    print("  2. 修改参数观察不同的生成效果")
    print("  3. 使用交互模式测试您自己的prompt")
    print("  4. 阅读README.md了解更多用法")


if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        main()
