"""
测试增强功能：概率分析和注意力追踪
"""

import torch
from pipeline import CodeGenerationPipeline
from generator import ProbabilityAnalyzer


def test_probability_analysis():
    """测试概率分布分析功能"""
    print("="*70)
    print("测试1: 概率分布分析")
    print("="*70)
    
    # 创建管道（启用概率分析）
    pipeline = CodeGenerationPipeline(
        vocab_size=1000,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        device='cpu'
    )
    
    prompt = "public class UserService"
    
    print(f"\nPrompt: {prompt}\n")
    
    # 生成代码并启用概率分析
    result = pipeline.generate(
        prompt=prompt,
        max_length=15,
        temperature=0.7,
        verbose=True,
        enable_probability_analysis=True
    )
    
    if result['success']:
        print(f"\n[OK] 生成成功")
        print(f"Token数: {result['token_count']}")
        
        if result['probability_analyses']:
            print(f"\n概率分析数据点数: {len(result['probability_analyses'])}")
            
            # 显示平均指标
            avg_entropy = sum(a['entropy'] for a in result['probability_analyses']) / len(result['probability_analyses'])
            avg_rank = sum(a['ranks'] for a in result['probability_analyses']) / len(result['probability_analyses'])
            
            print(f"平均熵: {avg_entropy:.4f} (越低表示模型越确定)")
            print(f"平均排名: {avg_rank:.1f} (越低表示选中的token概率越高)")


def test_attention_tracking():
    """测试注意力追踪功能"""
    print("\n" + "="*70)
    print("测试2: 注意力权重追踪")
    print("="*70)
    
    # 创建管道（启用注意力追踪）
    pipeline = CodeGenerationPipeline(
        vocab_size=1000,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        device='cpu',
        enable_attention_tracking=True
    )
    
    prompt = "public class OrderService"
    
    print(f"\nPrompt: {prompt}\n")
    
    # 生成代码
    result = pipeline.generate(
        prompt=prompt,
        max_length=10,
        temperature=0.7,
        verbose=True,
        enable_probability_analysis=False
    )
    
    if result['success']:
        print(f"\n[OK] 生成成功")
        
        # 获取注意力追踪器
        tracker = pipeline.model.get_attention_tracker()
        
        if tracker:
            print("\n--- 逐层注意力模式对比 ---")
            tracker.print_layer_comparison(
                num_encoder_layers=2,
                num_decoder_layers=2
            )
            
            # 尝试可视化（如果matplotlib可用）
            try:
                token_names = prompt.split()[:10]
                tracker.visualize_attention_evolution(
                    token_names=token_names,
                    save_path='attention_evolution_demo.png'
                )
            except Exception as e:
                print(f"\n注意: 可视化跳过 - {e}")


def test_probability_analyzer_directly():
    """直接测试ProbabilityAnalyzer类"""
    print("\n" + "="*70)
    print("测试3: ProbabilityAnalyzer独立测试")
    print("="*70)
    
    # 创建模拟logits
    vocab_size = 1000
    logits = torch.randn(1, vocab_size)
    
    # 假设选中了概率最高的token
    probs = torch.softmax(logits, dim=-1)
    selected_token_id = torch.argmax(probs).item()
    
    # 分析概率分布
    analysis = ProbabilityAnalyzer.analyze_distribution(logits, selected_token_id)
    
    print(f"\n分析结果:")
    print(f"  熵: {analysis['entropy']:.4f}")
    print(f"  选中token排名: #{analysis['ranks']}")
    print(f"  选中概率: {analysis['selected_probabilities'][0]:.4f}")
    print(f"  Top-1概率: {analysis['top1_probabilities'][0]:.4f}")
    print(f"  Top-5累积概率: {analysis['top5_cumulative_probabilities'][0]:.4f}")
    
    gini_val = analysis['gini_coefficients'] if isinstance(analysis['gini_coefficients'], float) else analysis['gini_coefficients'][0]
    vocab_val = analysis['vocab_coverage_90pct'] if isinstance(analysis['vocab_coverage_90pct'], int) else analysis['vocab_coverage_90pct'][0]
    
    print(f"  Gini系数: {gini_val:.4f}")
    print(f"  90%概率覆盖token数: {vocab_val}")
    
    # 格式化输出
    token_name = f"token_{selected_token_id}"
    formatted = ProbabilityAnalyzer.format_analysis(analysis, step=0, selected_token=token_name)
    print(formatted)


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "增强功能测试套件")
    print("="*70)
    
    try:
        test_probability_analyzer_directly()
        test_probability_analysis()
        test_attention_tracking()
        
        print("\n" + "="*70)
        print("所有测试完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
