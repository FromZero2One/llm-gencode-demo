"""快速演示增强功能"""
from pipeline import CodeGenerationPipeline

print("="*70)
print(" "*20 + "增强功能演示")
print("="*70)

# 创建管道（同时启用两个功能）
pipeline = CodeGenerationPipeline(
    vocab_size=1000, d_model=128, nhead=8,
    num_encoder_layers=2, num_decoder_layers=2, device='cpu',
    enable_attention_tracking=True
)

# 生成代码
result = pipeline.generate(
    prompt="public class Demo",
    max_length=8,
    temperature=0.7,
    enable_probability_analysis=True,
    verbose=False
)

# 显示概率分析摘要
print("\n" + "="*70)
print("概率分析摘要")
print("="*70)
print(f"生成Token数: {result['token_count']}")
if result['probability_analyses']:
    avg_entropy = sum(a['entropy'] for a in result['probability_analyses']) / len(result['probability_analyses'])
    avg_rank = sum(a['ranks'] for a in result['probability_analyses']) / len(result['probability_analyses'])
    print(f"平均熵: {avg_entropy:.4f}")
    print(f"平均排名: #{avg_rank:.1f}")

# 显示注意力追踪摘要
tracker = pipeline.model.get_attention_tracker()
print("\n" + "="*70)
print("注意力追踪摘要")
print("="*70)
tracker.print_layer_comparison(2, 2)

print("\n" + "="*70)
print("演示完成！")
print("="*70)
