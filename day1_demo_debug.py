#!/usr/bin/env python3
"""
第1天学习：基本代码生成流程演示（Debug模式）

用途：查看详细的技术实现细节和中间过程
特点：
  - 启用 debug_mode=True，输出完整的Transformer计算过程
  - 启用 verbose=True，显示每一步的详细日志
  - 简洁直接，没有额外的教学说明

适用场景：
  - 调试代码时查看底层实现
  - 深入理解Attention机制的计算过程
  - 观察Token化的详细步骤
  - 分析模型的前向传播

运行方式：
  python day1_demo_debug.py

注意：此脚本会产生大量技术细节输出，适合进阶用户
"""

import sys
import os

# 将scripts目录添加到Python路径
script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, script_dir)

from pipeline import CodeGenerationPipeline
from logger import logging_context


def main():
    print("=" * 80)
    print("第1天学习：基本代码生成流程")
    print("=" * 80)
    print()

    # 创建管道（启用 debug 模式查看详细日志）
    pipeline = CodeGenerationPipeline(
        vocab_size=1000,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        debug_mode=True  # 启用 debug 模式
    )

    # 生成代码
    prompt = "public class UserService"
    print(f"\n[输入] Prompt: '{prompt}'")
    print(f"\n{'='*80}")
    print("开始生成代码...")
    print(f"{'='*80}\n")

    result = pipeline.generate(
        prompt=prompt,
        max_length=50,
        temperature=0.7,
        verbose=True  # 显示详细日志
    )

    print(f"\n{'='*80}")
    print("生成结果")
    print(f"{'='*80}")
    print(f"\n[生成的代码]:\n{result['processed_code']}")
    print(f"\n[统计信息]:")
    print(f"  - Token 数量: {result['token_count']}")
    print(f"  - 生成时间: {result['processing_time']:.4f} 秒")
    print(f"  - 生成速度: {result['token_count']/result['processing_time']:.2f} tokens/秒")
    print(f"  - 来源: {result['source']}")


if __name__ == "__main__":
    # 使用日志上下文管理器，自动创建日志文件
    with logging_context(__file__):
        main()
