"""
可视化工具 - 用于可视化注意力权重和生成过程
"""

import sys
import os
import torch
from typing import List, Optional
import numpy as np
from logger import logging_context


class AttentionVisualizer:
    """注意力权重可视化工具"""
    
    def __init__(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self.available = True
            print("[Visualizer] Matplotlib和Seaborn已加载")
        except ImportError:
            self.available = False
            print("[Visualizer] 警告: 安装matplotlib和seaborn以启用可视化功能")
            print("  运行: pip install matplotlib seaborn")
    
    def visualize_attention(self, 
                           attention_weights: torch.Tensor,
                           token_names: List[str],
                           head_idx: int = 0,
                           layer_idx: int = 0,
                           save_path: Optional[str] = None):
        """
        可视化注意力权重
        
        Args:
            attention_weights: [batch, nhead, seq_len, seq_len]
            token_names: token名称列表
            head_idx: 要可视化的注意力头索引
            layer_idx: 层索引（用于标题）
            save_path: 保存路径（可选）
        """
        if not self.available:
            print("[Visualizer] 无法可视化：缺少依赖库")
            return
        
        # 获取第一个样本、指定头的注意力权重
        weights = attention_weights[0, head_idx].cpu().numpy()
        
        seq_len = weights.shape[0]
        token_display = token_names[:seq_len]
        
        # 创建热力图
        self.plt.figure(figsize=(12, 10))
        
        # 使用seaborn绘制热力图
        self.sns.heatmap(
            weights,
            xticklabels=token_display,
            yticklabels=token_display,
            cmap='viridis',
            annot=True,
            fmt='.2f',
            linewidths=0.5
        )
        
        self.plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})', 
                 fontsize=14, fontweight='bold')
        self.plt.xlabel('Key Tokens', fontsize=12)
        self.plt.ylabel('Query Tokens', fontsize=12)
        
        # 旋转x轴标签以便阅读
        self.plt.xticks(rotation=45, ha='right')
        self.plt.yticks(rotation=0)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] 注意力图保存到: {save_path}")
        
        self.plt.show()
    
    def visualize_attention_summary(self,
                                   attention_weights: torch.Tensor,
                                   token_names: List[str],
                                   num_heads: int = 8,
                                   save_path: Optional[str] = None):
        """
        可视化所有注意力头的摘要
        
        Args:
            attention_weights: [batch, nhead, seq_len, seq_len]
            token_names: token名称列表
            num_heads: 注意力头数量
            save_path: 保存路径
        """
        if not self.available:
            print("[Visualizer] 无法可视化：缺少依赖库")
            return
        
        nhead = attention_weights.shape[1]
        num_heads_to_show = min(num_heads, nhead)
        
        # 创建子图
        fig, axes = self.plt.subplots(2, (num_heads_to_show + 1) // 2, 
                                figsize=(6 * ((num_heads_to_show + 1) // 2), 10))
        
        if num_heads_to_show == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        seq_len = attention_weights.shape[2]
        token_display = token_names[:seq_len]
        
        for head_idx in range(num_heads_to_show):
            weights = attention_weights[0, head_idx].cpu().numpy()
            
            ax = axes[head_idx]
            im = ax.imshow(weights, cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
            
            # 设置刻度
            ax.set_xticks(range(len(token_display)))
            ax.set_yticks(range(len(token_display)))
            ax.set_xticklabels(token_display, rotation=45, ha='right', fontsize=6)
            ax.set_yticklabels(token_display, fontsize=6)
        
        # 隐藏多余的子图
        for idx in range(num_heads_to_show, len(axes)):
            axes[idx].set_visible(False)
        
        self.plt.suptitle('Multi-Head Attention Weights Overview', 
                    fontsize=14, fontweight='bold', y=1.02)
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] 注意力摘要保存到: {save_path}")
        
        self.plt.show()
    
    def visualize_generation_process(self,
                                    generation_details: list,
                                    save_path: Optional[str] = None):
        """
        可视化生成过程
        
        Args:
            generation_details: 生成详情列表
            save_path: 保存路径
        """
        if not self.available:
            print("[Visualizer] 无法可视化：缺少依赖库")
            return
        
        if not generation_details:
            print("[Visualizer] 没有生成详情可可视化")
            return
        
        # 提取数据
        steps = [d['step'] for d in generation_details]
        tokens = [d['token'] for d in generation_details]
        
        # 为每个步骤提取top-5概率
        fig, ax = self.plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(steps))
        width = 0.15
        
        for rank in range(5):
            probabilities = [
                d['top5_predictions'][rank]['probability'] 
                if rank < len(d['top5_predictions']) else 0
                for d in generation_details
            ]
            
            bars = ax.bar(x + rank * width, probabilities, width, 
                         label=f'Rank {rank + 1}')
        
        ax.set_xlabel('Generation Step', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Token Generation Process - Top 5 Predictions', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f"{s}\n({t})" for s, t in zip(steps, tokens)], 
                          fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] 生成过程图保存到: {save_path}")
        
        self.plt.show()
    
    def visualize_probability_distribution(self,
                                          logits: torch.Tensor,
                                          top_k: int = 20,
                                          save_path: Optional[str] = None):
        """
        可视化概率分布
        
        Args:
            logits: [vocab_size] 或 [batch, vocab_size]
            top_k: 显示前k个token
            save_path: 保存路径
        """
        if not self.available:
            print("[Visualizer] 无法可视化：缺少依赖库")
            return
        
        # 如果是一批，取第一个
        if logits.dim() == 2:
            logits = logits[0]
        
        # 转换为概率
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # 获取top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]
        top_tokens = [f"Token {idx}" for idx in top_indices]
        
        # 绘制条形图
        fig, ax = self.plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(top_tokens))
        bars = ax.bar(x, top_probs, color='steelblue', alpha=0.7)
        
        ax.set_xlabel('Tokens', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Top {top_k} Token Probabilities', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_tokens, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # 在条形上添加数值标签
        for bar, prob in zip(bars, top_probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.4f}',
                   ha='center', va='bottom', fontsize=7)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] 概率分布图保存到: {save_path}")
        
        self.plt.show()
    
    def visualize_model_architecture(self,
                                    num_encoder_layers: int = 2,
                                    num_decoder_layers: int = 2,
                                    save_path: Optional[str] = None):
        """
        可视化模型架构
        
        Args:
            num_encoder_layers: Encoder层数
            num_decoder_layers: Decoder层数
            save_path: 保存路径
        """
        if not self.available:
            print("[Visualizer] 无法可视化：缺少依赖库")
            return
        
        fig, ax = self.plt.subplots(figsize=(10, 12))
        
        # 定义组件
        components = ['Input\nEmbedding']
        components += [f'Encoder\nLayer {i+1}' for i in range(num_encoder_layers)]
        components += ['Positional\nEncoding']
        components += [f'Decoder\nLayer {i+1}' for i in range(num_decoder_layers)]
        components += ['Output\nProjection']
        components += ['Softmax']
        components += ['Output\nTokens']
        
        # 绘制流程图
        y_positions = range(len(components))
        colors = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#45B7D1', 
                 '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (component, color) in enumerate(zip(components, colors)):
            ax.barh(i, 1, color=color, edgecolor='black', linewidth=2)
            ax.text(0.5, i, component, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # 添加箭头
        for i in range(len(components) - 1):
            ax.annotate('', xy=(0.5, i + 1), xytext=(0.5, i),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(components) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('Transformer Architecture Overview', 
                    fontsize=16, fontweight='bold', pad=20)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] 模型架构图保存到: {save_path}")
        
        self.plt.show()


# 测试代码
if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        print("="*60)
        print("测试可视化工具")
        print("="*60)
        
        # 创建可视化输出目录
        output_dir = 'visualizations'
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n可视化文件将保存到: {output_dir}/\n")
        
        visualizer = AttentionVisualizer()
        
        if not visualizer.available:
            print("\n跳过可视化测试（缺少依赖库）")
        else:
            # 测试1: 模型架构可视化
            print("\n--- 测试1: 模型架构 ---")
            visualizer.visualize_model_architecture(
                num_encoder_layers=2,
                num_decoder_layers=2,
                save_path=os.path.join(output_dir, 'model_architecture.png')
            )
        
        # 测试2: 模拟注意力权重
        print("\n--- 测试2: 注意力权重 ---")
        batch_size = 1
        nhead = 4
        seq_len = 8
        
        # 创建模拟注意力权重
        attention_weights = torch.rand(batch_size, nhead, seq_len, seq_len)
        # 归一化
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        token_names = ['public', 'class', 'User', 'Service', '{', 
                      'private', 'User', 'repo']
        
        visualizer.visualize_attention(
            attention_weights,
            token_names,
            head_idx=0,
            save_path=os.path.join(output_dir, 'attention_head_0.png')
        )
        
        visualizer.visualize_attention_summary(
            attention_weights,
            token_names,
            num_heads=nhead,
            save_path=os.path.join(output_dir, 'attention_summary.png')
        )
        
        # 测试3: 概率分布
        print("\n--- 测试3: 概率分布 ---")
        vocab_size = 100
        logits = torch.randn(vocab_size)
        
        visualizer.visualize_probability_distribution(
            logits,
            top_k=15,
            save_path=os.path.join(output_dir, 'probability_distribution.png')
        )
        
        print(f"\n✓ 所有可视化测试完成")
        print(f"  生成的文件保存在: {output_dir}/ 目录")
        print(f"  - model_architecture.png")
        print(f"  - attention_head_0.png")
        print(f"  - attention_summary.png")
        print(f"  - probability_distribution.png")
