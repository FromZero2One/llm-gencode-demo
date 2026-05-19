"""
注意力机制模块 - 实现Multi-Head Attention
这是Transformer的核心组件，用于理解token之间的关系
"""

import sys
import torch
import torch.nn as nn
import math
import logging
from typing import Tuple, Optional
from scripts.utils.logger import logging_context

# 配置日志
logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    核心思想：让每个token关注序列中的其他相关token
    例如：在"public class UserService"中，"class"应该关注"public"和"UserService"
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8, dropout: float = 0.1, debug_mode: bool = False):
        """
        Args:
            d_model: 模型维度（嵌入维度）
            nhead: 注意力头的数量
            dropout: Dropout概率
            debug_mode: 是否启用调试模式
        """
        super().__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个头的维度
        self.debug_mode = debug_mode
        
        if self.debug_mode:
            logger.info(f"初始化多头注意力机制")
            logger.info(f"  - 模型维度 (d_model): {d_model}")
            logger.info(f"  - 注意力头数 (nhead): {nhead}")
            logger.info(f"  - 每头维度 (d_k): {self.d_k}")
        
        # Q, K, V的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
        # 用于存储注意力权重（可视化用）
        self.attention_weights = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: 可选的掩码 [batch_size, 1, 1, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, nhead, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        if self.debug_mode:
            logger.debug(f"前向传播")
            logger.debug(f"  - Batch size: {batch_size}")
            logger.debug(f"  - Query sequence length: {seq_len_q}")
            logger.debug(f"  - Key sequence length: {seq_len_k}")
            logger.debug(f"  - Input shape: {query.shape}")
        
        # 1. 线性变换并分割成多个头
        # [batch, seq_len, d_model] -> [batch, seq_len, nhead, d_k] -> [batch, nhead, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, seq_len_q, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        
        if self.debug_mode:
            logger.debug(f"  - Q/K/V shape after split: {Q.shape}")
        
        # 2. 计算注意力分数: Q * K^T / sqrt(d_k)
        # [batch, nhead, seq_len, d_k] @ [batch, nhead, d_k, seq_len] 
        # -> [batch, nhead, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if self.debug_mode:
            logger.debug(f"  - Attention scores shape: {scores.shape}")
        
        # 3. 应用mask（如果有）
        if mask is not None:
            # mask为0的位置填充极小值，使softmax后接近0
            scores = scores.masked_fill(mask == 0, -1e9)
            if self.debug_mode:
                logger.debug(f"  - Applied mask")
        
        # 4. Softmax获取注意力权重
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        
        # 保存注意力权重用于可视化
        self.attention_weights = attention_weights.detach()
        
        if self.debug_mode:
            logger.debug(f"  - Attention weights shape: {attention_weights.shape}")
            logger.debug(f"  - Attention weights sum (should be 1.0): {attention_weights.sum(dim=-1).mean().item():.4f}")
        
        # 5. 加权求和: Attention * V
        # [batch, nhead, seq_len, seq_len] @ [batch, nhead, seq_len, d_k]
        # -> [batch, nhead, seq_len, d_k]
        context = torch.matmul(attention_weights, V)
        
        # 6. 合并多头
        # [batch, nhead, seq_len_q, d_k] -> [batch, seq_len_q, nhead, d_k] -> [batch, seq_len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # 7. 输出线性变换
        output = self.W_o(context)
        
        if self.debug_mode:
            logger.debug(f"  - Output shape: {output.shape}")
        
        return output, attention_weights
    
    def explain_attention(self, token_names: list, head_idx: int = 0):
        """
        解释注意力权重（用于调试）
        
        Args:
            token_names: token名称列表
            head_idx: 要分析的注意力头索引
        """
        if self.attention_weights is None:
            print("[Attention] 没有可用的注意力权重")
            return
        
        # 获取第一个样本、指定头的注意力权重
        weights = self.attention_weights[0, head_idx].cpu().numpy()
        
        print(f"\n{'='*60}")
        print(f"[Attention] 注意力权重分析 (Head {head_idx})")
        print(f"{'='*60}")
        
        # 显示每个token最关注的其他token
        for i, token in enumerate(token_names[:min(10, len(token_names))]):
            if i >= weights.shape[0]:
                break
            
            # 获取该token对其他所有token的注意力权重
            attn_to_others = weights[i]
            
            # 找出最关注的3个token
            top_indices = attn_to_others.argsort()[::-1][:3]
            
            print(f"\nToken '{token}' (position {i}) 最关注:")
            for rank, idx in enumerate(top_indices, 1):
                if idx < len(token_names):
                    other_token = token_names[idx]
                    weight = attn_to_others[idx]
                    print(f"  {rank}. '{other_token}' (weight: {weight:.4f})")


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    Transformer本身没有位置信息，需要添加位置编码来标识token的顺序
    使用正弦和余弦函数生成位置编码
    """
    
    def __init__(self, d_model: int = 128, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # position: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term: [d_model/2]
        # 使用指数衰减让不同维度关注不同频率的位置信息
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        
        # 添加batch维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
        
        print(f"[PositionalEncoding] 初始化")
        print(f"  - d_model: {d_model}")
        print(f"  - max_len: {max_len}")
        print(f"  - PE shape: {pe.shape}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            x + positional_encoding: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # 截取相应长度的位置编码并添加到输入
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)
    
    def visualize_pe(self, d_model: int = 128, max_len: int = 100):
        """可视化位置编码（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            
            pe = self.pe[0, :max_len, :d_model].cpu().numpy()
            
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(pe, cmap='RdBu')
            plt.xlabel('Dimension')
            plt.ylabel('Position')
            plt.title('Positional Encoding')
            plt.colorbar()
            plt.show()
            
            print("[PositionalEncoding] 位置编码可视化完成")
        except ImportError:
            print("[PositionalEncoding] 安装matplotlib以启用可视化功能")


# 测试代码
if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        print("="*60)
        print("测试注意力机制")
        print("="*60)
        
        # 参数设置
        batch_size = 1
        seq_len = 10
        d_model = 128
        nhead = 8
        
        # 创建模拟输入
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")
    
    # 创建注意力层
    attention = MultiHeadAttention(d_model=d_model, nhead=nhead)
    
    # 前向传播
    output, attn_weights = attention(query, key, value)
    
    print(f"\n输出形状:")
    print(f"  Output: {output.shape}")
    print(f"  Attention Weights: {attn_weights.shape}")
    
    # 测试位置编码
    print("\n" + "="*60)
    print("测试位置编码")
    print("="*60)
    
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=100)
    embedded = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nEmbedding形状: {embedded.shape}")
    encoded = pos_encoder(embedded)
    print(f"编码后形状: {encoded.shape}")
    
    # 验证位置编码是否被添加
    diff = encoded - embedded
    print(f"\n位置编码的影响 (mean abs diff): {diff.abs().mean().item():.6f}")
