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
    多头注意力机制 (Multi-Head Attention)
    
    ════════════════════════════════════════════════════════════
    📚 核心概念：为什么需要"多头"？
    ════════════════════════════════════════════════════════════
    
    【直观理解】
    想象你在阅读一段代码时，会同时从多个角度理解每个token的含义：
    
    示例代码: "public class UserService extends BaseService implements IUserService"
    
    🔍 角度1 - 语法关系 (Syntactic Head):
       - "class" 关注 "public" → 确定访问修饰符
       - "extends" 关注 "class" → 确定继承关系
       - "implements" 关注 "class" → 确定接口实现
    
    🔍 角度2 - 语义关系 (Semantic Head):
       - "UserService" 关注 "Service" → 理解业务领域
       - "BaseService" 关注 "Service" → 理解基类功能
       - "IUserService" 关注 "UserService" → 理解接口对应
    
    🔍 角度3 - 结构关系 (Structural Head):
       - "extends" 关注 "BaseService" → 识别父类
       - "implements" 关注 "IUserService" → 识别接口
       - "class" 关注 "UserService" → 识别类名
    
    💡 **多头的好处**:
    每个头可以学习不同的关系模式，就像人类用多维度思维理解语言。
    如果只有单头，模型只能捕捉一种关系，表达能力受限。
    
    【数学原理】
    ┌─────────────────────────────────────────────────────┐
    │  Single-Head:  Attention = softmax(QK^T/√d)V        │
    │                                                       │
    │  Multi-Head:   head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)  │
    │                  output = Concat(head_1, ..., head_h)W^O   │
    │                                                       │
    │  其中 h = nhead (头的数量)                             │
    │        d_k = d_model / nhead (每头的维度)              │
    └─────────────────────────────────────────────────────┘
    
    【实际案例对比】
    ┌─────────────────────────────────────────────────────┐
    │  GPT-3:    d_model=12288, nhead=96,  d_k=128        │
    │  BERT:     d_model=768,   nhead=12,  d_k=64         │
    │  Llama-2:  d_model=4096,  nhead=32,  d_k=128        │
    │  本项目:   d_model=128,   nhead=8,   d_k=16         │
    └─────────────────────────────────────────────────────┘
    
    核心思想：让每个token关注序列中的其他相关token，但不同头关注不同类型的关系。
    例如：在"public class UserService"中：
      - Head 0可能关注语法修饰符(public → class)
      - Head 1可能关注命名模式(UserService → Service)
      - Head 2可能关注关键字(class → extends)
      - ...
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8, dropout: float = 0.1, debug_mode: bool = False):
        """
        初始化多头注意力机制
        
        Args:
            d_model: 模型维度（嵌入维度）
                    - 决定模型的表达能力
                    - 越大能捕捉越复杂的模式，但计算成本越高
                    - 常见值: 128(小型), 512(BERT-base), 768(BERT-large), 4096(Llama)
            
            nhead: 注意力头的数量
                   - 必须能整除d_model
                   - 越多头可以捕捉越多的关系类型
                   - 常见值: 8(小型), 12(BERT-base), 16(GPT-2 small), 32(Llama)
                   - 经验法则: d_model / nhead >= 16 (保证每头有足够维度)
            
            dropout: Dropout概率
                    - 防止过拟合的正则化技术
                    - 训练时随机丢弃部分神经元
                    - 常见值: 0.1(小数据集), 0.05(大数据集), 0.0(微调)
            
            debug_mode: 是否启用调试模式
                       - True: 输出详细的中间过程日志
                       - False: 静默运行，提升性能
        
        Raises:
            AssertionError: 如果d_model不能被nhead整除
        
        Example:
            >>> # 推荐配置
            >>> attention = MultiHeadAttention(d_model=128, nhead=8)  # 每头16维
            >>> attention = MultiHeadAttention(d_model=256, nhead=8)  # 每头32维
            >>> attention = MultiHeadAttention(d_model=512, nhead=16) # 每头32维
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
        前向传播 - Multi-Head Attention的核心计算流程
        
        ════════════════════════════════════════════════════════════
        📖 7步计算流程（建议对照代码逐行理解）
        ════════════════════════════════════════════════════════════
        
        Step 1: 线性变换并分割成多个头
          - Q = W_q * query  (Query投影)
          - K = W_k * key    (Key投影)
          - V = W_v * value  (Value投影)
          - 每个从 [batch, seq_len, d_model] 
            变为 [batch, nhead, seq_len, d_k]
        
        Step 2: 计算注意力分数
          - scores = Q @ K^T / sqrt(d_k)
          - 点积相似度，除以sqrt(d_k)防止梯度消失
          - 输出形状: [batch, nhead, seq_len_q, seq_len_k]
        
        Step 3: 应用mask（可选）
          - 对于decoder的causal mask，防止看到未来token
          - mask=0的位置填充-1e9，softmax后接近0
        
        Step 4: Softmax归一化
          - attention_weights = softmax(scores)
          - 每行的和为1.0，表示概率分布
        
        Step 5: 加权求和
          - context = attention_weights @ V
          - 根据注意力权重聚合value信息
        
        Step 6: 合并多头
          - [batch, nhead, seq_len, d_k] 
          - → [batch, seq_len, nhead, d_k]
          - → [batch, seq_len, d_model]
        
        Step 7: 输出线性变换
          - output = W_o * context
          - 融合所有头的信息
        
        Args:
            query: Query张量 [batch_size, seq_len_q, d_model]
                   - 在self-attention中，query=key=value=embedding
                   - 在cross-attention中，query来自decoder，key/value来自encoder
            
            key: Key张量 [batch_size, seq_len_k, d_model]
                 - 用于计算与query的相似度
            
            value: Value张量 [batch_size, seq_len_k, d_model]
                   - 被注意力权重加权的信息源
            
            mask: 可选的掩码张量 [batch_size, 1, 1, seq_len_k]
                  - 值为0或1，0表示需要屏蔽的位置
                  - 常见用途:
                    * Padding mask: 屏蔽序列中的padding token
                    * Causal mask: decoder中防止看到未来token
        
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, d_model]
                    - 融合了全局上下文信息的表示
            
            attention_weights: 注意力权重 [batch_size, nhead, seq_len_q, seq_len_k]
                               - 可用于可视化，观察模型关注哪些位置
                               - 第i行第j列表示query_i对key_j的关注程度
        
        Example:
            >>> # Self-attention示例
            >>> batch_size, seq_len, d_model = 2, 10, 128
            >>> embedding = torch.randn(batch_size, seq_len, d_model)
            >>> output, weights = attention(embedding, embedding, embedding)
            >>> print(output.shape)   # [2, 10, 128]
            >>> print(weights.shape)  # [2, 8, 10, 10]
            
            >>> # Cross-attention示例 (Decoder关注Encoder)
            >>> encoder_output = torch.randn(2, 15, 128)  # Encoder输出
            >>> decoder_hidden = torch.randn(2, 10, 128)  # Decoder隐藏状态
            >>> output, weights = attention(decoder_hidden, encoder_output, encoder_output)
            >>> print(output.shape)   # [2, 10, 128] - Decoder维度
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
        
        # ════════════════════════════════════════════════════════
        # Step 1: 线性变换并分割成多个头
        # ════════════════════════════════════════════════════════
        # 
        # 【为什么要分割？】
        # 将d_model维度拆分成nhead个小维度，让每个头独立学习不同的关系模式。
        # 例如：d_model=128, nhead=8 → 每头16维
        #   - Head 0: 关注语法关系 (public → class)
        #   - Head 1: 关注命名模式 (UserService → Service)
        #   - Head 2: 关注继承关系 (extends → BaseService)
        #   - ...
        #
        # 【形状变换详解】
        # [batch, seq_len, d_model]  (原始输入)
        #   ↓ self.W_q(query)  (线性投影)
        # [batch, seq_len, d_model]
        #   ↓ .view(batch, seq_len, nhead, d_k)  (重塑)
        # [batch, seq_len, 8, 16]
        #   ↓ .transpose(1, 2)  (交换维度，方便矩阵运算)
        # [batch, 8, seq_len, 16]  (最终Q/K/V的形状)
        #
        Q = self.W_q(query).view(batch_size, seq_len_q, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        
        if self.debug_mode:
            logger.debug(f"  - Q/K/V shape after split: {Q.shape}")
        
        # ════════════════════════════════════════════════════════
        # Step 2: 计算注意力分数 (Scaled Dot-Product Attention)
        # ════════════════════════════════════════════════════════
        #
        # 【数学公式】
        # scores = Q @ K^T / sqrt(d_k)
        #
        # 【为什么除以sqrt(d_k)？】
        # - 当d_k较大时，点积结果会很大，导致softmax梯度接近0
        # - 除以sqrt(d_k)可以保持方差稳定在1附近
        # - 这是Transformer论文的关键创新之一
        #
        # 【形状变化】
        # Q: [batch, 8, seq_len_q, 16]
        # K^T: [batch, 8, 16, seq_len_k]  (转置最后两维)
        #   ↓ torch.matmul(Q, K.transpose(-2, -1))
        # scores: [batch, 8, seq_len_q, seq_len_k]
        #   ↓ / math.sqrt(self.d_k)
        # scores: [batch, 8, seq_len_q, seq_len_k]  (缩放后)
        #
        # 【物理意义】
        # scores[i, j] 表示第i个query token对第j个key token的关注程度
        # 值越大表示越相关，后续softmax会将其转化为概率
        #
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if self.debug_mode:
            logger.debug(f"  - Attention scores shape: {scores.shape}")
        
        # ════════════════════════════════════════════════════════
        # Step 3: 应用mask（如果有）
        # ════════════════════════════════════════════════════════
        #
        # 【Mask的作用】
        # 防止模型关注不应该看到的位置：
        #   1. Padding Mask: 屏蔽序列中的<pad> token
        #      - 例如: "public class UserService <pad> <pad>"
        #      - 不应该关注<pad>，因为它们没有实际意义
        #
        #   2. Causal Mask (Decoder): 防止看到未来token
        #      - 生成第t个token时，只能看到前t-1个token
        #      - 否则会出现"偷看答案"的问题
        #
        # 【实现方式】
        # mask中值为0的位置会被填充为-1e9（极小值）
        # softmax(-1e9) ≈ 0，这样这些位置的权重就接近0
        #
        if mask is not None:
            # mask为0的位置填充极小值，使softmax后接近0
            scores = scores.masked_fill(mask == 0, -1e9)
            if self.debug_mode:
                logger.debug(f"  - Applied mask")
        
        # ════════════════════════════════════════════════════════
        # Step 4: Softmax获取注意力权重
        # ════════════════════════════════════════════════════════
        #
        # 【Softmax的作用】
        # 将原始分数(scores)转化为概率分布(attention_weights)
        #   - 所有权重的和为1.0
        #   - 权重越大表示越重要
        #
        # 【Dropout的作用】
        # 训练时随机丢弃部分注意力权重，防止过拟合
        #   - 测试时dropout不生效
        #   - 类似图像数据增强中的随机裁剪
        #
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        
        # 保存注意力权重用于可视化
        self.attention_weights = attention_weights.detach()
        
        if self.debug_mode:
            logger.debug(f"  - Attention weights shape: {attention_weights.shape}")
            logger.debug(f"  - Attention weights sum (should be 1.0): {attention_weights.sum(dim=-1).mean().item():.4f}")
        
        # ════════════════════════════════════════════════════════
        # Step 5: 加权求和 (Context聚合)
        # ════════════════════════════════════════════════════════
        #
        # 【核心思想】
        # 根据注意力权重，从Value中提取相关信息
        #   context_i = Σ_j (attention_weights[i,j] * V[j])
        #
        # 【直观理解】
        # 如果token i关注token j的程度是0.3，那么V[j]的信息会以0.3的权重
        # 加入到context[i]中。这样context[i]就融合了全局的上下文信息。
        #
        # 【形状变化】
        # attention_weights: [batch, 8, seq_len_q, seq_len_k]
        # V:                 [batch, 8, seq_len_k, 16]
        #   ↓ torch.matmul(attention_weights, V)
        # context:           [batch, 8, seq_len_q, 16]
        #
        context = torch.matmul(attention_weights, V)
        
        # ════════════════════════════════════════════════════════
        # Step 6: 合并多头
        # ════════════════════════════════════════════════════════
        #
        # 【为什么要合并？】
        # 每个头学习了不同的关系模式，现在需要将所有头的信息融合起来。
        #   - Head 0: 语法关系信息
        #   - Head 1: 语义关系信息
        #   - Head 2: 结构关系信息
        #   - ... → 拼接成完整的表示
        #
        # 【形状变换详解】
        # [batch, 8, seq_len_q, 16]  (8个头分开)
        #   ↓ .transpose(1, 2)  (交换维度)
        # [batch, seq_len_q, 8, 16]
        #   ↓ .contiguous().view(batch, seq_len_q, 8*16)  (重塑)
        # [batch, seq_len_q, 128]  (合并回d_model)
        #
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # ════════════════════════════════════════════════════════
        # Step 7: 输出线性变换
        # ════════════════════════════════════════════════════════
        #
        # 【作用】
        # W_o是一个可学习的线性变换，用于融合所有头的信息。
        # 如果没有这一层，相当于简单拼接8个头的输出，表达能力受限。
        #
        # 【类比】
        # 就像把8个专家的意见汇总后，再由一个决策者做最终判断。
        # W_o就是这个"决策者"，学习如何最佳地组合各个头的信息。
        #
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
