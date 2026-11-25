"""
Transformer模型核心架构
实现完整的Encoder-Decoder结构用于代码生成
"""

import sys
import torch
import torch.nn as nn
import math
from scripts.core.attention import MultiHeadAttention, PositionalEncoding
from scripts.utils.logger import logging_context


class RMSNorm(nn.Module):
    """
    RMSNorm（Root Mean Square Normalization，均方根归一化）

    ════════════════════════════════════════════════════════════
    📚 数学原理：RMSNorm vs LayerNorm
    ════════════════════════════════════════════════════════════

    LayerNorm（传统）：
        y = (x - μ) / sqrt(σ² + ε) * γ + β
        需要：减均值 μ、除标准差 σ、缩放 γ、偏移 β —— 共4个计算步骤

    RMSNorm（现代，Qwen3/LLaMA 标配）：
        y = x / sqrt(mean(x²) + ε) * γ
        只需要：除均方根 RMS、缩放 γ —— 去掉了均值中心化和偏移

    为什么去掉均值中心化？
    - 实验表明（Zhang & Sennrich, 2019）：LayerNorm 的收益主要来自
      缩放不变性，而非均值中心化。去掉中心化几乎不损失效果。
    - 少一次减法和一次加法，在深层网络中计算更快、显存更省。
    - 数值更稳定：不改变向量的方向信息，只归一化其模长。

    参数量：只有 γ（每个 d_model 维一个可学习缩放向量），无 β。
    ════════════════════════════════════════════════════════════

    Args:
        d_model (int): 归一化的特征维度
        eps (float): 数值稳定项，防止除零，默认 1e-6

    Example:
        >>> norm = RMSNorm(128)
        >>> x = torch.randn(1, 20, 128)
        >>> y = norm(x)          # y.shape == x.shape
        >>> # 每个位置的均方根接近1:
        >>> torch.sqrt((y ** 2).mean(dim=-1))
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习缩放参数 γ（无偏移 β）
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 计算每个位置特征维度的均方: mean(x²)
        #    注意: 不减均值（与LayerNorm的核心区别）
        x_squared_mean = x.pow(2).mean(dim=-1, keepdim=True)

        # 2. 除以均方根: x / sqrt(mean(x²) + ε)
        #    先转float32再归一化，避免fp16训练时数值溢出（工业惯例）
        x_normed = x * torch.rsqrt(x_squared_mean + self.eps)

        # 3. 乘以可学习缩放 γ
        return self.weight * x_normed


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder层 - 提取输入序列的特征表示
    
    核心功能：
    - 通过自注意力机制让每个token关注序列中的所有其他token
    - 通过前馈网络进行非线性特征变换
    - 使用残差连接和层归一化稳定训练
    
    架构流程：
    Input → [Self-Attention + Residual + LayerNorm] → [FFN + Residual + LayerNorm] → Output
    
    Example:
        >>> layer = TransformerEncoderLayer(d_model=128, nhead=8)
        [EncoderLayer] 初始化 (debug_mode=False)
          - d_model: 128
          - nhead: 8
          - dim_feedforward: 512
        >>> src = torch.randn(1, 20, 128)  # [batch, seq_len, d_model]
        >>> output, attn_weights = layer(src)
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 debug_mode: bool = False, norm_type: str = 'layernorm',
                 use_rope: bool = False, max_seq_length: int = 512):
        """
        初始化Encoder层
        
        Args:
            d_model (int): 模型维度，即token embedding的向量维度。
                          默认为128（教学简化值，真实模型通常512-4096）。
                          所有注意力头共享这个维度空间。
            
            nhead (int): 注意力头的数量，默认为8。
                        Multi-Head Attention将d_model分成nhead个头，
                        每个头的维度 = d_model / nhead = 128/8 = 16。
                        多头机制允许模型同时关注不同位置的不同信息。
            
            dim_feedforward (int): 前馈网络的隐藏层维度，默认为512。
                                  FFN结构: Linear(d_model→dim_feedforward) → ReLU → Dropout → Linear(dim_feedforward→d_model)
                                  通常设置为 d_model * 4 = 128 * 4 = 512。
            
            dropout (float): Dropout概率，默认为0.1。
                            用于防止过拟合，在训练时随机丢弃部分神经元。
            
            debug_mode (bool): 是否启用调试模式，默认为False。
                              启用后会输出详细的形状信息和中间结果。
            
            norm_type (str): 归一化类型，'layernorm'(默认) | 'rmsnorm'
            
            use_rope (bool): 是否使用RoPE旋转位置编码，默认False。
                            启用后位置信息由Attention内部旋转Q/K注入，
                            模型外部不应再叠加Sinusoidal PE。
            
            max_seq_length (int): RoPE预计算的最大位置数。
        
        Note:
            - d_model必须能被nhead整除，否则MultiHeadAttention会报错
            - dim_feedforward越大，模型表达能力越强，但计算量也越大
        """
        super().__init__()
        
        self.debug_mode = debug_mode
        
        print(f"[EncoderLayer] 初始化 (debug_mode={debug_mode})")
        print(f"  - d_model: {d_model}")
        print(f"  - nhead: {nhead}")
        print(f"  - dim_feedforward: {dim_feedforward}")
        
        # Self-Attention（use_rope=True时内部启用RoPE旋转Q/K）
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout,
                                            debug_mode=debug_mode,
                                            use_rope=use_rope,
                                            max_seq_length=max_seq_length)
        
        # Feed Forward Network (两层全连接)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Normalization（根据 norm_type 选择 LayerNorm 或 RMSNorm）
        # LayerNorm: 传统 Transformer 标准，中心化 + 缩放 + 偏移
        # RMSNorm:   现代大模型标准（Qwen3/LLaMA），只做均方根归一化，更快
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        elif norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"未知归一化类型: '{norm_type}', 可选: 'layernorm' | 'rmsnorm'")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Encoder层的前向传播
        
        处理流程：
        1. Self-Attention: 每个token关注序列中的所有token，捕获上下文关系
        2. Residual Connection: 原始输入 + 注意力输出，缓解梯度消失
        3. Layer Normalization: 归一化，稳定训练
        4. Feed Forward: 两层全连接网络，进行非线性特征变换
        5. Residual Connection + LayerNorm: 再次残差连接和归一化
        
        Args:
            src (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
                               - batch_size: 批次大小，通常为1（单样本推理）
                               - seq_len: 序列长度，如20个tokens
                               - d_model: 模型维度，如128维
                               示例: [1, 20, 128]
            
            src_mask (torch.Tensor, optional): 注意力掩码，形状为 [batch_size, 1, 1, seq_len]
                                              用于屏蔽padding位置或特定tokens。
                                              mask值为0表示可见，-inf表示屏蔽。
                                              如果为None，则所有位置都可见。
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 包含两个张量的元组
                - output (torch.Tensor): 输出张量，形状与输入相同 [batch_size, seq_len, d_model]
                                        经过自注意力和FFN处理后的特征表示
                - attn_weights (torch.Tensor): 注意力权重，形状为 [batch_size, nhead, seq_len, seq_len]
                                             展示了每个token如何关注其他tokens
                                             可用于可视化注意力分布
        
        Example:
            # >>> layer = TransformerEncoderLayer(d_model=128, nhead=8)
            # >>> src = torch.randn(1, 20, 128)  # [batch, seq_len, d_model]
            # >>> output, attn_weights = layer(src)
            # >>> output.shape
            # torch.Size([1, 20, 128])
            # >>> attn_weights.shape
            # torch.Size([1, 8, 20, 20])
        
        Note:
            - 输出维度与输入维度相同，便于堆叠多层Encoder
            - 注意力权重可用于分析模型关注了哪些位置
        """
        batch_size, seq_len, _ = src.shape
        
        if self.debug_mode:
            print(f"\n[EncoderLayer] 前向传播")
            print(f"  - Input shape: {src.shape}")
        
        # 1. Self-Attention + Residual + LayerNorm
        attn_output, attn_weights, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(attn_output)  # Residual connection
        src = self.norm1(src)  # Layer normalization
        
        if self.debug_mode:
            print(f"  - After Attention + Residual + Norm: {src.shape}")
        
        # 2. Feed Forward + Residual + LayerNorm
        ffn_output = self.ffn(src)
        src = src + ffn_output  # Residual connection
        src = self.norm2(src)  # Layer normalization
        
        if self.debug_mode:
            print(f"  - After FFN + Residual + Norm: {src.shape}")
        
        return src, attn_weights


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder层 - 根据Encoder输出和已生成tokens预测下一个token
    
    核心功能：
    - Masked Self-Attention: 只能看到之前的tokens，防止"偷看"未来
    - Cross-Attention: 关注Encoder的输出，理解源序列信息
    - Feed Forward: 非线性特征变换
    - 三重残差连接和层归一化
    
    架构流程：
    Input → [Masked Self-Attn + Residual + LayerNorm] → 
            [Cross-Attn(关注Encoder) + Residual + LayerNorm] → 
            [FFN + Residual + LayerNorm] → Output
    
    与Encoder的区别：
    - Encoder只有1个注意力机制（Self-Attention）
    - Decoder有3个注意力机制（Masked Self-Attn + Cross-Attn + FFN）
    - Decoder需要额外的因果掩码防止看到未来tokens
    
    Example:
        >>> decoder_layer = TransformerDecoderLayer(d_model=128, nhead=8)
        >>> tgt = torch.randn(1, 15, 128)    # Decoder输入
        >>> memory = torch.randn(1, 20, 128) # Encoder输出
        >>> output, cross_attn_weights = decoder_layer(tgt, memory)
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 debug_mode: bool = False, norm_type: str = 'layernorm',
                 use_rope: bool = False, max_seq_length: int = 512):
        """
        初始化Decoder层
        
        Args:
            d_model (int): 模型维度，默认为128。必须与Encoder的d_model一致。
            
            nhead (int): 注意力头数量，默认为8。
                        Self-Attention和Cross-Attention都使用相同的头数。
            
            dim_feedforward (int): 前馈网络隐藏层维度，默认为512。
                                  通常为 d_model * 4。
            
            dropout (float): Dropout概率，默认为0.1。
            
            debug_mode (bool): 是否启用调试模式，默认为False。
            
            norm_type (str): 归一化类型，'layernorm'(默认) | 'rmsnorm'
            
            use_rope (bool): 是否使用RoPE旋转位置编码，默认False。
                            启用后 Self-Attention 的 Q/K 会被旋转；
                            Cross-Attention 的 K/V 来自 Encoder（也已用相同
                            的 RoPE 旋转），无需特殊处理。
            
            max_seq_length (int): RoPE预计算的最大位置数。
        
        Note:
            - Decoder比Encoder多一个LayerNorm（共3个），因为有3个子层
            - Cross-Attention的Q来自Decoder，K和V来自Encoder输出
        """
        super().__init__()
        
        self.debug_mode = debug_mode
        
        print(f"[DecoderLayer] 初始化 (debug_mode={debug_mode})")
        
        # Masked Self-Attention（解码时只能看到之前的token）
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout,
                                            debug_mode=debug_mode,
                                            use_rope=use_rope,
                                            max_seq_length=max_seq_length)
        
        # Cross-Attention（关注Encoder的输出；Encoder输出已被同样的RoPE旋转）
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout,
                                             debug_mode=debug_mode,
                                             use_rope=use_rope,
                                             max_seq_length=max_seq_length)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Normalization（根据 norm_type 选择 LayerNorm 或 RMSNorm）
        # Decoder 有 3 个子层，因此需要 3 个归一化层
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        elif norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"未知归一化类型: '{norm_type}', 可选: 'layernorm' | 'rmsnorm'")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Decoder层的前向传播
        
        处理流程（3个注意力子层）：
        1. Masked Self-Attention: 
           - Q=K=V=tgt，使用因果掩码防止看到未来tokens
           - 让每个位置只关注它之前的位置
        
        2. Cross-Attention (关键！):
           - Q=tgt (Decoder的当前状态)
           - K=V=memory (Encoder的输出)
           - 让Decoder关注源序列的关键信息
           - 这是Encoder和Decoder之间的桥梁
        
        3. Feed Forward:
           - 两层全连接网络，进行非线性变换
        
        每个子层后都有：Residual Connection + Layer Normalization
        
        Args:
            tgt (torch.Tensor): Decoder输入，形状为 [batch_size, tgt_seq_len, d_model]
                               已生成的tokens的embedding表示。
                               示例: [1, 15, 128] 表示15个已生成的tokens
            
            memory (torch.Tensor): Encoder输出，形状为 [batch_size, src_seq_len, d_model]
                                  源序列的特征表示，由Encoder生成。
                                  示例: [1, 20, 128] 表示20个源tokens的特征
                                  在Cross-Attention中作为K和V
            
            tgt_mask (torch.Tensor, optional): Decoder自注意力掩码，形状为 [tgt_seq_len, tgt_seq_len]
                                              必须是因果掩码（上三角为-inf），确保每个位置只能看到之前的位置。
                                              通过 model.generate_square_subsequent_mask() 生成。
            
            memory_mask (torch.Tensor, optional): Cross-attention掩码，形状为 [src_seq_len, src_seq_len]
                                                 通常用于屏蔽Encoder输出中的padding位置。
                                                 如果为None，则所有位置都可见。
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 包含两个张量的元组
                - output (torch.Tensor): 输出张量，形状为 [batch_size, tgt_seq_len, d_model]
                                        Decoder处理后的特征表示
                - cross_attn_weights (torch.Tensor): Cross-attention权重，形状为 [batch_size, nhead, tgt_seq_len, src_seq_len]
                                                   展示了Decoder的每个位置如何关注Encoder的各个位置
                                                   可用于可视化"对齐"关系（哪个target token关注哪个source token）
        
        Note:
            - Cross-attention权重可用于分析模型如何将target与source对齐
            - 例如：生成"User"时可能关注source中的"User"token
        """
        if self.debug_mode:
            print(f"\n[DecoderLayer] 前向传播")
            print(f"  - Target shape: {tgt.shape}")
            print(f"  - Memory shape: {memory.shape}")
        
        # 1. Masked Self-Attention
        self_attn_output, _, present_self_kv = self.self_attn(
            tgt, tgt, tgt, tgt_mask, past_key_value=past_key_value
        )
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.norm1(tgt)
        
        # 2. Cross-Attention（关注Encoder输出）
        cross_attn_output, cross_attn_weights, _ = self.cross_attn(
            tgt, memory, memory, memory_mask
        )
        tgt = tgt + self.dropout(cross_attn_output)
        tgt = self.norm2(tgt)
        
        # 3. Feed Forward
        ffn_output = self.ffn(tgt)
        tgt = tgt + ffn_output
        tgt = self.norm3(tgt)
        
        if self.debug_mode:
            print(f"  - Output shape: {tgt.shape}")
        
        return tgt, cross_attn_weights, present_self_kv


class TransformerModel(nn.Module):
    """
    完整的Transformer模型 - Encoder-Decoder架构用于代码生成
    
    整体架构：
    ┌─────────────────────────────────────────────┐
    │  Input Tokens (src)                         │
    │       ↓                                     │
    │  Embedding + Positional Encoding            │
    │       ↓                                     │
    │  Encoder Layer × N  ← 提取源序列特征         │
    │       ↓                                     │
    │  Memory (编码后的源序列表示)                  │
    │       ↓                                     │
    │  Decoder Layer × N  ← 结合源信息和已生成tokens│
    │       ↓                                     │
    │  Output Projection   ← 映射到词汇表          │
    │       ↓                                     │
    │  Logits → Softmax → Next Token Prediction  │
    └─────────────────────────────────────────────┘
    
    核心组件：
    1. Embedding层: token ID → d_model维向量
    2. 位置编码: 添加序列位置信息（因为Transformer没有顺序概念）
    3. Encoder × N: 理解输入序列（prompt/code）
    4. Decoder × N: 生成输出序列（generated code）
    5. 输出投影: d_model → vocab_size，得到每个token的概率
    
    两种使用模式：
    - 训练模式 (forward): 同时输入src和tgt，计算loss
    - 推理模式 (generate_step): 只输入src，逐步生成tgt
    
    Example:
        >>> model = TransformerModel(
        ...     vocab_size=1000,
        ...     d_model=128,
        ...     nhead=8,
        ...     num_encoder_layers=2,
        ...     num_decoder_layers=2
        ... )
        >>> # 训练模式
        >>> src = torch.randint(0, 1000, (1, 20))  # prompt
        >>> tgt = torch.randint(0, 1000, (1, 15))  # target code
        >>> logits, enc_weights, dec_weights = model(src, tgt)
        >>> 
        >>> # 推理模式
        >>> model.clear_cache()
        >>> generated = torch.tensor([[BOS_TOKEN_ID]])
        >>> for _ in range(50):
        ...     logits = model.generate_step(src, generated)
        ...     next_token = torch.argmax(logits, dim=-1)
        ...     generated = torch.cat([generated, next_token], dim=1)
    """
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, 
                 nhead: int = 8, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_length: int = 128,
                 debug_mode: bool = False, norm_type: str = 'layernorm',
                 position_embedding_type: str = 'sinusoidal'):
        """
        初始化完整的Transformer模型
        
        Args:
            vocab_size (int): 词汇表大小，默认为1000。
                             必须与tokenizer的词汇表大小一致。
                             决定了embedding层的输入范围和output_projection的输出维度。
            
            d_model (int): 模型维度，默认为128。
                          token embedding的向量维度，也是所有层的隐藏状态维度。
                          教学简化值，真实模型通常512-4096。
                          必须能被nhead整除。
            
            nhead (int): 注意力头数量，默认为8。
                        Multi-Head Attention将d_model分成nhead个头并行计算。
                        每个头的维度 = d_model / nhead = 128/8 = 16。
            
            num_encoder_layers (int): Encoder层数，默认为2。
                                     层数越多，模型表达能力越强，但训练难度也越大。
                                     真实模型通常6-96层（如GPT-3有96层）。
            
            num_decoder_layers (int): Decoder层数，默认为2。
                                     通常与num_encoder_layers相同或略少。
            
            dim_feedforward (int): 前馈网络隐藏层维度，默认为512。
                                  通常为 d_model * 4 = 128 * 4 = 512。
                                  更大的值提供更强的非线性变换能力。
            
            dropout (float): Dropout概率，默认为0.1。
                            应用于attention输出、FFN中间层等位置。
                            防止过拟合，训练时随机丢弃部分神经元。
            
            max_seq_length (int): 最大序列长度，默认为128。
                                 位置编码的最大长度限制。
                                 如果输入序列超过此长度，位置编码会出错。
                                 本质上就是"上下文长度"的概念。
            
            debug_mode (bool): 是否启用调试模式，默认为False。
                              启用后会输出详细的形状信息和每层的处理过程。
        
        Note:
            - 总参数量 ≈ vocab_size * d_model (embedding) + 
                         num_encoder_layers * (d_model^2 * 4) +
                         num_decoder_layers * (d_model^2 * 6) +
                         d_model * vocab_size (output_projection)
            - 对于本配置：约 1000*128 + 2*(128^2*4) + 2*(128^2*6) + 128*1000 ≈ 500K参数
            - 真实LLM有数十亿到数万亿参数
        """
        super().__init__()
        
        self.debug_mode = debug_mode
        
        print(f"\n{'='*60}")
        print(f"[TransformerModel] 初始化完整模型 (debug_mode={debug_mode})")
        print(f"{'='*60}")
        print(f"  - 词汇表大小: {vocab_size}")
        print(f"  - 模型维度: {d_model}")
        print(f"  - 注意力头数: {nhead}")
        print(f"  - Encoder层数: {num_encoder_layers}")
        print(f"  - Decoder层数: {num_decoder_layers}")
        print(f"  - FeedForward维度: {dim_feedforward}")
        print(f"  - 最大序列长度: {max_seq_length}")
        print(f"  - 归一化类型: {norm_type}")
        print(f"  - 位置编码: {position_embedding_type}")
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.nhead = nhead                    # KV Cache 需要：注意力头数
        self.d_k = d_model // nhead           # KV Cache 需要：每个头的维度
        self.num_decoder_layers = num_decoder_layers  # KV Cache 需要：Decoder 层数
        
        # 位置编码类型:
        # - 'sinusoidal': 传统正弦/余弦绝对位置编码（加到 embedding 上）
        # - 'rope':       旋转位置编码（在 Attention 内部旋转 Q/K，不再叠加 Sinusoidal PE）
        if position_embedding_type not in ('sinusoidal', 'rope'):
            raise ValueError(
                f"未知位置编码类型: '{position_embedding_type}', "
                f"可选: 'sinusoidal' | 'rope'"
            )
        self.position_embedding_type = position_embedding_type
        self.use_rope = position_embedding_type == 'rope'
        
        # Embedding层：将token ID转换为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码（sinusoidal 模式下使用；rope 模式下位置信息由 Attention 内部的
        # RoPE 注入，embedding 只做缩放，不再叠加 PE）
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                    debug_mode=debug_mode, norm_type=norm_type,
                                    use_rope=self.use_rope,
                                    max_seq_length=max_seq_length)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                    debug_mode=debug_mode, norm_type=norm_type,
                                    use_rope=self.use_rope,
                                    max_seq_length=max_seq_length)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出投影层：将隐藏状态映射到词汇表
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._init_weights()
        
        print(f"{'='*60}\n")
    
    def _init_weights(self):
        """
        初始化模型权重
        
        使用Xavier均匀初始化（Glorot初始化）：
        - 适用于sigmoid/tanh等激活函数
        - 保持输入和输出的方差一致
        - 有助于梯度流动，加速收敛
        
        初始化规则：
        - 只对维度>1的张量初始化（如Linear层的权重矩阵）
        - bias通常初始化为0（nn.Linear默认行为）
        
        Note:
            - Xavier初始化公式: W ~ U[-a, a], 其中 a = gain * sqrt(6 / (fan_in + fan_out))
            - 对于ReLU激活函数，也可以使用Kaiming初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成因果掩码（Causal Mask）用于Decoder自注意力
        
        核心作用：
        - 确保每个位置只能看到它之前的位置（防止"偷看"未来）
        - 在训练时防止模型作弊看到target的未来tokens
        - 在推理时自然满足（因为是一次生成一个token）
        
        掩码结构：
        ```
        sz=4时的mask矩阵：
        [[0,   -inf, -inf, -inf],
         [0,    0,   -inf, -inf],
         [0,    0,    0,   -inf],
         [0,    0,    0,    0  ]]
        
        含义：
        - 位置0只能看到自己
        - 位置1能看到位置0和1
        - 位置2能看到位置0、1、2
        - 位置3能看到所有位置
        ```
        
        Args:
            sz (int): 序列长度，即要生成的掩码的大小。
                     通常等于target序列的长度。
                     示例: sz=15 表示生成长度为15的因果掩码
        
        Returns:
            torch.Tensor: 形状为 [sz, sz] 的浮点掩码矩阵
                         - 上三角部分（diagonal=1以上）为 float('-inf')，表示屏蔽
                         - 下三角部分（包括对角线）为 0.0，表示可见
                         
                         在注意力计算中：
                         attention_scores + mask
                         → 屏蔽位置的分数变成 -inf
                         → softmax后权重趋近于0
                         → 模型不会关注这些位置
        
        Example:
            >>> model = TransformerModel()
            >>> mask = model.generate_square_subsequent_mask(4)
            >>> print(mask)
            tensor([[0., -inf, -inf, -inf],
                    [0., 0., -inf, -inf],
                    [0., 0., 0., -inf],
                    [0., 0., 0., 0.]])
            
            >>> # 在Decoder中使用
            >>> tgt_seq_len = generated.size(1)
            >>> tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len)
            >>> output, _ = decoder_layer(tgt, memory, tgt_mask)
        
        Note:
            - 这个掩码是Decoder的核心机制之一
            - 没有它，模型会在训练时"作弊"看到未来的tokens
            - 掩码在每次生成新token时都需要重新创建（因为序列长度在变化）
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        # 将True的位置设为0（屏蔽），False的位置设为1
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Encoder过程 - 将输入序列编码为特征表示
        
        处理流程：
        1. Embedding: token IDs → d_model维向量
        2. Positional Encoding: 添加位置信息（因为Transformer没有顺序概念）
        3. Encoder Layers × N: 通过多层Encoder提取特征
        
        Args:
            src (torch.Tensor): 源序列的token IDs，形状为 [batch_size, src_seq_len]
                               通常是prompt或输入代码的分词结果。
                               示例: [1, 20] 表示1个样本，20个tokens
                               值范围: [0, vocab_size-1]
            
            src_mask (torch.Tensor, optional): 注意力掩码，形状为 [batch_size, 1, 1, src_seq_len]
                                              用于屏蔽padding位置。
                                              如果为None，则所有位置都可见。
                                              通常由tokenizer的attention_mask转换而来。
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: 包含两个元素的元组
                - memory (torch.Tensor): Encoder的输出，形状为 [batch_size, src_seq_len, d_model]
                                        源序列的特征表示，每个token都有一个d_model维的向量
                                        将作为Decoder中Cross-Attention的K和V
                                        示例: [1, 20, 128]
                
                - encoder_attn_weights (List[torch.Tensor]): 每层Encoder的注意力权重列表
                                                            长度为num_encoder_layers
                                                            每个元素形状为 [batch_size, nhead, src_seq_len, src_seq_len]
                                                            可用于可视化Encoder内部的注意力分布
        
        Example:
            >>> model = TransformerModel()
            >>> src = torch.randint(0, 1000, (1, 20))  # [batch, seq_len]
            >>> memory, enc_weights = model.encode(src)
            >>> print(memory.shape)        # [1, 20, 128]
            >>> print(len(enc_weights))    # 2 (num_encoder_layers)
        
        Note:
            - Embedding后乘以 sqrt(d_model) 是为了缩放，防止数值过大
            - memory是Encoder的最终输出，包含了源序列的上下文信息
            - Decoder会通过Cross-Attention关注memory的不同部分
        """
        if self.debug_mode:
            print(f"\n[Transformer] 开始Encoder过程")
            print(f"  - Source shape: {src.shape}")
        
        # 1. Embedding
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        if self.debug_mode:
            print(f"  - After embedding: {src_embedded.shape}")
        
        # 2. 位置编码
        # 位置编码：rope 模式下位置信息由 Attention 内部 RoPE 注入，这里只做 dropout
        if self.use_rope:
            src_encoded = self.pos_encoder.dropout(src_embedded)
        else:
            src_encoded = self.pos_encoder(src_embedded)
        if self.debug_mode:
            print(f"  - After positional encoding: {src_encoded.shape}")
        
        # 3. 通过多层Encoder
        memory = src_encoded
        encoder_attn_weights = []
        
        for i, layer in enumerate(self.encoder_layers):
            if self.debug_mode:
                print(f"\n  --- Encoder Layer {i+1} ---")
            memory, attn_weights = layer(memory, src_mask)
            encoder_attn_weights.append(attn_weights)
        
        if self.debug_mode:
            print(f"\n[Transformer] Encoder完成")
            print(f"  - Memory shape: {memory.shape}")
        
        return memory, encoder_attn_weights
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        """
        Decoder过程 - 根据Encoder输出和已生成tokens预测下一个token
        
        处理流程：
        1. Embedding: target token IDs → d_model维向量
        2. Positional Encoding: 添加位置信息
        3. Decoder Layers × N: 通过多层Decoder生成输出
           - 每层包含：Masked Self-Attention + Cross-Attention + FFN
        
        Args:
            tgt (torch.Tensor): 目标序列的token IDs，形状为 [batch_size, tgt_seq_len]
                               在训练时是完整的target序列（teacher forcing）
                               在推理时是已生成的部分序列
                               示例: [1, 15] 表示1个样本，15个tokens
            
            memory (torch.Tensor): Encoder的输出，形状为 [batch_size, src_seq_len, d_model]
                                  由 encode() 方法生成
                                  在Cross-Attention中作为K和V
                                  示例: [1, 20, 128]
            
            tgt_mask (torch.Tensor, optional): Decoder自注意力掩码，形状为 [tgt_seq_len, tgt_seq_len]
                                              必须是因果掩码，通过 generate_square_subsequent_mask() 生成
                                              确保每个位置只能看到之前的位置
            
            memory_mask (torch.Tensor, optional): Cross-attention掩码，形状为 [src_seq_len, src_seq_len]
                                                 用于屏蔽Encoder输出中的padding位置
                                                 如果为None，则所有位置都可见
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: 包含两个元素的元组
                - output (torch.Tensor): Decoder的输出，形状为 [batch_size, tgt_seq_len, d_model]
                                        每个位置的d_model维向量，包含了源序列信息和已生成tokens的信息
                                        将通过output_projection映射到词汇表
                                        示例: [1, 15, 128]
                
                - decoder_attn_weights (List[torch.Tensor]): 每层Decoder的cross-attention权重列表
                                                            长度为num_decoder_layers
                                                            每个元素形状为 [batch_size, nhead, tgt_seq_len, src_seq_len]
                                                            展示了Decoder如何关注Encoder的各个位置
                                                            可用于可视化"对齐"关系
        
        Example:
            >>> model = TransformerModel()
            >>> # 先编码源序列
            >>> memory, _ = model.encode(src)  # [1, 20, 128]
            >>> # 再解码目标序列
            >>> tgt = torch.randint(0, 1000, (1, 15))  # [1, 15]
            >>> tgt_mask = model.generate_square_subsequent_mask(15)
            >>> output, dec_weights = model.decode(tgt, memory, tgt_mask)
            >>> print(output.shape)       # [1, 15, 128]
            >>> print(len(dec_weights))   # 2 (num_decoder_layers)
        
        Note:
            - Decoder的输出还需要通过output_projection才能得到vocab概率分布
            - Cross-attention权重可用于分析模型如何将target与source对齐
            - 在推理时，tgt是逐步增长的（每次增加一个token）
        """
        if self.debug_mode:
            print(f"\n[Transformer] 开始Decoder过程")
            print(f"  - Target shape: {tgt.shape}")
            print(f"  - Memory shape: {memory.shape}")
        
        # 1. Embedding
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        if self.debug_mode:
            print(f"  - After embedding: {tgt_embedded.shape}")
        
        # 2. 位置编码
        # 位置编码：rope 模式下位置信息由 Attention 内部 RoPE 注入，这里只做 dropout
        if self.use_rope:
            tgt_encoded = self.pos_encoder.dropout(tgt_embedded)
        else:
            tgt_encoded = self.pos_encoder(tgt_embedded)
        if self.debug_mode:
            print(f"  - After positional encoding: {tgt_encoded.shape}")
        
        # 3. 通过多层Decoder
        output = tgt_encoded
        decoder_attn_weights = []
        
        for i, layer in enumerate(self.decoder_layers):
            if self.debug_mode:
                print(f"\n  --- Decoder Layer {i+1} ---")
            output, attn_weights, _ = layer(output, memory, tgt_mask, memory_mask)
            decoder_attn_weights.append(attn_weights)
        
        if self.debug_mode:
            print(f"\n[Transformer] Decoder完成")
            print(f"  - Output shape: {output.shape}")
        
        return output, decoder_attn_weights
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None):
        """
        完整的前向传播（训练时使用）
        
        这是Transformer的核心方法，用于训练阶段。
        同时处理源序列和目标序列，计算每个位置的logits。
        
        处理流程：
        1. Encode: src → memory (理解输入)
        2. Decode: tgt + memory → output (生成输出)
        3. Project: output → logits (映射到词汇表)
        
        Args:
            src (torch.Tensor): 源序列的token IDs，形状为 [batch_size, src_seq_len]
                               通常是prompt或输入代码。
                               示例: [1, 20] 表示1个样本，20个tokens
            
            tgt (torch.Tensor): 目标序列的token IDs，形状为 [batch_size, tgt_seq_len]
                               期望的输出代码（ground truth）。
                               在训练时使用teacher forcing，一次性输入完整序列。
                               示例: [1, 15] 表示1个样本，15个tokens
            
            src_mask (torch.Tensor, optional): 源序列注意力掩码
                                              用于屏蔽src中的padding位置
            
            tgt_mask (torch.Tensor, optional): 目标序列因果掩码
                                              必须是因果掩码，通过 generate_square_subsequent_mask() 生成
                                              防止Decoder看到未来的tokens
            
        Returns:
            Tuple[torch.Tensor, List, List]: 包含三个元素的元组
                - logits (torch.Tensor): 未归一化的概率分数，形状为 [batch_size, tgt_seq_len, vocab_size]
                                        每个位置的vocab_size维向量，表示该位置是每个token的"分数"
                                        需要通过softmax转换为概率分布
                                        示例: [1, 15, 1000]
                
                - encoder_weights (List): Encoder的注意力权重列表
                - decoder_weights (List): Decoder的cross-attention权重列表
        
        Example:
            >>> model = TransformerModel()
            >>> src = torch.randint(0, 1000, (1, 20))  # prompt
            >>> tgt = torch.randint(0, 1000, (1, 15))  # target code
            >>> tgt_mask = model.generate_square_subsequent_mask(15)
            >>> logits, enc_w, dec_w = model(src, tgt, tgt_mask=tgt_mask)
            >>> print(logits.shape)  # [1, 15, 1000]
            >>> 
            >>> # 计算loss
            >>> loss_fn = nn.CrossEntropyLoss()
            >>> loss = loss_fn(logits.view(-1, 1000), tgt.view(-1))
        
        Note:
            - 训练时使用完整的tgt序列（teacher forcing）
            - logits需要与target比较计算cross-entropy loss
            - 这是监督学习的关键步骤
        """
        # Encode
        memory, encoder_weights = self.encode(src, src_mask)
        
        # Decode
        output, decoder_weights = self.decode(tgt, memory, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        if self.debug_mode:
            print(f"\n[Transformer] 最终输出")
            print(f"  - Logits shape: {logits.shape}")
        
        return logits, encoder_weights, decoder_weights
    
    def generate_step(self, src: torch.Tensor, generated: torch.Tensor,
                      src_mask: torch.Tensor = None,
                      kv_cache_manager=None,
                      use_cache: bool = True):
        """
        单步生成（推理时使用）
        
        这是Transformer的推理方法，用于逐步生成代码。
        每次调用只预测下一个token，然后将其添加到generated序列中。
        
        与forward的区别：
        - forward: 训练时用，一次性处理完整tgt序列
        - generate_step: 推理时用，每次只预测一个token
        
        优化策略：
        - Encoder输出被缓存（_cached_memory），避免重复计算
        - 因为src在生成过程中不会改变
        
        Args:
            src (torch.Tensor): 源序列的token IDs，形状为 [batch_size, src_seq_len]
                               prompt或输入代码，在整个生成过程中保持不变。
                               示例: [1, 20]
            
            generated (torch.Tensor): 已生成的序列，形状为 [batch_size, current_seq_len]
                                     初始时只包含BOS token
                                     随着生成进行逐渐增长
                                     示例: 
                                     - 第1步: [1, 1] (只有BOS)
                                     - 第2步: [1, 2] (BOS + "public")
                                     - 第3步: [1, 3] (BOS + "public" + "class")
            
            src_mask (torch.Tensor, optional): 源序列注意力掩码
                                              用于屏蔽src中的padding位置
            
        Returns:
            torch.Tensor: 下一个token的logits，形状为 [batch_size, vocab_size]
                         表示下一个位置是每个token的分数
                         需要通过softmax转换为概率，然后采样或argmax选择下一个token
                         示例: [1, 1000]
        
        Example:
            >>> model = TransformerModel(debug_mode=False)
            >>> src = torch.randint(0, 1000, (1, 20))  # prompt
            >>> BOS_TOKEN_ID = 2
            >>> generated = torch.tensor([[BOS_TOKEN_ID]])  # [1, 1]
            >>> logits = model.generate_step(src, generated)
            >>> logits.shape
            torch.Size([1, 1000])
        
        Note:
            - 这是autoregressive（自回归）生成：每个token依赖于之前的所有tokens
            - 需要清除缓存才能开始新的生成任务（调用 clear_cache()）
            - 可以使用不同的采样策略（greedy、top-k、top-p等）来选择next_token
        """
        # Encode（只需要做一次，缓存 memory）
        if not hasattr(self, '_cached_memory'):
            self._cached_memory, _ = self.encode(src, src_mask)
        
        memory = self._cached_memory
        
        # ========== 无缓存路径（对比基准）：整段序列喂给 Decoder 重算所有位置 ==========
        if not use_cache:
            tgt_seq_len = generated.size(1)
            tgt_mask_full = self.generate_square_subsequent_mask(tgt_seq_len).to(generated.device)
            output, _ = self.decode(generated, memory, tgt_mask_full)
            logits = self.output_projection(output[:, -1, :])
            return logits, None
        
        # ========== KV Cache 路径 ==========
        # 如果没有传入 kv_cache_manager，创建一个临时的（兼容旧调用，但不会跨步复用）
        if kv_cache_manager is None:
            from scripts.optimization.kv_cache import KVCacheManager
            device = generated.device
            kv_cache_manager = KVCacheManager(
                num_layers=self.num_decoder_layers,
                batch_size=generated.size(0),
                num_heads=self.nhead,
                max_seq_len=self.max_seq_length,
                head_dim=self.d_k,
                device=device
            )
            kv_cache_manager.initialize()
        
        # 关键：只取最后一个 token 作为当前输入 [batch, 1]
        # 历史位置的 K/V 已经缓存在 kv_cache_manager 中，无需重算
        last_token = generated[:, -1:]
        # 长度为 1 的因果掩码：单 token 可见自身及所有历史缓存位置（广播到 [1, p+1]）
        tgt_mask = self.generate_square_subsequent_mask(1).to(generated.device)
        
        # 当前 token 的绝对位置 = 缓存中已有的长度
        position = kv_cache_manager.get_current_length()
        
        tgt_embedded = self.embedding(last_token) * math.sqrt(self.d_model)
        
        if self.use_rope:
            # RoPE 模式：位置信息由 Attention 内部旋转注入，embedding 只做 dropout
            tgt_encoded = self.pos_encoder.dropout(tgt_embedded)
        else:
            # Sinusoidal 模式：单 token 输入时 pos_encoder 会加 PE(0)，
            # 必须手动取该 token 真实位置 t 的 PE(t)
            position = kv_cache_manager.get_current_length()
            pe = self.pos_encoder.pe[:, position:position + 1, :]  # [1, 1, d_model]
            tgt_encoded = tgt_embedded + pe.to(tgt_embedded.dtype)
        
        # 先快照所有层的历史 K/V，再逐层计算+更新
        # （current_length 在各层间共享，若边读边写会导致后面的层读到错误长度）
        past_kvs = [kv_cache_manager.get_layer(i) for i in range(self.num_decoder_layers)]
        
        # 逐层 Decoder，传递 KV Cache
        output = tgt_encoded
        for i, layer in enumerate(self.decoder_layers):
            output, cross_attn_weights, present_self_kv = layer(
                output, memory, tgt_mask, None, past_key_value=past_kvs[i]
            )
            # 只把新 token 产生的 K/V（最后一个位置）写入缓存，避免重复写入历史
            kv_cache_manager.update_layer(
                i,
                present_self_kv[0][:, :, -1:, :],
                present_self_kv[1][:, :, -1:, :],
                position=position
            )
        
        # 只取最后一个位置（序列长度为1，即 output[:, -1, :] == output[:, 0, :]）
        last_output = output[:, -1, :]  # [batch_size, d_model]
        
        # 投影到词汇表
        logits = self.output_projection(last_output)  # [batch_size, vocab_size]
        
        return logits, kv_cache_manager
    
    def clear_cache(self):
        """
        清除缓存（用于新的生成任务）
        
        作用：
        - 删除_cached_memory属性，强制下一次generate_step时重新编码src
        - 在开始新的生成任务前必须调用此方法
        
        为什么需要缓存？
        - Encoder只需要运行一次（src不变）
        - 缓存memory可以避免重复计算，加速生成过程
        
        为什么需要清除？
        - 不同的生成任务有不同的src
        - 如果不清除，会错误地使用上一个任务的memory
        
        Example:
            >>> model = TransformerModel()
            >>> 
            >>> # 第一个生成任务
            >>> src1 = torch.randint(0, 1000, (1, 20))
            >>> generated1 = torch.tensor([[BOS_TOKEN_ID]])
            >>> for _ in range(50):
            ...     logits = model.generate_step(src1, generated1)
            ...     next_token = torch.argmax(logits, dim=-1, keepdim=True)
            ...     generated1 = torch.cat([generated1, next_token], dim=1)
            >>> 
            >>> # 清除缓存，准备第二个任务
            >>> model.clear_cache()
            >>> 
            >>> # 第二个生成任务
            >>> src2 = torch.randint(0, 1000, (1, 15))
            >>> generated2 = torch.tensor([[BOS_TOKEN_ID]])
            >>> for _ in range(50):
            ...     logits = model.generate_step(src2, generated2)
            ...     ...
        
        Note:
            - 忘记清除缓存会导致生成结果错误（使用了错误的memory）
            - 这是一个常见的bug来源
        """
        if hasattr(self, '_cached_memory'):
            delattr(self, '_cached_memory')


# 测试代码
if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        print("="*60)
        print("测试Transformer模型")
        print("="*60)
        
        # 参数
        vocab_size = 1000
        d_model = 128
        nhead = 8
        batch_size = 1
        src_seq_len = 20
        tgt_seq_len = 15
        
        # 创建模型
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        # 创建模拟输入
        src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
        
        print(f"\n输入:")
        print(f"  Source shape: {src.shape}")
        print(f"  Target shape: {tgt.shape}")
        
        # 创建掩码
        src_mask = torch.ones(batch_size, 1, 1, src_seq_len)
        tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len)
        
        print(f"\nMasks:")
        print(f"  Source mask shape: {src_mask.shape}")
        print(f"  Target mask shape: {tgt_mask.shape}")
        
        # 前向传播
        logits, enc_weights, dec_weights = model(src, tgt, src_mask, tgt_mask)
        
        print(f"\n输出:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Vocabulary size: {vocab_size}")
        
        # 转换为概率
        probs = torch.softmax(logits, dim=-1)
        print(f"  Probabilities sum (should be 1.0): {probs.sum(dim=-1).mean().item():.4f}")
        
        # 获取最可能的token
        predicted_tokens = torch.argmax(logits, dim=-1)
        print(f"  Predicted tokens shape: {predicted_tokens.shape}")
        print(f"  Sample predicted token IDs: {predicted_tokens[0, :5].tolist()}")
        
        print(f"\n{'='*60}")
        print(f"[OK] Transformer测试完成！")
        print(f"{'='*60}")
        print(f"\n提示: 完整日志已自动保存到 logs/ 目录")
        
        # ==================== KV Cache 加速演示 ====================
        print("\n" + "="*60)
        print("KV Cache 加速演示")
        print("="*60)
        
        import time
        from scripts.optimization.kv_cache import KVCacheManager
        
        model_kv = TransformerModel(vocab_size=1000, d_model=128, nhead=8,
                                    num_encoder_layers=2, num_decoder_layers=2)
        src_kv = torch.randint(0, 1000, (1, 20))
        BOS_TOKEN_ID = 2
        STEPS = 30
        
        # 方式一：无 KV Cache（传统方式：每步把整个 generated 序列喂给 Decoder 重算全部位置）
        model_kv.clear_cache()
        generated = torch.tensor([[BOS_TOKEN_ID]])
        start = time.time()
        for _ in range(STEPS):
            logits, _ = model_kv.generate_step(src_kv, generated, use_cache=False)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        time_no_cache = time.time() - start
        
        # 方式二：有 KV Cache（每步只输入最新 token，历史 K/V 直接复用）
        model_kv.clear_cache()
        generated = torch.tensor([[BOS_TOKEN_ID]])
        kv_manager = KVCacheManager(num_layers=2, batch_size=1, num_heads=8,
                                    max_seq_len=128, head_dim=16,
                                    device=torch.device('cpu'))
        kv_manager.initialize()
        start = time.time()
        for _ in range(STEPS):
            logits, kv_manager = model_kv.generate_step(src_kv, generated, kv_cache_manager=kv_manager)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        time_with_cache = time.time() - start
        
        print(f"\n生成 {STEPS} 个 token：")
        print(f"  无 KV Cache 耗时: {time_no_cache:.4f}s")
        print(f"  有 KV Cache 耗时: {time_with_cache:.4f}s")
        if time_with_cache > 0:
            print(f"  加速比: {time_no_cache / time_with_cache:.2f}x")
        
        # ==================== RMSNorm 验证 ====================
        print("\n" + "="*60)
        print("RMSNorm 验证 (LayerNorm vs RMSNorm)")
        print("="*60)
        
        rms = RMSNorm(128)
        ln = nn.LayerNorm(128)
        x = torch.randn(4, 10, 128)
        
        y_rms = rms(x)
        y_ln = ln(x)
        
        # RMSNorm 输出的均方根应接近 1 * |γ|=1（未中心化）
        rms_per_pos = torch.sqrt(y_rms.pow(2).mean(dim=-1))
        # LayerNorm 输出的标准差应接近 1（中心化后）
        std_per_pos = y_ln.std(dim=-1, unbiased=False)
        
        print(f"  输入 shape: {tuple(x.shape)}")
        print(f"  RMSNorm 输出均方根 (应≈1): {rms_per_pos.mean().item():.4f}")
        print(f"  LayerNorm 输出标准差 (应≈1): {std_per_pos.mean().item():.4f}")
        print(f"  RMSNorm 参数量: {sum(p.numel() for p in rms.parameters())} (仅γ, 无β)")
        print(f"  LayerNorm 参数量: {sum(p.numel() for p in ln.parameters())} (γ + β)")
        
        assert abs(rms_per_pos.mean().item() - 1.0) < 0.1, "RMSNorm归一化异常"
        assert abs(std_per_pos.mean().item() - 1.0) < 0.1, "LayerNorm归一化异常"
        print(f"  [OK] RMSNorm 验证通过")
        
        # 用 RMSNorm 构建模型并跑一次前向
        model_rms = TransformerModel(vocab_size=1000, d_model=128, nhead=8,
                                     num_encoder_layers=2, num_decoder_layers=2,
                                     norm_type='rmsnorm')
        src_rms = torch.randint(0, 1000, (1, 20))
        tgt_rms = torch.randint(0, 1000, (1, 15))
        logits_rms, _, _ = model_rms(src_rms, tgt_rms)
        print(f"  RMSNorm 模型前向输出 shape: {tuple(logits_rms.shape)}")
        print(f"  [OK] RMSNorm 模型测试通过")
        
        # ==================== RoPE 验证 ====================
        print("\n" + "="*60)
        print("RoPE 验证 (相对位置性质)")
        print("="*60)
        
        from scripts.core.attention import RoPEPositionalEncoding, apply_rotary_pos_emb
        
        rope = RoPEPositionalEncoding(head_dim=16, max_len=64)
        
        # 验证1：RoPE 不改变向量模长（旋转是正交变换）
        x = torch.randn(1, 1, 8, 16)
        cos, sin = rope(8)
        x_rot, _ = apply_rotary_pos_emb(x, x, cos, sin)
        norm_before = x.norm(dim=-1).mean().item()
        norm_after = x_rot.norm(dim=-1).mean().item()
        print(f"  旋转前模长: {norm_before:.4f}, 旋转后模长: {norm_after:.4f} (应相等)")
        assert abs(norm_before - norm_after) < 1e-4, "RoPE应保持模长不变"
        
        # 验证2：相对位置不变性 —— 平移序列后两两内积不变
        # 构造两个向量分别位于位置 m, n，旋转后内积只依赖 (n-m)
        from scripts.core.attention import _rotate_half
        
        def rot_at(v: torch.Tensor, pos: int) -> torch.Tensor:
            """把单个向量旋转到绝对位置 pos（v: [1,1,1,head_dim]）"""
            cos_p = rope.cos_cached[pos:pos + 1].repeat(1, 2)  # [1, d_k]（复制拼接成完整表）
            sin_p = rope.sin_cached[pos:pos + 1].repeat(1, 2)
            v4 = v.view(1, 1, 1, -1)
            return v4 * cos_p.view(1, 1, 1, -1) + _rotate_half(v4) * sin_p.view(1, 1, 1, -1)
        
        q1 = torch.randn(1, 1, 1, 16)
        k1 = torch.randn(1, 1, 1, 16)
        def pair_dot(pos_q: int, pos_k: int) -> float:
            return (rot_at(q1, pos_q) * rot_at(k1, pos_k)).sum().item()
        
        # 位置 (2,5) 与平移后的 (6,9)：相对距离都是 3，内积应相等
        dot_a = pair_dot(2, 5)
        dot_b = pair_dot(6, 9)
        print(f"  内积@[m=2,n=5]: {dot_a:.6f}")
        print(f"  内积@[m=6,n=9]: {dot_b:.6f} (相对距离均为3，应相等)")
        assert abs(dot_a - dot_b) < 1e-4, "RoPE内积应只依赖相对位置"
        print(f"  [OK] RoPE 相对位置性质验证通过")
        
        # 验证3：RoPE 模式模型端到端 + KV Cache 推理
        model_rope = TransformerModel(vocab_size=1000, d_model=128, nhead=8,
                                      num_encoder_layers=2, num_decoder_layers=2,
                                      position_embedding_type='rope')
        src_rope = torch.randint(0, 1000, (1, 20))
        tgt_rope = torch.randint(0, 1000, (1, 15))
        logits_rope, _, _ = model_rope(src_rope, tgt_rope)
        print(f"  RoPE 模型训练前向输出 shape: {tuple(logits_rope.shape)}")
        
        model_rope.clear_cache()
        from scripts.optimization.kv_cache import KVCacheManager as KVMgr
        kv_rope = KVMgr(num_layers=2, batch_size=1, num_heads=8,
                        max_seq_len=64, head_dim=16, device=torch.device('cpu'))
        kv_rope.initialize()
        gen = torch.tensor([[2]])  # BOS
        for _ in range(5):
            logits_rope, kv_rope = model_rope.generate_step(src_rope, gen, kv_cache_manager=kv_rope)
            gen = torch.cat([gen, torch.argmax(logits_rope, dim=-1, keepdim=True)], dim=1)
        print(f"  RoPE 模型 KV Cache 推理 5 步成功, 缓存长度: {kv_rope.get_current_length()}")
        print(f"  [OK] RoPE 模型测试通过")
