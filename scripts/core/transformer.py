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
                 debug_mode: bool = False):
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
        
        # Self-Attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, debug_mode=debug_mode)
        
        # Feed Forward Network (两层全连接)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
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
        attn_output, attn_weights = self.self_attn(src, src, src, src_mask)
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
                 debug_mode: bool = False):
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
        
        Note:
            - Decoder比Encoder多一个LayerNorm（共3个），因为有3个子层
            - Cross-Attention的Q来自Decoder，K和V来自Encoder输出
        """
        super().__init__()
        
        self.debug_mode = debug_mode
        
        print(f"[DecoderLayer] 初始化 (debug_mode={debug_mode})")
        
        # Masked Self-Attention（解码时只能看到之前的token）
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, debug_mode=debug_mode)
        
        # Cross-Attention（关注Encoder的输出）
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout, debug_mode=debug_mode)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
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
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.norm1(tgt)
        
        # 2. Cross-Attention（关注Encoder输出）
        cross_attn_output, cross_attn_weights = self.cross_attn(
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
        
        return tgt, cross_attn_weights


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
                 debug_mode: bool = False):
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
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embedding层：将token ID转换为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, debug_mode=debug_mode)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, debug_mode=debug_mode)
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
        tgt_encoded = self.pos_encoder(tgt_embedded)
        if self.debug_mode:
            print(f"  - After positional encoding: {tgt_encoded.shape}")
        
        # 3. 通过多层Decoder
        output = tgt_encoded
        decoder_attn_weights = []
        
        for i, layer in enumerate(self.decoder_layers):
            if self.debug_mode:
                print(f"\n  --- Decoder Layer {i+1} ---")
            output, attn_weights = layer(output, memory, tgt_mask, memory_mask)
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
                      src_mask: torch.Tensor = None):
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
        # Encode（只需要做一次）
        if not hasattr(self, '_cached_memory'):
            self._cached_memory, _ = self.encode(src, src_mask)
        
        memory = self._cached_memory
        
        # 创建因果掩码
        tgt_seq_len = generated.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(generated.device)
        
        # Decode
        output, _ = self.decode(generated, memory, tgt_mask)
        
        # 只取最后一个位置的输出
        last_output = output[:, -1, :]  # [batch_size, d_model]
        
        # 投影到词汇表
        logits = self.output_projection(last_output)  # [batch_size, vocab_size]
        
        return logits
    
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
