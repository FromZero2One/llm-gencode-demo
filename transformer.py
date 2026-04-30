"""
Transformer模型核心架构
实现完整的Encoder-Decoder结构用于代码生成
"""

import torch
import torch.nn as nn
import math
from attention import MultiHeadAttention, PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder层
    
    包含：
    1. Multi-Head Self-Attention
    2. Feed Forward Network
    3. Residual Connection + Layer Normalization
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8, 
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        print(f"[EncoderLayer] 初始化")
        print(f"  - d_model: {d_model}")
        print(f"  - nhead: {nhead}")
        print(f"  - dim_feedforward: {dim_feedforward}")
        
        # Self-Attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
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
        Args:
            src: [batch_size, seq_len, d_model]
            src_mask: 可选的掩码
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = src.shape
        
        print(f"\n[EncoderLayer] 前向传播")
        print(f"  - Input shape: {src.shape}")
        
        # 1. Self-Attention + Residual + LayerNorm
        attn_output, attn_weights = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(attn_output)  # Residual connection
        src = self.norm1(src)  # Layer normalization
        
        print(f"  - After Attention + Residual + Norm: {src.shape}")
        
        # 2. Feed Forward + Residual + LayerNorm
        ffn_output = self.ffn(src)
        src = src + ffn_output  # Residual connection
        src = self.norm2(src)  # Layer normalization
        
        print(f"  - After FFN + Residual + Norm: {src.shape}")
        
        return src, attn_weights


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder层
    
    包含：
    1. Masked Multi-Head Self-Attention（防止看到未来token）
    2. Cross-Attention（关注Encoder输出）
    3. Feed Forward Network
    4. Residual Connection + Layer Normalization
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        print(f"[DecoderLayer] 初始化")
        
        # Masked Self-Attention（解码时只能看到之前的token）
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Cross-Attention（关注Encoder的输出）
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        
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
        Args:
            tgt: Decoder输入 [batch_size, tgt_seq_len, d_model]
            memory: Encoder输出 [batch_size, src_seq_len, d_model]
            tgt_mask: Decoder自注意力掩码
            memory_mask: Cross-attention掩码
            
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
        """
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
        
        print(f"  - Output shape: {tgt.shape}")
        
        return tgt, cross_attn_weights


class TransformerModel(nn.Module):
    """
    完整的Transformer模型
    
    架构：
    1. Embedding层
    2. 位置编码
    3. N层Encoder
    4. N层Decoder
    5. 输出投影层
    """
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, 
                 nhead: int = 8, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_length: int = 128):
        super().__init__()
        
        print(f"\n{'='*60}")
        print(f"[TransformerModel] 初始化完整模型")
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
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出投影层：将隐藏状态映射到词汇表
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._init_weights()
        
        print(f"{'='*60}\n")
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成因果掩码（用于Decoder自注意力）
        确保每个位置只能看到之前的位置
        
        Returns:
            mask: [sz, sz] 上三角矩阵为0，下三角为1
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        # 将True的位置设为0（屏蔽），False的位置设为1
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Encoder过程
        
        Args:
            src: [batch_size, src_seq_len] token IDs
            src_mask: [batch_size, 1, 1, src_seq_len] 注意力掩码
            
        Returns:
            memory: [batch_size, src_seq_len, d_model]
        """
        print(f"\n[Transformer] 开始Encoder过程")
        print(f"  - Source shape: {src.shape}")
        
        # 1. Embedding
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        print(f"  - After embedding: {src_embedded.shape}")
        
        # 2. 位置编码
        src_encoded = self.pos_encoder(src_embedded)
        print(f"  - After positional encoding: {src_encoded.shape}")
        
        # 3. 通过多层Encoder
        memory = src_encoded
        encoder_attn_weights = []
        
        for i, layer in enumerate(self.encoder_layers):
            print(f"\n  --- Encoder Layer {i+1} ---")
            memory, attn_weights = layer(memory, src_mask)
            encoder_attn_weights.append(attn_weights)
        
        print(f"\n[Transformer] Encoder完成")
        print(f"  - Memory shape: {memory.shape}")
        
        return memory, encoder_attn_weights
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        """
        Decoder过程
        
        Args:
            tgt: [batch_size, tgt_seq_len] token IDs
            memory: Encoder输出 [batch_size, src_seq_len, d_model]
            tgt_mask: Decoder自注意力掩码
            memory_mask: Cross-attention掩码
            
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
        """
        print(f"\n[Transformer] 开始Decoder过程")
        print(f"  - Target shape: {tgt.shape}")
        print(f"  - Memory shape: {memory.shape}")
        
        # 1. Embedding
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        print(f"  - After embedding: {tgt_embedded.shape}")
        
        # 2. 位置编码
        tgt_encoded = self.pos_encoder(tgt_embedded)
        print(f"  - After positional encoding: {tgt_encoded.shape}")
        
        # 3. 通过多层Decoder
        output = tgt_encoded
        decoder_attn_weights = []
        
        for i, layer in enumerate(self.decoder_layers):
            print(f"\n  --- Decoder Layer {i+1} ---")
            output, attn_weights = layer(output, memory, tgt_mask, memory_mask)
            decoder_attn_weights.append(attn_weights)
        
        print(f"\n[Transformer] Decoder完成")
        print(f"  - Output shape: {output.shape}")
        
        return output, decoder_attn_weights
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None):
        """
        完整的前向传播（训练时使用）
        
        Args:
            src: [batch_size, src_seq_len] 源序列（prompt）
            tgt: [batch_size, tgt_seq_len] 目标序列（期望输出）
            
        Returns:
            logits: [batch_size, tgt_seq_len, vocab_size]
        """
        # Encode
        memory, encoder_weights = self.encode(src, src_mask)
        
        # Decode
        output, decoder_weights = self.decode(tgt, memory, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        print(f"\n[Transformer] 最终输出")
        print(f"  - Logits shape: {logits.shape}")
        
        return logits, encoder_weights, decoder_weights
    
    def generate_step(self, src: torch.Tensor, generated: torch.Tensor,
                      src_mask: torch.Tensor = None):
        """
        单步生成（推理时使用）
        
        Args:
            src: [batch_size, src_seq_len] 源序列
            generated: [batch_size, current_seq_len] 已生成的序列
            
        Returns:
            next_token_logits: [batch_size, vocab_size] 下一个token的概率分布
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
        """清除缓存（用于新的生成任务）"""
        if hasattr(self, '_cached_memory'):
            delattr(self, '_cached_memory')


# 测试代码
if __name__ == '__main__':
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
