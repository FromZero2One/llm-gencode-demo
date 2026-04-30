"""
Transformer模型核心架构
实现完整的Encoder-Decoder结构用于代码生成
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List
from attention import MultiHeadAttention, PositionalEncoding


class AttentionTracker:
    """
    注意力权重追踪器
    
    记录每一层Encoder和Decoder的注意力模式，帮助学习者理解：
    - 信息如何在不同层之间流动
    - 深层网络如何逐步抽象特征
    - 不同层的注意力模式差异
    """
    
    def __init__(self):
        self.encoder_attn_by_layer: Dict[int, torch.Tensor] = {}
        self.decoder_self_attn_by_layer: Dict[int, torch.Tensor] = {}
        self.decoder_cross_attn_by_layer: Dict[int, torch.Tensor] = {}
        self.current_step = 0
    
    def reset(self):
        """重置追踪器"""
        self.encoder_attn_by_layer.clear()
        self.decoder_self_attn_by_layer.clear()
        self.decoder_cross_attn_by_layer.clear()
        self.current_step = 0
    
    def record_encoder_attention(self, layer_idx: int, attn_weights: torch.Tensor):
        """记录Encoder某一层的注意力权重"""
        self.encoder_attn_by_layer[layer_idx] = attn_weights.detach().cpu()
    
    def record_decoder_attention(self, layer_idx: int, self_attn: torch.Tensor, 
                                 cross_attn: torch.Tensor):
        """记录Decoder某一层的自注意力和交叉注意力权重"""
        self.decoder_self_attn_by_layer[layer_idx] = self_attn.detach().cpu()
        self.decoder_cross_attn_by_layer[layer_idx] = cross_attn.detach().cpu()
    
    def get_layer_summary(self, layer_idx: int) -> Dict:
        """获取某一层的注意力统计摘要"""
        summary = {}
        
        if layer_idx in self.encoder_attn_by_layer:
            enc_attn = self.encoder_attn_by_layer[layer_idx]
            summary['encoder'] = {
                'shape': list(enc_attn.shape),
                'mean_attention': enc_attn.mean().item(),
                'max_attention': enc_attn.max().item(),
                'attention_entropy': self._calculate_attention_entropy(enc_attn).item()
            }
        
        if layer_idx in self.decoder_self_attn_by_layer:
            dec_self = self.decoder_self_attn_by_layer[layer_idx]
            summary['decoder_self'] = {
                'shape': list(dec_self.shape),
                'mean_attention': dec_self.mean().item(),
                'max_attention': dec_self.max().item(),
                'attention_entropy': self._calculate_attention_entropy(dec_self).item()
            }
        
        if layer_idx in self.decoder_cross_attn_by_layer:
            dec_cross = self.decoder_cross_attn_by_layer[layer_idx]
            summary['decoder_cross'] = {
                'shape': list(dec_cross.shape),
                'mean_attention': dec_cross.mean().item(),
                'max_attention': dec_cross.max().item(),
                'attention_entropy': self._calculate_attention_entropy(dec_cross).item()
            }
        
        return summary
    
    def _calculate_attention_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """计算注意力分布的熵（衡量注意力的集中度）"""
        # attn_weights: [batch, nhead, seq_len, seq_len]
        # 对最后一个维度计算熵
        eps = 1e-10
        entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
        return entropy.mean()
    
    def print_layer_comparison(self, num_encoder_layers: int, num_decoder_layers: int):
        """打印各层注意力模式的对比"""
        print(f"\n{'='*70}")
        print(f"[AttentionTracker] 逐层注意力模式对比")
        print(f"{'='*70}")
        
        print(f"\n--- Encoder Layers ---")
        for i in range(num_encoder_layers):
            if i in self.encoder_attn_by_layer:
                summary = self.get_layer_summary(i)['encoder']
                print(f"\nLayer {i+1}:")
                print(f"  Shape: {summary['shape']}")
                print(f"  Mean attention: {summary['mean_attention']:.6f}")
                print(f"  Max attention: {summary['max_attention']:.4f}")
                print(f"  Attention entropy: {summary['attention_entropy']:.4f}")
                print(f"  {'(More focused)' if summary['attention_entropy'] < 2.0 else '(More distributed)'}")
        
        print(f"\n--- Decoder Layers ---")
        for i in range(num_decoder_layers):
            if i in self.decoder_self_attn_by_layer:
                self_summary = self.get_layer_summary(i)['decoder_self']
                cross_summary = self.get_layer_summary(i)['decoder_cross']
                
                print(f"\nLayer {i+1}:")
                print(f"  Self-Attention:")
                print(f"    Entropy: {self_summary['attention_entropy']:.4f} "
                      f"{'(focused)' if self_summary['attention_entropy'] < 2.0 else '(distributed)'}")
                print(f"  Cross-Attention:")
                print(f"    Entropy: {cross_summary['attention_entropy']:.4f} "
                      f"{'(focused)' if cross_summary['attention_entropy'] < 2.0 else '(distributed)'}")
    
    def visualize_attention_evolution(self, token_names: List[str], 
                                     save_path: str = 'attention_evolution.png'):
        """
        可视化注意力权重随层数的演变
        
        Args:
            token_names: token名称列表
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            num_encoder = len(self.encoder_attn_by_layer)
            num_decoder = len(self.decoder_cross_attn_by_layer)
            
            if num_encoder == 0 and num_decoder == 0:
                print("[AttentionTracker] 没有可可视化的注意力数据")
                return
            
            total_layers = num_encoder + num_decoder
            fig, axes = plt.subplots(1, total_layers, figsize=(5 * total_layers, 4))
            
            if total_layers == 1:
                axes = [axes]
            
            layer_idx = 0
            
            # 绘制Encoder各层
            for i in range(num_encoder):
                if i in self.encoder_attn_by_layer:
                    attn = self.encoder_attn_by_layer[i][0, 0].numpy()  # 第一个样本，第一个头
                    
                    ax = axes[layer_idx]
                    sns.heatmap(attn, ax=ax, cmap='viridis', cbar=False)
                    ax.set_title(f'Enc Layer {i+1}', fontsize=10)
                    
                    if len(token_names) <= 20:
                        ax.set_xticks(range(len(token_names)))
                        ax.set_yticks(range(len(token_names)))
                        ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=6)
                        ax.set_yticklabels(token_names, fontsize=6)
                    
                    layer_idx += 1
            
            # 绘制Decoder各层（交叉注意力）
            for i in range(num_decoder):
                if i in self.decoder_cross_attn_by_layer:
                    attn = self.decoder_cross_attn_by_layer[i][0, 0].numpy()
                    
                    ax = axes[layer_idx]
                    sns.heatmap(attn, ax=ax, cmap='viridis', cbar=False)
                    ax.set_title(f'Dec Layer {i+1} (Cross)', fontsize=10)
                    
                    if len(token_names) <= 20:
                        ax.set_xticks(range(attn.shape[1]))
                        ax.set_yticks(range(attn.shape[0]))
                        ax.set_xticklabels([f'Src{j}' for j in range(attn.shape[1])], 
                                          rotation=45, ha='right', fontsize=6)
                        ax.set_yticklabels([f'Tgt{j}' for j in range(attn.shape[0])], fontsize=6)
                    
                    layer_idx += 1
            
            plt.suptitle('Attention Evolution Across Layers', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[AttentionTracker] 注意力演变图保存到: {save_path}")
            plt.show()
            
        except ImportError:
            print("[AttentionTracker] 安装matplotlib和seaborn以启用可视化功能")


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
            self_attn_weights: 自注意力权重
            cross_attn_weights: 交叉注意力权重
        """
        print(f"\n[DecoderLayer] 前向传播")
        print(f"  - Target shape: {tgt.shape}")
        print(f"  - Memory shape: {memory.shape}")
        
        # 1. Masked Self-Attention
        self_attn_output, self_attn_weights = self.self_attn(tgt, tgt, tgt, tgt_mask)
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
        
        return tgt, self_attn_weights, cross_attn_weights


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
                 dropout: float = 0.1, max_seq_length: int = 128,
                 enable_attention_tracking: bool = False):
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
        print(f"  - 注意力追踪: {'启用' if enable_attention_tracking else '禁用'}")
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.enable_attention_tracking = enable_attention_tracking
        
        # 注意力追踪器
        if enable_attention_tracking:
            self.attention_tracker = AttentionTracker()
        else:
            self.attention_tracker = None
        
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
            
            # 记录注意力权重（如果启用追踪）
            if self.enable_attention_tracking and self.attention_tracker is not None:
                self.attention_tracker.record_encoder_attention(i, attn_weights)
        
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
        decoder_self_attn_weights = []
        
        for i, layer in enumerate(self.decoder_layers):
            print(f"\n  --- Decoder Layer {i+1} ---")
            output, self_attn_w, cross_attn_w = layer(output, memory, tgt_mask, memory_mask)
            decoder_attn_weights.append(cross_attn_w)
            decoder_self_attn_weights.append(self_attn_w)
            
            # 记录注意力权重（如果启用追踪）
            if self.enable_attention_tracking and self.attention_tracker is not None:
                self.attention_tracker.record_decoder_attention(i, self_attn_w, cross_attn_w)
        
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
        
        # 重置注意力追踪器
        if self.enable_attention_tracking and self.attention_tracker is not None:
            self.attention_tracker.reset()
    
    def get_attention_tracker(self) -> AttentionTracker:
        """获取注意力追踪器"""
        return self.attention_tracker


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
