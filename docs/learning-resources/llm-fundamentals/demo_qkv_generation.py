"""
Q、K、V向量生成详解 - 从Embedding到注意力向量的完整流程

这个脚本演示了每个token的Q、K、V向量是如何通过线性变换生成的。
"""

import torch
import torch.nn as nn
import numpy as np


def demo_qkv_generation():
    """演示Q、K、V向量的生成过程"""
    
    print("=" * 70)
    print("1. Q、K、V向量的来源")
    print("=" * 70)
    
    # ==================== 配置参数 ====================
    d_model = 128      # 模型维度
    seq_len = 5        # 序列长度
    batch_size = 1     # 批量大小
    
    print(f"\n配置:")
    print(f"  - d_model (模型维度): {d_model}")
    print(f"  - seq_len (序列长度): {seq_len}")
    print(f"  - batch_size (批量): {batch_size}")
    
    # ==================== Step 1: Token IDs ====================
    print("\n" + "=" * 70)
    print("Step 1: Token IDs (原始输入)")
    print("=" * 70)
    
    token_ids = torch.tensor([[2, 15, 23, 89, 1]])  # [BOS, public, class, User, EOS]
    print(f"\nToken IDs: {token_ids}")
    print(f"Shape: {token_ids.shape}")
    print(f"\n对应关系:")
    print(f"  ID 2  → BOS")
    print(f"  ID 15 → 'public'")
    print(f"  ID 23 → 'class'")
    print(f"  ID 89 → 'User'")
    print(f"  ID 1  → EOS")
    
    # ==================== Step 2: Embedding Lookup ====================
    print("\n" + "=" * 70)
    print("Step 2: Embedding Lookup (查找词嵌入)")
    print("=" * 70)
    
    vocab_size = 1000
    embedding_layer = nn.Embedding(vocab_size, d_model)
    embeddings = embedding_layer(token_ids)
    
    print(f"\nEmbedding层形状: [{vocab_size}, {d_model}]")
    print(f"输出形状: {embeddings.shape}")
    print(f"\n每个token都有一个{d_model}维的向量表示")
    print(f"\n示例: 'public' (ID=15) 的embedding前10维:")
    print(f"  {embeddings[0, 1, :10].detach().numpy()}")
    
    # ==================== Step 3: 创建三个线性层 ====================
    print("\n" + "=" * 70)
    print("Step 3: 创建三个线性变换层 (W_q, W_k, W_v)")
    print("=" * 70)
    
    W_q = nn.Linear(d_model, d_model)  # Query投影
    W_k = nn.Linear(d_model, d_model)  # Key投影
    W_v = nn.Linear(d_model, d_model)  # Value投影
    
    print(f"\n三个线性层的权重矩阵形状:")
    print(f"  W_q.weight.shape: {W_q.weight.shape}")  # [128, 128]
    print(f"  W_k.weight.shape: {W_k.weight.shape}")  # [128, 128]
    print(f"  W_v.weight.shape: {W_v.weight.shape}")  # [128, 128]
    
    print(f"\n偏置项形状:")
    print(f"  W_q.bias.shape: {W_q.bias.shape}")      # [128]
    print(f"  W_k.bias.shape: {W_k.bias.shape}")      # [128]
    print(f"  W_v.bias.shape: {W_v.bias.shape}")      # [128]
    
    print(f"\n【重要】这三个线性层是独立的可学习参数!")
    print(f"  - 训练开始时: 随机初始化")
    print(f"  - 训练过程中: 通过反向传播自动更新")
    print(f"  - 训练结束后: 学习到不同的投影模式")
    
    # ==================== Step 4: 生成Q、K、V ====================
    print("\n" + "=" * 70)
    print("Step 4: 通过线性变换生成Q、K、V")
    print("=" * 70)
    
    Q = W_q(embeddings)  # Query
    K = W_k(embeddings)  # Key
    V = W_v(embeddings)  # Value
    
    print(f"\n输入embeddings.shape: {embeddings.shape}")
    print(f"\n经过线性变换后:")
    print(f"  Q.shape: {Q.shape}")
    print(f"  K.shape: {K.shape}")
    print(f"  V.shape: {V.shape}")
    
    # ==================== Step 5: 单个token的详细计算 ====================
    print("\n" + "=" * 70)
    print("Step 5: 单个token的详细计算过程")
    print("=" * 70)
    
    # 选择"public"这个token (位置1)
    token_idx = 1
    token_name = "public"
    
    x_token = embeddings[0, token_idx, :]  # [128]
    
    print(f"\n以 '{token_name}' (位置{token_idx}) 为例:")
    print(f"\n原始embedding (前5维): {x_token[:5].detach().numpy()}")
    
    # 手动计算Q、K、V
    q_manual = W_q(x_token)
    k_manual = W_k(x_token)
    v_manual = W_v(x_token)
    
    print(f"\nQuery向量 (前5维): {q_manual[:5].detach().numpy()}")
    print(f"Key向量   (前5维): {k_manual[:5].detach().numpy()}")
    print(f"Value向量 (前5维): {v_manual[:5].detach().numpy()}")
    
    print(f"\n数学公式:")
    print(f"  Q = W_q @ x + b_q")
    print(f"  K = W_k @ x + b_k")
    print(f"  V = W_v @ x + b_v")
    
    print(f"\n其中:")
    print(f"  x: token的embedding [{d_model}]")
    print(f"  W_q, W_k, W_v: 权重矩阵 [{d_model}×{d_model}]")
    print(f"  b_q, b_k, b_v: 偏置向量 [{d_model}]")
    
    # 验证手动计算与批量计算一致
    assert torch.allclose(q_manual, Q[0, token_idx, :], atol=1e-5)
    assert torch.allclose(k_manual, K[0, token_idx, :], atol=1e-5)
    assert torch.allclose(v_manual, V[0, token_idx, :], atol=1e-5)
    print(f"\n✅ 验证: 手动计算与批量计算结果一致!")
    
    # ==================== Step 6: 可视化权重矩阵 ====================
    print("\n" + "=" * 70)
    print("Step 6: 权重矩阵的可视化")
    print("=" * 70)
    
    print(f"\nW_q权重矩阵 (部分):")
    print(f"  形状: {W_q.weight.shape}")
    print(f"  第1行前10个值: {W_q.weight[0, :10].detach().numpy()}")
    print(f"  第2行前10个值: {W_q.weight[1, :10].detach().numpy()}")
    
    print(f"\nW_k权重矩阵 (部分):")
    print(f"  形状: {W_k.weight.shape}")
    print(f"  第1行前10个值: {W_k.weight[0, :10].detach().numpy()}")
    print(f"  第2行前10个值: {W_k.weight[1, :10].detach().numpy()}")
    
    print(f"\nW_v权重矩阵 (部分):")
    print(f"  形状: {W_v.weight.shape}")
    print(f"  第1行前10个值: {W_v.weight[0, :10].detach().numpy()}")
    print(f"  第2行前10个值: {W_v.weight[1, :10].detach().numpy()}")
    
    print(f"\n【观察】三个权重矩阵的值不同，说明它们学习不同的投影!")
    
    # ==================== Step 7: 为什么需要三个不同的向量？ ====================
    print("\n" + "=" * 70)
    print("Step 7: 为什么需要Q、K、V三个不同的向量？")
    print("=" * 70)
    
    print(f"\n类比数据库查询:")
    print(f"  Query (Q): 你的搜索请求 - '我想找什么？'")
    print(f"  Key (K):   数据库索引   - '我有什么特征？'")
    print(f"  Value (V): 实际数据     - '我的内容是什么？'")
    
    print(f"\n在Attention中的作用:")
    print(f"  1. Q和K做点积 → 计算相似度分数")
    print(f"  2. Softmax归一化 → 得到注意力权重")
    print(f"  3. 用权重加权V → 获取相关信息")
    
    print(f"\n为什么要分开？")
    print(f"  ✅ 灵活性: Q关注'我要找什么',K关注'我是什么',V关注'我有什么'")
    print(f"  ✅ 表达能力: 三个独立空间可以学习更复杂的关系")
    print(f"  ✅ 对称性: Q-K的点积天然衡量匹配程度")
    
    # ==================== Step 8: 多头情况下的Q、K、V ====================
    print("\n" + "=" * 70)
    print("Step 8: 多头注意力中的Q、K、V分割")
    print("=" * 70)
    
    nhead = 8
    d_k = d_model // nhead
    
    print(f"\n配置:")
    print(f"  - nhead (头数): {nhead}")
    print(f"  - d_k (每头维度): {d_k}")
    
    # 重塑并分割成多个头
    Q_split = Q.view(batch_size, seq_len, nhead, d_k).transpose(1, 2)
    K_split = K.view(batch_size, seq_len, nhead, d_k).transpose(1, 2)
    V_split = V.view(batch_size, seq_len, nhead, d_k).transpose(1, 2)
    
    print(f"\n分割前形状:")
    print(f"  Q: {Q.shape}  → [batch, seq_len, d_model]")
    
    print(f"\n分割后形状:")
    print(f"  Q_split: {Q_split.shape}  → [batch, nhead, seq_len, d_k]")
    print(f"  K_split: {K_split.shape}  → [batch, nhead, seq_len, d_k]")
    print(f"  V_split: {V_split.shape}  → [batch, nhead, seq_len, d_k]")
    
    print(f"\n【理解】每个头独立处理自己的Q、K、V:")
    for head_idx in range(min(3, nhead)):
        print(f"  Head {head_idx}: Q[{head_idx}] shape = {Q_split[0, head_idx].shape}")
    
    print(f"\n每个头学习不同的关系模式:")
    print(f"  Head 0: 可能关注语法关系 (public → class)")
    print(f"  Head 1: 可能关注命名模式 (UserService → Service)")
    print(f"  Head 2: 可能关注继承关系 (extends → BaseService)")
    print(f"  ...")
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("📚 总结: Q、K、V的完整生成流程")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────┐
│  Token IDs                                           │
│  [2, 15, 23, 89, 1]                                 │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Embedding Lookup                                    │
│  X = Embedding(token_ids)                           │
│  Shape: [batch, seq_len, d_model]                   │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┬──────────┐
        ▼                     ▼          ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Linear W_q  │    │  Linear W_k  │    │  Linear W_v  │
│  Q = X@W_q   │    │  K = X@W_k   │    │  V = X@W_v   │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Q Matrix    │    │  K Matrix    │    │  V Matrix    │
│  [b, s, 128] │    │  [b, s, 128] │    │  [b, s, 128] │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Split into Heads                                    │
│  [batch, nhead, seq_len, d_k]                       │
│  [1, 8, 5, 16]                                      │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Attention Computation                               │
│  scores = Q @ K^T / sqrt(d_k)                       │
│  weights = softmax(scores)                          │
│  output = weights @ V                               │
└─────────────────────────────────────────────────────┘
""")
    
    print("\n🎯 关键要点:")
    print("  1. Q、K、V来自同一个embedding，但经过不同的线性变换")
    print("  2. 三个线性层(W_q, W_k, W_v)是可学习的参数")
    print("  3. 训练过程中，它们会自动学习不同的投影模式")
    print("  4. 多头机制让每个头从不同角度理解token关系")


if __name__ == "__main__":
    demo_qkv_generation()
