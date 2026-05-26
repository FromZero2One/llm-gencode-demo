"""
Embedding工作原理详解 - 从ID到向量的映射过程

这个脚本演示了Embedding如何将离散的token ID转换为连续的向量表示。
"""

import torch
import torch.nn as nn
import numpy as np


def demo_embedding_basic():
    """基础演示: Embedding的查找表机制"""
    
    print("=" * 70)
    print("1. Embedding基础: 查找表机制")
    print("=" * 70)
    
    # 创建Embedding层
    vocab_size = 1000  # 词汇表大小
    d_model = 128      # 向量维度
    
    embedding_layer = nn.Embedding(vocab_size, d_model)
    
    print(f"\nEmbedding矩阵形状: {embedding_layer.weight.shape}")
    print(f"  - 行数 (vocab_size): {vocab_size} ← 每个token一行")
    print(f"  - 列数 (d_model): {d_model} ← 每个token的向量维度")
    print(f"  - 总参数数量: {vocab_size * d_model:,}")
    
    # 查看ID 15对应的向量
    token_id = 15
    vector = embedding_layer.weight[token_id]
    
    print(f"\nToken ID {token_id} 对应的向量:")
    print(f"  形状: {vector.shape}")
    print(f"  前10个值: {vector[:10].detach().numpy()}")
    print(f"  这是随机初始化的值，训练后会学习到语义!")
    

def demo_embedding_lookup():
    """演示: 如何使用Embedding进行查找"""
    
    print("\n" + "=" * 70)
    print("2. Embedding查找: 从ID到向量")
    print("=" * 70)
    
    vocab_size = 1000
    d_model = 128
    
    embedding_layer = nn.Embedding(vocab_size, d_model)
    
    # 模拟一批token IDs
    batch_token_ids = torch.tensor([
        [2, 15, 23, 89, 1],      # 样本1: [BOS, "public", "class", "User", EOS]
        [2, 45, 67, 1, 0],       # 样本2: [BOS, "return", "null", EOS, PAD]
    ])
    
    print(f"\n输入token IDs:")
    print(f"  形状: {batch_token_ids.shape}")
    print(f"  内容:\n{batch_token_ids}")
    
    # 使用Embedding进行查找
    embeddings = embedding_layer(batch_token_ids)
    
    print(f"\n输出embeddings:")
    print(f"  形状: {embeddings.shape}")
    print(f"  解释: [batch_size=2, seq_len=5, d_model=128]")
    print(f"  含义: 2个样本，每个样本5个tokens，每个token是128维向量")
    
    # 查看第一个样本的第一个token (BOS)
    bos_vector = embeddings[0, 0, :]
    print(f"\n第一个样本的第一个token (BOS, ID=2):")
    print(f"  向量形状: {bos_vector.shape}")
    print(f"  前10个值: {bos_vector[:10].detach().numpy()}")
    
    # 验证: 直接索引和Embedding查找结果相同
    direct_lookup = embedding_layer.weight[2]
    print(f"\n验证: 直接查找embedding_matrix[2]")
    print(f"  两种方法结果是否相同: {torch.allclose(bos_vector, direct_lookup)}")


def demo_embedding_training():
    """演示: Embedding在训练中如何学习"""
    
    print("\n" + "=" * 70)
    print("3. Embedding训练: 从随机到语义")
    print("=" * 70)
    
    vocab_size = 100
    d_model = 16  # 用小维度方便展示
    
    embedding_layer = nn.Embedding(vocab_size, d_model)
    
    # 训练前的向量(随机初始化)
    print("\n训练前 (随机初始化):")
    public_id = 15
    class_id = 23
    
    print(f"  ID {public_id} ('public'): {embedding_layer.weight[public_id][:5].detach().numpy()}")
    print(f"  ID {class_id} ('class'):  {embedding_layer.weight[class_id][:5].detach().numpy()}")
    
    # 计算相似度(余弦相似度)
    def cosine_similarity(v1, v2):
        return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    
    similarity_before = cosine_similarity(
        embedding_layer.weight[public_id],
        embedding_layer.weight[class_id]
    )
    print(f"  两者相似度: {similarity_before.item():.4f} (随机值)")
    
    # 模拟训练过程(简化版)
    print("\n模拟训练过程...")
    
    # 创建一个简单的任务: 预测下一个token
    optimizer = torch.optim.SGD(embedding_layer.parameters(), lr=0.1)
    
    # 模拟几个训练步骤
    for step in range(5):
        optimizer.zero_grad()
        
        # 假设"public"后面经常跟着"class"
        input_ids = torch.tensor([public_id])
        target_ids = torch.tensor([class_id])
        
        # 获取embeddings
        input_emb = embedding_layer(input_ids)
        
        # 简单损失函数: 让input_emb接近target对应的embedding
        target_emb = embedding_layer.weight[target_ids].detach()
        loss = torch.mean((input_emb - target_emb) ** 2)
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 2 == 0:
            sim = cosine_similarity(
                embedding_layer.weight[public_id],
                embedding_layer.weight[class_id]
            )
            print(f"  Step {step+1}: Loss={loss.item():.4f}, 相似度={sim.item():.4f}")
    
    # 训练后的向量
    print("\n训练后 (学习到关系):")
    print(f"  ID {public_id} ('public'): {embedding_layer.weight[public_id][:5].detach().numpy()}")
    print(f"  ID {class_id} ('class'):  {embedding_layer.weight[class_id][:5].detach().numpy()}")
    
    similarity_after = cosine_similarity(
        embedding_layer.weight[public_id],
        embedding_layer.weight[class_id]
    )
    print(f"  两者相似度: {similarity_after.item():.4f} (应该更高)")
    print(f"  变化: {similarity_after.item() - similarity_before.item():.4f}")


def demo_embedding_visualization():
    """演示: Embedding可视化"""
    
    print("\n" + "=" * 70)
    print("4. Embedding可视化: 理解向量空间")
    print("=" * 70)
    
    vocab_size = 20
    d_model = 4  # 用4维方便可视化
    
    embedding_layer = nn.Embedding(vocab_size, d_model)
    
    # 定义一些token
    tokens = {
        0: "<PAD>",
        1: "<EOS>",
        2: "<BOS>",
        3: "public",
        4: "private",
        5: "class",
        6: "interface",
        7: "extends",
        8: "implements",
    }
    
    print("\nEmbedding矩阵 (部分):")
    print("-" * 70)
    print(f"{'ID':<4} {'Token':<12} {'Dim 0':<8} {'Dim 1':<8} {'Dim 2':<8} {'Dim 3':<8}")
    print("-" * 70)
    
    for token_id, token_name in tokens.items():
        vector = embedding_layer.weight[token_id].detach().numpy()
        print(f"{token_id:<4} {token_name:<12} {vector[0]:<8.3f} {vector[1]:<8.3f} "
              f"{vector[2]:<8.3f} {vector[3]:<8.3f}")
    
    print("\n观察:")
    print("  - 每一行是一个token的向量表示")
    print("  - 训练前这些值是随机的")
    print("  - 训练后，相似的词会有相似的向量")
    print("  - 例如: 'public'和'private'都是访问修饰符，它们的向量会相似")


def demo_real_world_example():
    """真实场景示例: 完整流程"""
    
    print("\n" + "=" * 70)
    print("5. 真实场景: 从文本到向量")
    print("=" * 70)
    
    # 模拟tokenizer的输出
    text = "public class User"
    token_ids = [2, 15, 23, 89, 1]  # [BOS, "public", "class", "User", EOS]
    token_names = ["<BOS>", "public", "class", "User", "<EOS>"]
    
    print(f"\n原始文本: '{text}'")
    print(f"Token IDs: {token_ids}")
    print(f"Token名称: {token_names}")
    
    # 创建Embedding层
    vocab_size = 1000
    d_model = 128
    
    embedding_layer = nn.Embedding(vocab_size, d_model)
    
    # 转换为tensor并获取embeddings
    ids_tensor = torch.tensor([token_ids])  # [1, 5]
    embeddings = embedding_layer(ids_tensor)  # [1, 5, 128]
    
    print(f"\nEmbeddings形状: {embeddings.shape}")
    print(f"  解释: 1个样本, 5个tokens, 每个token是128维向量")
    
    # 展示每个token的向量统计信息
    print(f"\n每个token的向量统计:")
    print("-" * 70)
    print(f"{'Token':<10} {'ID':<4} {'均值':<10} {'标准差':<10} {'最小值':<10} {'最大值':<10}")
    print("-" * 70)
    
    for i, (name, token_id) in enumerate(zip(token_names, token_ids)):
        vector = embeddings[0, i, :].detach().numpy()
        print(f"{name:<10} {token_id:<4} {vector.mean():<10.4f} {vector.std():<10.4f} "
              f"{vector.min():<10.4f} {vector.max():<10.4f}")


def demo_mathematical_interpretation():
    """数学解释: Embedding的本质"""
    
    print("\n" + "=" * 70)
    print("6. 数学本质: Embedding是什么?")
    print("=" * 70)
    
    print("\nEmbedding本质上是一个矩阵乘法:")
    print("  E = W @ one_hot(x)")
    print("\n其中:")
    print("  - W: Embedding矩阵, 形状 [vocab_size, d_model]")
    print("  - x: token ID (标量)")
    print("  - one_hot(x): one-hot编码, 形状 [vocab_size]")
    print("  - E: 输出向量, 形状 [d_model]")
    
    # 实际演示
    vocab_size = 10
    d_model = 4
    
    embedding_weight = nn.Embedding(vocab_size, d_model).weight.detach()
    
    token_id = 3
    
    # 方法1: 直接索引(高效)
    result1 = embedding_weight[token_id]
    
    # 方法2: one-hot编码 + 矩阵乘法(等价但低效)
    one_hot = torch.zeros(vocab_size)
    one_hot[token_id] = 1.0
    result2 = torch.matmul(one_hot.unsqueeze(0), embedding_weight).squeeze(0)
    
    print(f"\n示例: token_id = {token_id}")
    print(f"  One-hot编码: {one_hot.numpy()}")
    print(f"  方法1 (索引): {result1.numpy()}")
    print(f"  方法2 (矩阵乘法): {result2.numpy()}")
    print(f"  结果相同: {torch.allclose(result1, result2)}")
    
    print("\n为什么用索引而不是矩阵乘法?")
    print("  ✅ 索引: O(1) 时间复杂度, 只需一次内存访问")
    print("  ❌ 矩阵乘法: O(vocab_size × d_model), 浪费计算")
    print("  → PyTorch的nn.Embedding内部就是使用索引实现!")


if __name__ == "__main__":
    print("\n" + "🎓" * 35)
    print("Embedding工作原理详解")
    print("🎓" * 35 + "\n")
    
    # 运行所有演示
    demo_embedding_basic()
    demo_embedding_lookup()
    demo_embedding_training()
    demo_embedding_visualization()
    demo_real_world_example()
    demo_mathematical_interpretation()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
1. Embedding是一个可学习的查找表 (矩阵)
   - 形状: [vocab_size, d_model]
   - 每一行对应一个token的向量表示

2. 工作流程:
   Token ID → 查找Embedding矩阵 → 得到向量
   
3. 关键特点:
   - 初始化时是随机值
   - 通过训练学习到语义
   - 相似的词有相似的向量
   - 支持数学运算 (如: King - Man + Woman ≈ Queen)

4. 实现方式:
   - PyTorch: nn.Embedding(vocab_size, d_model)
   - 本质: 高效的索引操作 (不是矩阵乘法)
   - 用法: embedding_layer(token_ids)

5. 在Transformer中的作用:
   - 将离散符号转换为连续向量
   - 使神经网络能够处理文本
   - 捕捉语义关系
""")
