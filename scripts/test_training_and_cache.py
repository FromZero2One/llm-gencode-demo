"""
训练系统和KV Cache的使用示例

本脚本演示如何使用新增的训练功能和KV Cache优化。
"""

import torch
import sys
import os

# 添加scripts目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from transformer import Transformer
from trainer import Trainer, TextDataset
from kv_cache import KVCacheManager, demonstrate_kv_cache_benefit


def demo_training():
    """
    演示训练系统的使用
    
    展示如何：
    1. 准备训练数据
    2. 配置训练器
    3. 执行训练
    4. 保存和加载模型
    """
    print("=" * 70)
    print("训练系统演示")
    print("=" * 70)
    
    # 1. 创建简单的训练数据
    print("\n步骤1: 准备训练数据")
    texts = [
        "def hello_world():",
        "    print('Hello, World!')",
        "",
        "def add_numbers(a, b):",
        "    return a + b",
        "",
        "def multiply(x, y):",
        "    result = x * y",
        "    return result",
    ]
    
    print(f"  - 样本数量: {len(texts)}")
    print(f"  - 示例文本: {texts[0]}")
    
    # 注意：这里需要一个tokenizer，我们用一个简化的版本
    class SimpleTokenizer:
        """简化版tokenizer用于演示"""
        def __init__(self):
            self.vocab = {}
            self.inv_vocab = {}
            self.idx = 0
        
        def encode(self, text):
            """将文本编码为token IDs"""
            tokens = []
            for char in text:
                if char not in self.vocab:
                    self.vocab[char] = self.idx
                    self.inv_vocab[self.idx] = char
                    self.idx += 1
                tokens.append(self.vocab[char])
            return tokens
        
        @property
        def vocab_size(self):
            return len(self.vocab)
    
    tokenizer = SimpleTokenizer()
    
    # 编码所有文本以构建词汇表
    for text in texts:
        tokenizer.encode(text)
    
    print(f"  - 词汇表大小: {tokenizer.vocab_size}")
    
    # 2. 创建数据集
    print("\n步骤2: 创建数据集")
    dataset = TextDataset(texts, tokenizer, max_len=64)
    print(f"  - 数据集大小: {len(dataset)}")
    
    # 3. 创建模型
    print("\n步骤3: 创建Transformer模型")
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,  # 小模型用于快速演示
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 模型参数量: {total_params:,}")
    
    # 4. 配置训练器
    print("\n步骤4: 配置训练器")
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        val_dataset=None,  # 演示中不使用验证集
        batch_size=2,
        lr=1e-3,
        epochs=3,  # 少量epoch用于演示
        warmup_ratio=0.2,
        weight_decay=0.01,
        save_dir="demo_checkpoints"
    )
    
    # 5. 开始训练
    print("\n步骤5: 开始训练")
    print("-" * 70)
    history = trainer.train()
    
    # 6. 显示训练结果
    print("\n步骤6: 训练结果")
    print(f"  - 最终训练损失: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"  - 最终验证损失: {history['val_loss'][-1]:.4f}")
    
    print("\n✓ 训练演示完成!")
    print("=" * 70)


def demo_kv_cache():
    """
    演示KV Cache的使用和性能优势
    """
    print("\n\n" + "=" * 70)
    print("KV Cache演示")
    print("=" * 70)
    
    # 1. 展示性能对比
    print("\n部分1: 性能优势分析")
    print("-" * 70)
    demonstrate_kv_cache_benefit()
    
    # 2. 实际使用示例
    print("\n\n部分2: KV Cache使用示例")
    print("-" * 70)
    
    # 创建KV Cache管理器
    print("\n步骤1: 初始化KV Cache Manager")
    manager = KVCacheManager(
        num_layers=2,
        batch_size=1,
        num_heads=4,
        max_seq_len=128,
        head_dim=16,
        device=torch.device('cpu')
    )
    
    manager.initialize()
    print(f"  ✓ KV Cache已初始化")
    
    # 模拟生成过程
    print("\n步骤2: 模拟自回归生成")
    seq_length = 10
    
    for step in range(seq_length):
        # 在实际使用中，这里会：
        # 1. 将新token输入模型
        # 2. 模型计算Q, K, V
        # 3. 更新KV Cache
        # 4. 使用cache中的K/V计算attention
        
        # 这里我们只演示cache的更新
        position = manager.get_current_length()
        
        # 模拟每一层的K/V更新
        for layer_idx in range(2):
            # 生成新的K/V（实际中由模型计算）
            new_k = torch.randn(1, 4, 1, 16)
            new_v = torch.randn(1, 4, 1, 16)
            
            # 更新cache
            manager.update_layer(layer_idx, new_k, new_v, position)
        
        current_length = manager.get_current_length()
        print(f"  Step {step+1:2d}: 缓存长度 = {current_length:3d}")
    
    # 3. 显示统计信息
    print("\n步骤3: 缓存统计信息")
    stats = manager.get_stats()
    print(f"  - 当前长度: {stats['current_length']}")
    print(f"  - 最大长度: {stats['max_seq_len']}")
    print(f"  - 内存占用: {stats['memory_kb']:.2f} KB")
    print(f"  - 利用率: {stats['utilization']*100:.1f}%")
    
    # 4. 重置cache
    print("\n步骤4: 重置缓存")
    manager.reset()
    print(f"  ✓ 缓存已重置，当前长度: {manager.get_current_length()}")
    
    print("\n✓ KV Cache演示完成!")
    print("=" * 70)


def demo_integration():
    """
    演示如何将KV Cache集成到Transformer模型中
    
    这是一个概念性演示，展示如何在实际的模型中使用KV Cache
    """
    print("\n\n" + "=" * 70)
    print("KV Cache与Transformer集成示例（概念演示）")
    print("=" * 70)
    
    print("""
在实际的Transformer实现中，需要修改Decoder层以支持KV Cache：

1. 修改Self-Attention层：
   ```python
   def forward(self, x, cache=None, cache_position=None):
       Q = self.linear_q(x)
       K = self.linear_k(x)
       V = self.linear_v(x)
       
       if cache is not None:
           # 从cache获取历史的K/V
           cached_k, cached_v = cache.get(self.layer_idx)
           
           # 将新的K/V添加到cache
           cache.update(self.layer_idx, K, V, cache_position)
           
           # 拼接历史和新计算的K/V
           K = torch.cat([cached_k, K], dim=2)
           V = torch.cat([cached_v, V], dim=2)
       
       # 正常的attention计算
       attention_output = self.attention(Q, K, V)
       return attention_output
   ```

2. 修改生成流程：
   ```python
   # Prompt处理阶段（一次性处理）
   outputs = model(prompt_ids, use_cache=False)
   
   # 初始化cache
   cache_manager.initialize()
   
   # 生成阶段（逐步生成）
   for step in range(max_steps):
       # 使用cache加速
       outputs = model(
           current_token, 
           use_cache=True,
           cache_manager=cache_manager,
           cache_position=step
       )
       
       # 采样下一个token
       next_token = sample(outputs)
   ```

3. 性能提升：
   - 无Cache: 每步O(n²)，总复杂度O(n³)
   - 有Cache: 每步O(n)，总复杂度O(n²)
   - 对于长序列，加速可达10-100倍
""")
    
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("LLM CodeGen Demo - 训练系统和KV Cache演示")
    print("🚀" * 35 + "\n")
    
    # 运行演示
    try:
        demo_training()
    except Exception as e:
        print(f"\n⚠️  训练演示遇到问题: {e}")
        print("这可能是由于缺少完整的tokenizer或其他依赖")
    
    demo_kv_cache()
    demo_integration()
    
    print("\n\n" + "✨" * 35)
    print("所有演示完成！")
    print("✨" * 35)
    print("\n下一步：")
    print("  1. 阅读 docs/TRAINING_AND_KV_CACHE_GUIDE.md 了解详细文档")
    print("  2. 尝试在自己的数据上训练模型")
    print("  3. 将KV Cache集成到现有的generator.py中")
