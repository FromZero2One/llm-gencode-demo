"""
KV Cache模块 - 实现高效的推理优化

KV Cache（Key-Value缓存）是LLM推理加速的核心技术。

核心思想：
在自回归生成过程中，每次只需要计算新生成token的K/V，
历史token的K/V可以缓存并复用，避免重复计算。

性能提升：
- 时间复杂度：从O(n²)降低到O(n)
- 实际加速：10-50倍（取决于序列长度）

教学要点：
- 理解为什么需要缓存
- 掌握缓存的实现细节
- 学会分析性能提升
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class KVCache:
    """
    Key-Value缓存
    
    存储Decoder中每个层的K和V张量，用于加速自回归生成。
    
    工作原理：
    1. 初始时cache为空
    2. 每生成一个token，计算其K/V并添加到cache
    3. 下次计算attention时，使用cache中的历史K/V + 新的K/V
    
    内存布局：
    对于每一层，存储：
    - key_cache: (batch_size, num_heads, max_seq_len, head_dim)
    - value_cache: (batch_size, num_heads, max_seq_len, head_dim)
    
    Example:
        >>> cache = KVCache(num_layers=2, batch_size=1, num_heads=8, 
        ...                 max_seq_len=128, head_dim=16)
        >>> # 第1步：处理prompt
        >>> cache.update(layer_idx=0, new_k=k1, new_v=v1, position=0)
        >>> # 第2步：生成第一个token
        >>> cached_k, cached_v = cache.get(layer_idx=0)
        >>> # 合并：cached_k + new_k, cached_v + new_v
    """
    
    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 max_seq_len: int, head_dim: int, device: torch.device = None):
        """
        初始化KV Cache
        
        Args:
            num_layers (int): Transformer的层数
                             每层都需要独立的K/V缓存
            
            batch_size (int): 批次大小
                             支持批量生成（多个序列同时生成）
            
            num_heads (int): 注意力头的数量
                            每个头有独立的K/V
            
            max_seq_len (int): 最大序列长度
                              预分配内存以提高效率
            
            head_dim (int): 每个注意力头的维度
                           head_dim = d_model / num_heads
            
            device (torch.device, optional): 设备（CPU或GPU）
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device or torch.device('cpu')
        
        # 当前缓存的长度（已填充的位置数）
        self.current_length = 0
        
        # 预分配缓存内存
        # 形状: (num_layers, batch_size, num_heads, max_seq_len, head_dim)
        self.key_cache = torch.zeros(
            (num_layers, batch_size, num_heads, max_seq_len, head_dim),
            device=self.device
        )
        self.value_cache = torch.zeros(
            (num_layers, batch_size, num_heads, max_seq_len, head_dim),
            device=self.device
        )
        
        logger.info(f"KV Cache初始化:")
        logger.info(f"  - 层数: {num_layers}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - 注意力头: {num_heads}")
        logger.info(f"  - 最大长度: {max_seq_len}")
        logger.info(f"  - 头维度: {head_dim}")
        logger.info(f"  - 总内存: {self._calculate_memory() / 1024:.2f} KB")
    
    def _calculate_memory(self) -> int:
        """计算缓存占用的内存（字节）"""
        # 每个tensor的大小：num_layers * batch * heads * max_len * head_dim * 4 bytes (float32)
        single_tensor_size = (
            self.num_layers * self.batch_size * self.num_heads * 
            self.max_seq_len * self.head_dim * 4
        )
        # K和V两个tensor
        return single_tensor_size * 2
    
    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor,
               position: Optional[int] = None):
        """
        更新缓存：添加新的K/V
        
        Args:
            layer_idx (int): 层索引（0到num_layers-1）
            
            new_k (torch.Tensor): 新的key张量
                                 形状: (batch_size, num_heads, seq_len, head_dim)
                                 seq_len通常是1（单步生成）
            
            new_v (torch.Tensor): 新的value张量
                                 形状: (batch_size, num_heads, seq_len, head_dim)
            
            position (int, optional): 插入位置
                                     如果为None，自动追加到末尾
        """
        if position is None:
            position = self.current_length
        
        # 获取要更新的seq_len
        seq_len = new_k.shape[2]
        
        # 检查边界
        if position + seq_len > self.max_seq_len:
            raise ValueError(
                f"KV Cache溢出! 当前位置: {position}, "
                f"新增长度: {seq_len}, 最大长度: {self.max_seq_len}"
            )
        
        # 将新的K/V复制到缓存中
        # 注意：这里使用切片赋值，避免重新分配内存
        self.key_cache[layer_idx, :, :, position:position+seq_len, :] = new_k
        self.value_cache[layer_idx, :, :, position:position+seq_len, :] = new_v
        
        # 更新当前长度
        self.current_length = max(self.current_length, position + seq_len)
    
    def get(self, layer_idx: int, length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取缓存的K/V
        
        Args:
            layer_idx (int): 层索引
            
            length (int, optional): 要获取的长度
                                   如果为None，返回所有缓存的内容
        
        Returns:
            tuple: (key, value)
                  key: (batch_size, num_heads, length, head_dim)
                  value: (batch_size, num_heads, length, head_dim)
        """
        if length is None:
            length = self.current_length
        
        if length == 0:
            # 空缓存，返回空tensor
            return (
                torch.zeros(
                    (self.batch_size, self.num_heads, 0, self.head_dim),
                    device=self.device
                ),
                torch.zeros(
                    (self.batch_size, self.num_heads, 0, self.head_dim),
                    device=self.device
                )
            )
        
        key = self.key_cache[layer_idx, :, :, :length, :]
        value = self.value_cache[layer_idx, :, :, :length, :]
        
        return key, value
    
    def reset(self):
        """重置缓存（清空所有内容）"""
        self.current_length = 0
        self.key_cache.zero_()
        self.value_cache.zero_()
        logger.debug("KV Cache已重置")
    
    def get_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            dict: 包含缓存状态的字典
        """
        return {
            'num_layers': self.num_layers,
            'batch_size': self.batch_size,
            'num_heads': self.num_heads,
            'current_length': self.current_length,
            'max_seq_len': self.max_seq_len,
            'head_dim': self.head_dim,
            'memory_kb': self._calculate_memory() / 1024,
            'utilization': self.current_length / self.max_seq_len
        }


class KVCacheManager:
    """
    KV Cache管理器
    
    管理多层Transformer的KV Cache，提供统一的接口。
    
    使用场景：
    1. Prompt处理阶段：一次性处理整个prompt，填充cache
    2. 生成阶段：逐步生成token，每次更新cache
    
    Example:
        >>> manager = KVCacheManager(model, batch_size=1, max_seq_len=128)
        >>> 
        >>> # 阶段1：处理prompt
        >>> outputs = model(prompt_ids, use_cache=True, cache_manager=manager)
        >>> 
        >>> # 阶段2：生成token
        >>> for step in range(max_steps):
        ...     next_token = sample(outputs)
        ...     outputs = model(next_token, use_cache=True, cache_manager=manager)
    """
    
    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 max_seq_len: int, head_dim: int, device: torch.device = None):
        """
        初始化KV Cache管理器
        
        Args:
            num_layers (int): Transformer层数
            batch_size (int): 批次大小
            num_heads (int): 注意力头数
            max_seq_len (int): 最大序列长度
            head_dim (int): 每个头的维度
            device (torch.device, optional): 设备
        """
        self.cache = KVCache(
            num_layers=num_layers,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            device=device
        )
        self.is_initialized = False
    
    def initialize(self):
        """初始化缓存（重置所有状态）"""
        self.cache.reset()
        self.is_initialized = True
        logger.info("KV Cache Manager已初始化")
    
    def update_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
                     position: Optional[int] = None):
        """
        更新某一层的缓存
        
        Args:
            layer_idx (int): 层索引
            k (torch.Tensor): Key张量
            v (torch.Tensor): Value张量
            position (int, optional): 位置
        """
        self.cache.update(layer_idx, k, v, position)
    
    def get_layer(self, layer_idx: int, length: Optional[int] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取某一层的缓存
        
        Args:
            layer_idx (int): 层索引
            length (int, optional): 长度
        
        Returns:
            tuple: (key, value)
        """
        return self.cache.get(layer_idx, length)
    
    def reset(self):
        """重置所有缓存"""
        self.cache.reset()
        self.is_initialized = False
    
    def get_current_length(self) -> int:
        """获取当前缓存长度"""
        return self.cache.current_length
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.cache.get_stats()


def demonstrate_kv_cache_benefit():
    """
    演示KV Cache的性能优势
    
    通过对比有无cache的计算量，直观展示加速效果。
    """
    print("=" * 70)
    print("KV Cache性能优势演示")
    print("=" * 70)
    
    # 假设参数
    seq_len = 100  # 序列长度
    d_model = 512
    num_heads = 8
    head_dim = d_model // num_heads
    
    print(f"\n配置:")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 模型维度: {d_model}")
    print(f"  - 注意力头: {num_heads}")
    print(f"  - 头维度: {head_dim}")
    
    # 无Cache的情况
    print(f"\n❌ 不使用KV Cache:")
    total_ops_no_cache = 0
    for i in range(1, seq_len + 1):
        # 每一步都要重新计算所有历史token的QK^T
        # 计算量 ≈ i * d_model * num_heads
        ops = i * d_model * num_heads
        total_ops_no_cache += ops
    
    print(f"  - 总计算量: {total_ops_no_cache:,} 次操作")
    print(f"  - 时间复杂度: O(n²)")
    
    # 有Cache的情况
    print(f"\n✅ 使用KV Cache:")
    total_ops_with_cache = 0
    for i in range(1, seq_len + 1):
        # 每一步只计算新token的K/V，然后与cache拼接
        # 计算量 ≈ 1 * d_model * num_heads（只算新token）
        ops = 1 * d_model * num_heads
        total_ops_with_cache += ops
    
    print(f"  - 总计算量: {total_ops_with_cache:,} 次操作")
    print(f"  - 时间复杂度: O(n)")
    
    # 加速比
    speedup = total_ops_no_cache / total_ops_with_cache
    print(f"\n📊 性能对比:")
    print(f"  - 加速比: {speedup:.1f}x")
    print(f"  - 节省计算: {(1 - 1/speedup) * 100:.1f}%")
    
    print(f"\n💡 结论:")
    print(f"  KV Cache将二次复杂度降为线性复杂度，")
    print(f"  对于长序列（如1000+ tokens），加速可达100倍以上！")
    print("=" * 70)


if __name__ == "__main__":
    # 运行演示
    demonstrate_kv_cache_benefit()
    
    # 测试KV Cache基本功能
    print("\n\n测试KV Cache功能...")
    
    device = torch.device('cpu')
    cache = KVCache(
        num_layers=2,
        batch_size=1,
        num_heads=4,
        max_seq_len=10,
        head_dim=16,
        device=device
    )
    
    # 模拟生成过程
    print("\n模拟生成长度为5的序列:")
    for step in range(5):
        # 生成新的K/V（这里用随机数模拟）
        new_k = torch.randn(1, 4, 1, 16, device=device)
        new_v = torch.randn(1, 4, 1, 16, device=device)
        
        # 更新cache
        cache.update(layer_idx=0, new_k=new_k, new_v=new_v)
        cache.update(layer_idx=1, new_k=new_k, new_v=new_v)
        
        # 获取cache
        cached_k, cached_v = cache.get(layer_idx=0)
        print(f"  Step {step+1}: 缓存长度 = {cached_k.shape[2]}")
    
    print(f"\n最终缓存统计:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ KV Cache测试完成!")
