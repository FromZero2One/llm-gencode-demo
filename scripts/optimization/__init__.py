"""
[Optimization] 优化模块包
========================
包含 KV Cache 和结果缓存机制
"""

from scripts.optimization.kv_cache import KVCache
from scripts.optimization.cache import GenerationCache

__all__ = [
    'KVCache',
    'GenerationCache'
]
