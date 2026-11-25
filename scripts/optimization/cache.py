"""
缓存模块 - 实现生成结果的缓存机制
提高重复或相似请求的响应速度

╔═══════════════════════════════════════════════════════════╗
║  📚 为什么需要缓存？                                     ║
╚═══════════════════════════════════════════════════════════╝

在代码生成场景中，用户经常会提出相似的请求：
  - "创建一个UserService类"
  - "创建一个UserController类"
  - "创建一个OrderService类"

如果没有缓存，每次都要重新生成，浪费计算资源。
通过缓存，我们可以：
  1. 精确匹配：完全相同的请求直接返回缓存结果
  2. 语义匹配：相似的请求可以复用已有的生成结果
  3. 性能提升：缓存命中时响应时间从秒级降至毫秒级

【LRU策略】
Least Recently Used（最近最少使用）是最常用的缓存淘汰策略：
  - 核心思想：如果数据最近被访问过，那么将来被访问的几率也更高
  - 实现方式：使用OrderedDict保持插入顺序，新访问的移到末尾
  - 淘汰规则：缓存满时，删除最久未访问的条目

【实际应用案例】
  - GitHub Copilot: 缓存常见的代码模式
  - Tabnine: 缓存频繁使用的代码片段
  - Amazon CodeWhisperer: 缓存安全相关的代码建议
"""

import sys
import hashlib
import time
import re
from typing import Optional, Dict, List
from collections import OrderedDict
from scripts.utils.logger import logging_context


class SemanticCache:
    """
    语义缓存 - 基于向量相似度的近似匹配缓存

    ════════════════════════════════════════════════════════════
    📚 原理：为什么要替代关键词 MD5？
    ════════════════════════════════════════════════════════════

    旧方案（GenerationCache.get_similar）的缺陷：
        提取 Java 关键字（public/class/Service...）排序后取 MD5。
        问题1: 中文自然语言（"创建一个用户服务类"）提取不到任何关键字
               → 所有中文 prompt 得到相同的空串 MD5 → 必然误命中！
        问题2: MD5 是雪崩哈希，"UserService" 和 "OrderService" 的
               语义差一个词，MD5 却完全不同 → 又漏掉了真相似对。

    新方案：字符 n-gram 词袋 + 余弦相似度
        1. 把 prompt 拆成字符 2-gram 的多重集合
           "用户服务" → {用户, 户服, 服务}
           （对中英文都鲁棒，无需分词器，零外部依赖）
        2. 每个条目存一个归一化向量
        3. 查询时算余弦相似度，超过阈值即命中

    余弦相似度公式：
        sim(a, b) = a · b / (||a|| * ||b||)
        值域 [0, 1]（n-gram 计数非负），1 表示完全相同。

    复杂度：put O(L)，get O(n·L)（n 为缓存条数）。
            教学规模（n≤几百）完全够用；生产环境应换 ANN 索引
            （如 FAISS / pgvector），此处注释说明即可。
    ════════════════════════════════════════════════════════════

    Args:
        max_size: 最大缓存条数（LRU 淘汰）
        ngram_size: 字符 n-gram 的 n（2 适合中英混合短文本）
        similarity_threshold: 命中阈值（0-1）。过高达不到模糊效果，
                              过低会误命中。0.7 左右适合短 prompt。
    """

    def __init__(self, max_size: int = 50, ngram_size: int = 2,
                 similarity_threshold: float = 0.7):
        self.max_size = max_size
        self.ngram_size = ngram_size
        self.similarity_threshold = similarity_threshold

        # prompt_hash -> {'prompt', 'result', 'vector', 'timestamp'}
        self.cache = OrderedDict()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0,
                      'semantic_hits': 0, 'exact_hits': 0}

        print(f"[SemanticCache] 初始化语义缓存")
        print(f"  - 最大容量: {max_size}")
        print(f"  - n-gram size: {ngram_size}")
        print(f"  - 相似度阈值: {similarity_threshold}")

    def _tokenize(self, text: str) -> List[str]:
        """
        提取字符 n-gram（对中英文统一处理）

        例（n=2）: "public class" → ['pu','ub','bl','li','ic','c ','cl',...]
                   "用户服务"     → ['用户','户服','服务']
        """
        text = text.strip().lower()
        if len(text) < self.ngram_size:
            return [text] if text else []
        return [text[i:i + self.ngram_size]
                for i in range(len(text) - self.ngram_size + 1)]

    def _vectorize(self, text: str) -> Dict[str, float]:
        """
        文本 → 归一化 n-gram 计数向量（稀疏 dict 表示）

        L2 归一化后，两个向量的点积即余弦相似度
        """
        vec: Dict[str, float] = {}
        for gram in self._tokenize(text):
            vec[gram] = vec.get(gram, 0.0) + 1.0
        norm = sum(v * v for v in vec.values()) ** 0.5
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """两个归一化稀疏向量的点积 = 余弦相似度"""
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    def get(self, prompt: str) -> Optional[str]:
        """
        查询缓存：先精确匹配（O(1)），再语义匹配（O(n)）

        Returns:
            命中返回缓存结果；未命中返回 None
        """
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()

        # 1. 精确匹配
        if prompt_hash in self.cache:
            self.stats['hits'] += 1
            self.stats['exact_hits'] += 1
            self.cache.move_to_end(prompt_hash)
            item = self.cache[prompt_hash]
            print(f"[SemanticCache] 精确命中: {prompt[:30]}...")
            return item['result']

        # 2. 语义匹配（余弦相似度）
        query_vec = self._vectorize(prompt)
        best_hash, best_sim = None, 0.0
        for h, item in self.cache.items():
            sim = self._cosine(query_vec, item['vector'])
            if sim > best_sim:
                best_hash, best_sim = h, sim

        if best_sim >= self.similarity_threshold:
            self.stats['hits'] += 1
            self.stats['semantic_hits'] += 1
            self.cache.move_to_end(best_hash)
            item = self.cache[best_hash]
            print(f"[SemanticCache] 语义命中 (sim={best_sim:.3f} >= {self.similarity_threshold}): "
                  f"缓存自 \"{item['prompt'][:30]}...\"")
            return item['result']

        self.stats['misses'] += 1
        if self.cache and best_hash is not None:
            print(f"[SemanticCache] 未命中 (最高相似度 {best_sim:.3f} < {self.similarity_threshold})")
        else:
            print(f"[SemanticCache] 未命中 (缓存为空)")
        return None

    def put(self, prompt: str, result: str):
        """存入缓存（LRU 淘汰）"""
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        if prompt_hash in self.cache:
            self.cache.move_to_end(prompt_hash)
            return

        if len(self.cache) >= self.max_size:
            evicted_hash, _ = self.cache.popitem(last=False)
            self.stats['evictions'] += 1
            print(f"[SemanticCache] LRU淘汰: {evicted_hash[:8]}...")

        self.cache[prompt_hash] = {
            'prompt': prompt,
            'result': result,
            'vector': self._vectorize(prompt),
            'timestamp': time.time(),
        }
        print(f"[SemanticCache] 存入 ({len(self.cache)}/{self.max_size}): {prompt[:30]}...")

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0,
                      'semantic_hits': 0, 'exact_hits': 0}
        print("[SemanticCache] 缓存已清空")

    def get_stats(self) -> Dict:
        """统计信息"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'exact_hits': self.stats['exact_hits'],
            'semantic_hits': self.stats['semantic_hits'],
            'evictions': self.stats['evictions'],
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total,
        }


class GenerationCache:
    """
    生成结果缓存 (Generation Cache)
    
    ════════════════════════════════════════════════════════════
    🎯 核心功能
    ════════════════════════════════════════════════════════════
    
    1. **精确缓存查找** (get/put)
       - 基于MD5哈希的快速查找 O(1)
       - 适用于完全相同的请求
       
    2. **语义缓存查找** (get_similar)
       - 提取关键特征（如关键字、类名）
       - 相似的请求可以复用缓存结果
       - 例如："创建UserService" 和 "创建UserController" 
               都包含"创建"和"Service/Controller"模式
    
    3. **LRU淘汰策略**
       - 缓存满时自动删除最久未使用的条目
       - 保持高频访问的内容在缓存中
    
    4. **统计信息**
       - 命中率、未命中率、淘汰次数
       - 帮助优化缓存大小配置
    
    ════════════════════════════════════════════════════════════
    📖 使用示例
    ════════════════════════════════════════════════════════════
    
    Example 1: 基本用法
        >>> cache = GenerationCache(max_size=100)
        >>> 
        >>> # 首次生成（未命中）
        >>> result = cache.get("public class UserService")
        >>> if result is None:
        ...     result = generate_code("public class UserService")
        ...     cache.put("public class UserService", result)
        >>> 
        >>> # 再次生成（命中）
        >>> cached_result = cache.get("public class UserService")
        >>> print(f"从缓存获取: {cached_result[:50]}...")
    
    Example 2: 语义缓存
        >>> cache = GenerationCache(max_size=100)
        >>> 
        >>> # 缓存一个请求
        >>> cache.put("创建UserService类", "class UserService { ... }")
        >>> 
        >>> # 查找相似的请求
        >>> similar = cache.get_similar("创建OrderService类")
        >>> if similar:
        ...     print(f"找到相似结果: {similar[:50]}...")
    
    Example 3: 查看统计信息
        >>> cache.display_stats()
        [Cache] 缓存统计信息
        ============================================================
          当前大小: 50/100
          总请求数: 200
          命中次数: 80
          未命中次数: 120
          淘汰次数: 10
          命中率: 40.00%
          平均访问次数: 2.50
        ============================================================
    
    ════════════════════════════════════════════════════════════
    🔧 内部实现细节
    ════════════════════════════════════════════════════════════
    
    数据结构:
      - cache: OrderedDict[str, Dict]  # 哈希 -> 缓存条目
      - access_count: Dict[str, int]   # 哈希 -> 访问次数
      - stats: Dict                    # 统计信息
    
    缓存条目结构:
      {
          'prompt': str,              # 原始提示文本
          'result': str,              # 生成结果
          'semantic_hash': str,       # 语义哈希
          'timestamp': float,         # 时间戳
          'prompt_length': int,       # 提示长度
          'result_length': int        # 结果长度
      }
    
    哈希算法:
      - 精确哈希: MD5(prompt) → 64位十六进制字符串
      - 语义哈希: MD5(sorted(keywords)) → 提取关键字后排序再哈希
    
    时间复杂度:
      - get(): O(1)  # 哈希表查找
      - put(): O(1)  # 哈希表插入
      - get_similar(): O(n)  # 需要遍历所有缓存条目
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()  # 保持插入顺序
        self.access_count = {}  # 访问计数
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        print(f"[Cache] 初始化缓存")
        print(f"  - 最大容量: {max_size}")
    
    def _compute_hash(self, text: str) -> str:
        """
        计算文本的哈希值
        
        Args:
            text: 输入文本
            
        Returns:
            MD5哈希字符串
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _compute_semantic_hash(self, text: str) -> str:
        """
        计算语义哈希（简化版本）
        
        提取关键特征来计算哈希，使得相似的请求有相同的哈希
        
        Args:
            text: 输入文本
            
        Returns:
            语义哈希字符串
        """
        # 提取关键字（简化：只提取Java关键字和类名）
        import re
        keywords = re.findall(r'\b(public|private|class|interface|method|function|Service|Controller)\w*', 
                             text, re.IGNORECASE)
        
        # 排序以确保相同内容产生相同哈希
        semantic_content = ''.join(sorted(keywords))
        
        return hashlib.md5(semantic_content.encode('utf-8')).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        """
        从缓存中获取结果
        
        Args:
            prompt: 提示文本
            
        Returns:
            缓存的结果，如果不存在返回None
        """
        prompt_hash = self._compute_hash(prompt)
        
        if prompt_hash in self.cache:
            # 命中缓存
            self.stats['hits'] += 1
            
            # 移动到末尾（标记为最近使用）
            self.cache.move_to_end(prompt_hash)
            self.access_count[prompt_hash] = self.access_count.get(prompt_hash, 0) + 1
            
            cached_item = self.cache[prompt_hash]
            
            print(f"[Cache] [HIT] Cache HIT")
            print(f"  - Prompt hash: {prompt_hash[:8]}...")
            print(f"  - Access count: {self.access_count[prompt_hash]}")
            print(f"  - Cache size: {len(self.cache)}/{self.max_size}")
            
            return cached_item['result']
        else:
            # 未命中
            self.stats['misses'] += 1
            
            print(f"[Cache] [MISS] Cache MISS")
            print(f"  - Prompt hash: {prompt_hash[:8]}...")
            print(f"  - Cache size: {len(self.cache)}/{self.max_size}")
            
            return None
    
    def get_similar(self, prompt: str, similarity_threshold: float = 0.9) -> Optional[str]:
        """
        查找相似的缓存结果
        
        Args:
            prompt: 提示文本
            similarity_threshold: 相似度阈值（0-1）
            
        Returns:
            相似的缓存结果，如果不存在返回None
        """
        semantic_hash = self._compute_semantic_hash(prompt)
        
        # 检查是否有相同的语义哈希
        for cached_hash, cached_item in self.cache.items():
            if cached_item.get('semantic_hash') == semantic_hash:
                self.stats['hits'] += 1
                self.cache.move_to_end(cached_hash)
                
                print(f"[Cache] [SIMILAR-HIT] Similar cache HIT (semantic match)")
                print(f"  - Original prompt: {cached_item['prompt'][:50]}...")
                
                return cached_item['result']
        
        self.stats['misses'] += 1
        return None
    
    def put(self, prompt: str, result: str):
        """
        将结果存入缓存
        
        Args:
            prompt: 提示文本
            result: 生成结果
        """
        prompt_hash = self._compute_hash(prompt)
        semantic_hash = self._compute_semantic_hash(prompt)
        
        # 如果缓存已满，移除最久未使用的
        if len(self.cache) >= self.max_size:
            evicted_hash, _ = self.cache.popitem(last=False)
            if evicted_hash in self.access_count:
                del self.access_count[evicted_hash]
            self.stats['evictions'] += 1
            
            print(f"[Cache] Evicted oldest entry")
            print(f"  - Evicted hash: {evicted_hash[:8]}...")
        
        # 添加新条目
        self.cache[prompt_hash] = {
            'prompt': prompt,
            'result': result,
            'semantic_hash': semantic_hash,
            'timestamp': time.time(),
            'prompt_length': len(prompt),
            'result_length': len(result)
        }
        
        self.access_count[prompt_hash] = 1
        
        print(f"[Cache] Added new entry")
        print(f"  - Hash: {prompt_hash[:8]}...")
        print(f"  - Prompt length: {len(prompt)}")
        print(f"  - Result length: {len(result)}")
        print(f"  - Cache size: {len(self.cache)}/{self.max_size}")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
        print("[Cache] Cache cleared")
    
    def get_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            包含统计信息的字典
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total_requests,
            'avg_access_count': (
                sum(self.access_count.values()) / len(self.access_count)
                if self.access_count else 0
            )
        }
    
    def display_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print(f"[Cache] 缓存统计信息")
        print(f"{'='*60}")
        print(f"  当前大小: {stats['size']}/{stats['max_size']}")
        print(f"  总请求数: {stats['total_requests']}")
        print(f"  命中次数: {stats['hits']}")
        print(f"  未命中次数: {stats['misses']}")
        print(f"  淘汰次数: {stats['evictions']}")
        print(f"  命中率: {stats['hit_rate']}")
        print(f"  平均访问次数: {stats['avg_access_count']:.2f}")
        print(f"{'='*60}\n")
    
    def list_entries(self) -> List[Dict]:
        """
        列出所有缓存条目
        
        Returns:
            缓存条目列表
        """
        entries = []
        for hash_key, item in self.cache.items():
            entries.append({
                'hash': hash_key[:8],
                'prompt_preview': item['prompt'][:50],
                'result_length': item['result_length'],
                'access_count': self.access_count.get(hash_key, 0),
                'timestamp': item['timestamp']
            })
        
        return entries


# 测试代码
if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        print("="*60)
        print("测试缓存模块")
        print("="*60)
        
        # 创建缓存
        cache = GenerationCache(max_size=5)
    
    # 测试基本操作
    print("\n--- 测试1: 基本缓存操作 ---")
    
    prompt1 = "public class UserService"
    result1 = "生成的UserService代码..."
    
    # 首次访问（未命中）
    print("\n首次访问:")
    cached = cache.get(prompt1)
    print(f"缓存结果: {cached}")
    
    # 存入缓存
    print("\n存入缓存:")
    cache.put(prompt1, result1)
    
    # 再次访问（命中）
    print("\n再次访问:")
    cached = cache.get(prompt1)
    print(f"缓存结果: {cached}")
    
    # 测试多个条目
    print("\n--- 测试2: 多个条目 ---")
    
    for i in range(6):  # 超过max_size，会触发淘汰
        prompt = f"public class TestClass{i}"
        result = f"Generated code for TestClass{i}"
        cache.put(prompt, result)
    
    cache.display_stats()
    
    # 测试语义缓存
    print("\n--- 测试3: 语义缓存 ---")
    
    cache.clear()
    
    prompt_a = "创建一个UserService类包含findById方法"
    prompt_b = "创建一个UserService类包含findAll方法"
    
    # 这两个prompt应该有相似的语义哈希
    cache.put(prompt_a, "Result A")
    
    similar_result = cache.get_similar(prompt_b)
    print(f"相似搜索结果: {similar_result}")
    
    # 显示最终统计
    cache.display_stats()
    
    # 列出所有条目
    print("\n缓存条目列表:")
    entries = cache.list_entries()
    for entry in entries:
        print(f"  - Hash: {entry['hash']}, Preview: {entry['prompt_preview']}, "
              f"Access: {entry['access_count']}")
    
    # ==================== 测试4: SemanticCache（向量语义缓存） ====================
    print("\n" + "="*60)
    print("测试4: SemanticCache (n-gram 向量语义缓存)")
    print("="*60)
    
    from scripts.optimization.cache import SemanticCache
    
    scache = SemanticCache(max_size=10, similarity_threshold=0.7)
    
    # 中文场景（旧方案的关键词 MD5 在此必然误命中）
    scache.put("创建一个用户服务类", "class UserService { ... }")
    
    print("\n[场景1] 中文相似请求（应语义命中）:")
    r1 = scache.get("创建一个用户服务类")           # 精确命中
    r2 = scache.get("创建一个用户服务类，包含增删改查")  # 语义命中
    print(f"  精确查询命中: {r1 is not None}")
    print(f"  相似查询命中: {r2 is not None}")
    
    print("\n[场景2] 不相关请求（应未命中，不误报）:")
    r3 = scache.get("def fibonacci(n): return n if n<2 else fib(n-1)+fib(n-2)")
    print(f"  完全不同的请求命中: {r3 is not None} (应为 False)")
    
    print("\n[场景3] 英文代码请求:")
    scache.put("public class UserService {", "Result B")
    r4 = scache.get("public class UserService extends Base {")  # 应语义命中
    print(f"  英文相似请求命中: {r4 is not None}")
    
    stats = scache.get_stats()
    print(f"\n统计: {stats}")
    
    assert r1 is not None, "精确查询应命中"
    assert r2 is not None, "相似查询应语义命中"
    assert r3 is None, "不相关请求不应命中（旧方案会误命中！）"
    print("\n[OK] SemanticCache 验证通过：中文语义命中、不相关不误报")
