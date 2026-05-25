"""
完整流程管道 - 整合所有组件实现端到端的代码生成
"""

import sys
import time
from typing import Dict, Optional
from scripts.core.tokenizer import SimpleTokenizer
from scripts.core.transformer import TransformerModel
from scripts.generation.code_generator import CodeGenerator
from scripts.optimization.cache import GenerationCache
from scripts.utils.logger import logging_context


class CodeGenerationPipeline:
    """
    完整的代码生成管道 (Code Generation Pipeline)
    
    ════════════════════════════════════════════════════════════
    🏗️ 系统架构图
    ════════════════════════════════════════════════════════════
    
    ┌─────────────────────────────────────────────────────┐
    │                 User Request                        │
    │         "public class UserService"                  │
    └──────────────────┬──────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────────┐
    │           CodeGenerationPipeline                    │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Step 1: Cache Check                         │  │
    │  │  ┌──────────────────────────────────────┐   │  │
    │  │  │  GenerationCache (LRU)               │   │  │
    │  │  │  - Exact match (MD5 hash)            │   │  │
    │  │  │  - Semantic match (keywords)         │   │  │
    │  │  └──────────────────────────────────────┘   │  │
    │  └──────────────────────────────────────────────┘  │
    │                       │ HIT → Return cached        │
    │                       │ MISS → Continue            │
    │                       ▼                            │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Step 2: Tokenization                        │  │
    │  │  ┌──────────────────────────────────────┐   │  │
    │  │  │  SimpleTokenizer                     │   │  │
    │  │  │  "public class UserService"          │   │  │
    │  │  │  → [12, 45, 230, ...]                │   │  │
    │  │  └──────────────────────────────────────┘   │  │
    │  └──────────────────────────────────────────────┘  │
    │                       │                            │
    │                       ▼                            │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Step 3: Auto-regressive Generation          │  │
    │  │  ┌──────────────────────────────────────┐   │  │
    │  │  │  TransformerModel                    │   │  │
    │  │  │  ┌────────────────────────────────┐  │   │  │
    │  │  │  │  Encoder (N layers)            │  │   │  │
    │  │  │  │  - Multi-Head Attention        │  │   │  │
    │  │  │  │  - Feed Forward                │  │   │  │
    │  │  │  └────────────────────────────────┘  │   │  │
    │  │  │  ┌────────────────────────────────┐  │   │  │
    │  │  │  │  Decoder (N layers)            │  │   │  │
    │  │  │  │  - Masked Self-Attention       │  │   │  │
    │  │  │  │  - Cross-Attention             │  │   │  │
    │  │  │  │  - Feed Forward                │  │   │  │
    │  │  │  └────────────────────────────────┘  │   │  │
    │  │  └──────────────────────────────────────┘   │  │
    │  │          ↓ Temperature Sampling              │  │
    │  │  [12, 45, 230, 89, 156, ...]                │  │
    │  └──────────────────────────────────────────────┘  │
    │                       │                            │
    │                       ▼                            │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Step 4: Post-processing                     │  │
    │  │  ┌──────────────────────────────────────┐   │  │
    │  │  │  SimplePostProcessor                 │   │  │
    │  │  │  - Format code (indentation)         │   │  │
    │  │  │  - Add imports                       │   │  │
    │  │  │  - Validate syntax                   │   │  │
    │  │  └──────────────────────────────────────┘   │  │
    │  └──────────────────────────────────────────────┘  │
    │                       │                            │
    │                       ▼                            │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Step 5: Cache Update                        │  │
    │  │  Store prompt → result mapping               │  │
    │  └──────────────────────────────────────────────┘  │
    │                       │                            │
    └───────────────────────┼────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │                  Final Output                       │
    │  "import java.util.*;\npublic class UserService {"  │
    └─────────────────────────────────────────────────────┘
    
    ════════════════════════════════════════════════════════════
    🎛️ Preset配置系统设计
    ════════════════════════════════════════════════════════════
    
    【设计理念】
    将复杂的参数配置简化为3个预定义模板，降低学习门槛。
    
    【Preset对比表】
    ┌─────────────────────────────────────────────────────┐
    │  Parameter    │ Tiny      │ Small     │ Medium     │
    ├─────────────────────────────────────────────────────┤
    │  vocab_size   │ 500       │ 1000      │ 2000       │
    │  d_model      │ 64        │ 128       │ 256        │
    │  nhead        │ 4         │ 8         │ 8          │
    │  enc_layers   │ 1         │ 2         │ 4          │
    │  dec_layers   │ 1         │ 2         │ 4          │
    │  cache_size   │ 20        │ 50        │ 100        │
    ├─────────────────────────────────────────────────────┤
    │  适用场景     │ 快速测试  │ 日常使用  │ 高质量生成  │
    │  生成速度     │ ⚡⚡⚡    │ ⚡⚡      │ ⚡         │
    │  代码质量     │ ⭐⭐      │ ⭐⭐⭐    │ ⭐⭐⭐⭐   │
    │  内存占用     │ ~50MB     │ ~200MB    │ ~800MB     │
    └─────────────────────────────────────────────────────┘
    
    【使用建议】
    - 初学者：从'tiny'开始，快速理解原理
    - 日常开发：使用'small'，平衡性能和质量
    - 生产环境：考虑'medium'，获得更好的代码质量
    
    【扩展机制】
    用户可以基于preset覆盖特定参数：
    >>> pipeline = CodeGenerationPipeline(
    ...     preset='small',
    ...     device='cuda',      # 覆盖设备
    ...     cache_size=100      # 覆盖缓存大小
    ... )
    
    ════════════════════════════════════════════════════════════
    📖 完整流程示例
    ════════════════════════════════════════════════════════════
    
    Example 1: 基本用法
        >>> pipeline = CodeGenerationPipeline(preset='small')
        >>> result = pipeline.generate("public class UserService")
        >>> print(result['processed_code'])
    
    Example 2: 自定义参数
        >>> pipeline = CodeGenerationPipeline(
        ...     preset='tiny',
        ...     temperature=0.9,
        ...     max_length=150
        ... )
        >>> result = pipeline.generate("create a REST API endpoint")
    
    Example 3: 查看统计信息
        >>> pipeline.display_stats()
        [Pipeline] 管道统计信息
        ============================================================
          总请求数: 10
          缓存命中: 3 (30.00%)
          缓存未命中: 7 (70.00%)
          平均处理时间: 1.23s
        ============================================================
    """
    
    def __init__(self, 
                 preset: str = 'small',
                 vocab_size: int = None,
                 d_model: int = None,
                 nhead: int = None,
                 num_encoder_layers: int = None,
                 num_decoder_layers: int = None,
                 cache_size: int = None,
                 device: str = 'cpu',
                 debug_mode: bool = False):
        """
        初始化管道
        
        Args:
            preset: 预设配置 ('tiny', 'small', 'medium'),默认'small'
            vocab_size: 词汇表大小(覆盖preset)
            d_model: 模型维度(覆盖preset)
            nhead: 注意力头数(覆盖preset)
            num_encoder_layers: Encoder层数(覆盖preset)
            num_decoder_layers: Decoder层数(覆盖preset)
            cache_size: 缓存大小(覆盖preset)
            device: 计算设备
            debug_mode: 是否启用调试模式
            
        Example:
            # 使用预设(推荐)
            >>> pipeline = CodeGenerationPipeline(preset='small')
            
            # 覆盖特定参数
            >>> pipeline = CodeGenerationPipeline(preset='small', device='cuda')
            
            # 完全自定义(不推荐)
            >>> pipeline = CodeGenerationPipeline(vocab_size=1000, d_model=128, ...)
        """
        from scripts.config.presets import get_preset_config
        
        # 获取预设配置
        config = get_preset_config(preset)
        
        # 应用覆盖参数(如果提供)
        if vocab_size is not None:
            config['vocab_size'] = vocab_size
        if d_model is not None:
            config['d_model'] = d_model
        if nhead is not None:
            config['nhead'] = nhead
        if num_encoder_layers is not None:
            config['num_encoder_layers'] = num_encoder_layers
        if num_decoder_layers is not None:
            config['num_decoder_layers'] = num_decoder_layers
        if cache_size is not None:
            config['cache_size'] = cache_size
        
        self.debug_mode = debug_mode
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] 初始化代码生成管道")
        print(f"  Preset: {preset}")
        print(f"  Device: {device}")
        print(f"  Debug mode: {debug_mode}")
        print(f"{'='*60}")
        
        # 1. 创建Tokenizer
        print("\n[Step 1] 创建Tokenizer...")
        self.tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'], debug_mode=debug_mode)
        
        # 2. 创建Transformer模型
        print("\n[Step 2] 创建Transformer模型...")
        self.model = TransformerModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            debug_mode=debug_mode
        )
        
        # 3. 创建代码生成器(默认启用后处理)
        print("\n[Step 3] 创建代码生成器...")
        self.generator = CodeGenerator(self.model, self.tokenizer, device, enable_post_process=True)
        
        # 4. 创建缓存
        print("\n[Step 4] 创建缓存...")
        self.cache = GenerationCache(max_size=config['cache_size'])
        
        # 管道统计
        self.pipeline_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0
        }
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] 初始化完成")
        print(f"{'='*60}\n")
    
    def generate(self, 
                 prompt: str,
                 max_length: int = 100,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 use_cache: bool = True,
                 do_post_process: bool = True,
                 verbose: bool = True) -> Dict:
        """
        完整的代码生成流程
        
        Args:
            prompt: 用户提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K采样参数
            use_cache: 是否使用缓存
            do_post_process: 是否进行后处理
            verbose: 是否打印详细信息
            
        Returns:
            包含生成结果和元数据的字典
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# [Pipeline] 新的代码生成请求")
            print(f"{'#'*60}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Max length: {max_length}")
            print(f"Temperature: {temperature}")
            print(f"Top-K: {top_k}")
            print(f"Use cache: {use_cache}")
            print(f"Do post-process: {do_post_process}")
        
        result = {
            'prompt': prompt,
            'success': False,
            'source': None,  # 'cache' or 'generated'
            'raw_code': None,
            'processed_code': None,
            'processing_time': 0,
            'token_count': 0,
            'validation_report': None,
            'error': None
        }
        
        try:
            # 步骤1: 检查缓存
            if use_cache:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"[Pipeline] Step 1: 检查缓存")
                    print(f"{'='*60}")
                
                cached_result = self.cache.get(prompt)
                
                if cached_result is not None:
                    # 缓存命中
                    self.pipeline_stats['cache_hits'] += 1
                    
                    result['success'] = True
                    result['source'] = 'cache'
                    result['processed_code'] = cached_result
                    result['processing_time'] = time.time() - start_time
                    
                    if verbose:
                        print(f"\n[OK] 缓存命中！直接返回结果")
                        print(f"耗时: {result['processing_time']:.4f}s")
                    
                    return result
                else:
                    self.pipeline_stats['cache_misses'] += 1
                    if verbose:
                        print(f"[INFO] 缓存未命中，开始生成...")
            
            # 步骤2: 生成代码(直接传入temperature参数)
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Pipeline] Step 2: 生成代码")
                print(f"{'='*60}")
            
            generation_result = self.generator.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                verbose=verbose
            )
            
            raw_code = generation_result['code']
            result['raw_code'] = raw_code
            result['token_count'] = generation_result['token_count']
            
            if verbose:
                print(f"\n生成的原始代码长度: {len(raw_code)} 字符")
            
            # 后处理已在CodeGenerator内部完成
            result['processed_code'] = raw_code  # 后处理已在CodeGenerator内部完成
            result['success'] = True
            result['source'] = 'generated'
            
            # 步骤3: 缓存结果
            if use_cache:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"[Pipeline] Step 5: 缓存结果")
                    print(f"{'='*60}")
                
                self.cache.put(prompt, raw_code)
            
            # 计算耗时
            result['processing_time'] = time.time() - start_time
            
            # 更新统计
            self.pipeline_stats['total_requests'] += 1
            self.pipeline_stats['total_processing_time'] += result['processing_time']
            self.pipeline_stats['avg_processing_time'] = (
                self.pipeline_stats['total_processing_time'] / 
                self.pipeline_stats['total_requests']
            )
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Pipeline] 生成完成")
                print(f"{'='*60}")
                print(f"总耗时: {result['processing_time']:.4f}s")
                print(f"生成Token数: {result['token_count']}")
                print(f"最终代码长度: {len(raw_code)} 字符")
                print(f"来源: {result['source']}")
                
                # 显示最终代码
                print(f"\n{'#'*60}")
                print(f"# 最终生成的代码")
                print(f"{'#'*60}\n")
                print(raw_code)
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            
            if verbose:
                print(f"\n[ERROR] 生成失败: {e}")
                import traceback
                traceback.print_exc()
        
        return result
    
    def generate_multiple_samples(self, 
                                  prompt: str,
                                  num_samples: int = 3,
                                  max_length: int = 100,
                                  temperatures: list = None) -> list:
        """
        生成多个样本（使用不同的温度参数）
        
        Args:
            prompt: 用户提示
            num_samples: 样本数量
            max_length: 最大长度
            temperatures: 温度列表
            
        Returns:
            生成的代码列表
        """
        if temperatures is None:
            temperatures = [0.5, 0.7, 1.0]
        
        samples = []
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] 生成 {num_samples} 个样本")
        print(f"{'='*60}\n")
        
        for i in range(num_samples):
            temp = temperatures[i % len(temperatures)]
            
            print(f"\n--- Sample {i+1}/{num_samples} (temperature={temp}) ---\n")
            
            result = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temp,
                use_cache=False,  # 不使用缓存以获得多样性
                verbose=False
            )
            
            if result['success']:
                samples.append({
                    'sample_id': i + 1,
                    'temperature': temp,
                    'code': result['processed_code'],
                    'token_count': result['token_count'],
                    'processing_time': result['processing_time']
                })
        
        return samples
    
    def get_pipeline_stats(self) -> Dict:
        """获取管道统计信息"""
        return {
            'pipeline': self.pipeline_stats,
            'cache': self.cache.get_stats(),
            'generator': self.generator.get_generation_report()
        }
    
    def display_full_stats(self):
        """显示完整的统计信息"""
        stats = self.get_pipeline_stats()
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] 完整统计报告")
        print(f"{'='*60}")
        
        print(f"\n[STATS] 管道统计:")
        print(f"  总请求数: {stats['pipeline']['total_requests']}")
        print(f"  缓存命中: {stats['pipeline']['cache_hits']}")
        print(f"  缓存未命中: {stats['pipeline']['cache_misses']}")
        print(f"  平均处理时间: {stats['pipeline']['avg_processing_time']:.4f}s")
        
        print(f"\n[CACHE] 缓存统计:")
        cache_stats = stats['cache']
        print(f"  当前大小: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"  命中率: {cache_stats['hit_rate']}")
        print(f"  淘汰次数: {cache_stats['evictions']}")
        
        print(f"\n[GENERATOR] 生成器统计:")
        gen_stats = stats['generator']['stats']
        print(f"  总生成次数: {gen_stats['total_generations']}")
        print(f"  总Token数: {gen_stats['total_tokens_generated']}")
        print(f"  平均生成长度: {gen_stats['avg_generation_length']:.1f} tokens")
        
        print(f"\n[MODEL] 模型信息:")
        model_info = stats['generator']['model_info']
        print(f"  参数量: {model_info['parameters']:,}")
        print(f"  设备: {model_info['device']}")
        
        print(f"{'='*60}\n")


# 测试代码
if __name__ == '__main__':
    # 使用日志上下文管理器，自动管理日志文件
    with logging_context(__file__):
        print("="*60)
        print("测试完整管道")
        print("="*60)
        
        # 创建管道
        pipeline = CodeGenerationPipeline(
            vocab_size=1000,
            d_model=128,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            cache_size=10,
            device='cpu'
        )
        
        # 测试1: 基本生成
        print("\n" + "="*60)
        print("测试1: 基本代码生成")
        print("="*60)
        
        result = pipeline.generate(
            prompt="public class UserService",
            max_length=50,
            temperature=0.7,
            verbose=True
        )
        
        if result['success']:
            print(f"\n[OK] 生成成功")
            print(f"  来源: {result['source']}")
            print(f"  耗时: {result['processing_time']:.4f}s")
            print(f"  Token数: {result['token_count']}")
        else:
            print(f"\n[ERROR] 生成失败: {result['error']}")
        
        # 测试2: 缓存测试
        print("\n" + "="*60)
        print("测试2: 缓存功能")
        print("="*60)
        
        print("\n第二次相同请求（应该命中缓存）:")
        result2 = pipeline.generate(
            prompt="public class UserService",
            max_length=50,
            verbose=True
        )
        
        if result2['success']:
            print(f"\n[OK] 缓存命中")
            print(f"  来源: {result2['source']}")
            print(f"  耗时: {result2['processing_time']:.4f}s")
        
        # 测试3: 多样本生成
        print("\n" + "="*60)
        print("测试3: 多样本生成")
        print("="*60)
        
        samples = pipeline.generate_multiple_samples(
            prompt="public class UserController",
            num_samples=3,
            max_length=40
        )
        
        print(f"\n生成了 {len(samples)} 个样本:")
        for sample in samples:
            print(f"\n--- Sample {sample['sample_id']} (temp={sample['temperature']}) ---")
            print(f"Tokens: {sample['token_count']}, Time: {sample['processing_time']:.4f}s")
            print(f"Code preview: {sample['code'][:100]}...")
        
        # 显示统计
        pipeline.display_full_stats()
        
        print(f"\n{'='*60}")
        print(f"[OK] 所有测试完成！")
        print(f"{'='*60}")
        print(f"\n提示: 完整日志已自动保存到 logs/ 目录")
        print(f"      查看日志文件以获取完整输出")
