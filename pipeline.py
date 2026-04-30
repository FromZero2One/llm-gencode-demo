"""
完整流程管道 - 整合所有组件实现端到端的代码生成
"""

import time
from typing import Dict, Optional
from tokenizer import SimpleTokenizer
from transformer import TransformerModel
from generator import CodeGenerator, SamplingStrategy, TemperatureSampling
from postprocessor import CodePostProcessor
from cache import GenerationCache


class CodeGenerationPipeline:
    """
    完整的代码生成管道
    
    流程：
    1. 接收用户请求
    2. 检查缓存
    3. 预处理（tokenization）
    4. 模型推理和生成
    5. 后处理（格式化、验证）
    6. 缓存结果
    7. 返回最终代码
    """
    
    def __init__(self, 
                 vocab_size: int = 1000,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 cache_size: int = 50,
                 device: str = 'cpu',
                 enable_attention_tracking: bool = False):
        """
        初始化管道
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            nhead: 注意力头数
            num_encoder_layers: Encoder层数
            num_decoder_layers: Decoder层数
            cache_size: 缓存大小
            device: 计算设备
            enable_attention_tracking: 是否启用注意力追踪
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline] 初始化代码生成管道")
        print(f"{'='*60}")
        
        # 1. 创建Tokenizer
        print("\n[Step 1] 创建Tokenizer...")
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        
        # 2. 创建Transformer模型
        print("\n[Step 2] 创建Transformer模型...")
        self.model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            enable_attention_tracking=enable_attention_tracking
        )
        
        # 3. 创建代码生成器
        print("\n[Step 3] 创建代码生成器...")
        self.generator = CodeGenerator(self.model, self.tokenizer, device)
        
        # 4. 创建后处理器
        print("\n[Step 4] 创建后处理器...")
        self.post_processor = CodePostProcessor()
        
        # 5. 创建缓存
        print("\n[Step 5] 创建缓存...")
        self.cache = GenerationCache(max_size=cache_size)
        
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
                 verbose: bool = True,
                 enable_probability_analysis: bool = False) -> Dict:
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
            enable_probability_analysis: 是否启用概率分布分析
            
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
            print(f"Probability analysis: {enable_probability_analysis}")
        
        result = {
            'prompt': prompt,
            'success': False,
            'source': None,  # 'cache' or 'generated'
            'raw_code': None,
            'processed_code': None,
            'processing_time': 0,
            'token_count': 0,
            'validation_report': None,
            'probability_analyses': [],
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
            
            # 步骤2: 创建采样策略
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Pipeline] Step 2: 创建采样策略")
                print(f"{'='*60}")
            
            strategy = TemperatureSampling(temperature=temperature)
            
            # 步骤3: 生成代码
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Pipeline] Step 3: 生成代码")
                print(f"{'='*60}")
            
            generation_result = self.generator.generate(
                prompt=prompt,
                max_length=max_length,
                strategy=strategy,
                verbose=verbose,
                enable_probability_analysis=enable_probability_analysis
            )
            
            raw_code = generation_result['generated_code']
            result['raw_code'] = raw_code
            result['token_count'] = generation_result['token_count']
            result['probability_analyses'] = generation_result.get('probability_analyses', [])
            
            if verbose:
                print(f"\n生成的原始代码长度: {len(raw_code)} 字符")
            
            # 步骤4: 后处理
            final_code = raw_code
            if do_post_process:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"[Pipeline] Step 4: 后处理")
                    print(f"{'='*60}")
                
                post_process_result = self.post_processor.process(raw_code)
                final_code = post_process_result['processed_code']
                result['validation_report'] = post_process_result['validation_report']
                
                if verbose:
                    print(f"\n后处理后的代码长度: {len(final_code)} 字符")
            
            result['processed_code'] = final_code
            result['success'] = True
            result['source'] = 'generated'
            
            # 步骤5: 缓存结果
            if use_cache:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"[Pipeline] Step 5: 缓存结果")
                    print(f"{'='*60}")
                
                self.cache.put(prompt, final_code)
            
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
                print(f"最终代码长度: {len(final_code)} 字符")
                print(f"来源: {result['source']}")
                
                # 显示最终代码
                print(f"\n{'#'*60}")
                print(f"# 最终生成的代码")
                print(f"{'#'*60}\n")
                print(final_code)
            
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
