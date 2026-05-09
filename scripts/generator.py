"""
代码生成器模块 - 实现各种采样策略和代码生成逻辑
"""

import sys
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from transformer import TransformerModel
from tokenizer import SimpleTokenizer
from logger import logging_context


class SamplingStrategy:
    """采样策略基类"""
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GreedySampling(SamplingStrategy):
    """
    贪婪采样：总是选择概率最高的token
    
    优点：确定性高，生成质量稳定
    缺点：缺乏多样性，可能重复
    """
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, vocab_size]
            
        Returns:
            token_ids: [batch_size]
        """
        return torch.argmax(logits, dim=-1)


class TemperatureSampling(SamplingStrategy):
    """
    温度采样：通过温度参数控制随机性
    
    temperature < 1: 更确定性（集中在高概率token）
    temperature = 1: 原始分布
    temperature > 1: 更随机（均匀分布）
    """
    
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        print(f"[Sampling] 使用Temperature采样, temperature={temperature}")
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # 应用温度缩放
        scaled_logits = logits / self.temperature
        
        # 转换为概率分布
        probs = F.softmax(scaled_logits, dim=-1)
        
        # 从分布中采样
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopKSampling(SamplingStrategy):
    """
    Top-K采样：只从概率最高的K个token中采样
    
    避免低概率的荒谬token，同时保持一定多样性
    """
    
    def __init__(self, top_k: int = 50, temperature: float = 0.7):
        self.top_k = top_k
        self.temperature = temperature
        print(f"[Sampling] 使用Top-K采样, top_k={top_k}, temperature={temperature}")
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = logits.shape
        
        # 应用温度
        scaled_logits = logits / self.temperature
        
        # 获取top-k
        top_k = min(self.top_k, vocab_size)
        top_values, top_indices = torch.topk(scaled_logits, top_k, dim=-1)
        
        # 创建屏蔽后的logits（非top-k的位置设为-inf）
        filtered_logits = torch.full_like(scaled_logits, float('-inf'))
        filtered_logits.scatter_(1, top_indices, top_values)
        
        # 转换为概率并采样
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopPSampling(SamplingStrategy):
    """
    Nucleus Sampling (Top-P)：从累积概率达到P的最小token集合中采样
    
    动态调整候选token数量，比Top-K更灵活
    """
    
    def __init__(self, top_p: float = 0.9, temperature: float = 0.7):
        self.top_p = top_p
        self.temperature = temperature
        print(f"[Sampling] 使用Top-P采样, top_p={top_p}, temperature={temperature}")
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = logits.shape
        
        # 应用温度
        scaled_logits = logits / self.temperature
        
        # 转换为概率并排序
        probs = F.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过top_p的位置
        sorted_indices_to_remove = cumulative_probs > self.top_p
        
        # 保留第一个超过阈值的token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 屏蔽低概率token
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
        
        # 采样
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class CodeGenerator:
    """
    代码生成器
    
    核心流程：
    1. 接收prompt
    2. 逐步生成token
    3. 应用采样策略
    4. 直到生成EOS或达到最大长度
    """
    
    def __init__(self, model: TransformerModel, tokenizer: SimpleTokenizer,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # 设置为评估模式
        
        # 统计信息
        self.generation_stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'avg_generation_length': 0
        }
        
        print(f"[CodeGenerator] 初始化完成")
        print(f"  - Device: {device}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def generate(self, prompt: str, max_length: int = 100,
                 strategy: Optional[SamplingStrategy] = None,
                 verbose: bool = True) -> Dict:
        """
        生成代码的主函数
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            strategy: 采样策略（默认使用TemperatureSampling）
            verbose: 是否打印详细信息
            
        Returns:
            dict包含生成的代码和统计信息
        """
        if strategy is None:
            strategy = TemperatureSampling(temperature=0.7)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[CodeGenerator] 开始代码生成")
            print(f"{'='*60}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Max length: {max_length}")
        
        # 1. Tokenize prompt (use shorter max_length to leave room for generation)
        prompt_tokens, prompt_mask = self.tokenizer.encode(prompt, max_length=32)
        prompt_tensor = torch.tensor([prompt_tokens]).to(self.device)
        prompt_mask_tensor = torch.tensor([prompt_mask]).unsqueeze(1).unsqueeze(2).to(self.device)
        
        if verbose:
            print(f"\n[CodeGenerator] Prompt tokenized: {len(prompt_tokens)} tokens")
        
        # 2. 清除之前的缓存
        self.model.clear_cache()
        
        # 3. 逐步生成
        generated_tokens = []
        # Start with just the BOS token for decoder input
        bos_tensor = torch.tensor([[self.tokenizer.vocab[self.tokenizer.BOS_TOKEN]]]).to(self.device)
        current_sequence = bos_tensor
        
        generation_details = []
        
        for step in range(max_length):
            with torch.no_grad():
                # 前向传播获取logits
                logits = self.model.generate_step(
                    prompt_tensor, 
                    current_sequence,
                    prompt_mask_tensor
                )
                
                # 采样下一个token
                next_token = strategy.sample(logits)
                
                # 记录生成详情
                if verbose and step < 10:  # 只记录前10步的详细信息
                    probs = F.softmax(logits, dim=-1)
                    top5_probs, top5_indices = torch.topk(probs, 5)
                    
                    detail = {
                        'step': step,
                        'token_id': next_token.item(),
                        'token': self.tokenizer.reverse_vocab.get(next_token.item(), '<UNK>'),
                        'top5_predictions': [
                            {
                                'token_id': idx.item(),
                                'token': self.tokenizer.reverse_vocab.get(idx.item(), '<UNK>'),
                                'probability': prob.item()
                            }
                            for idx, prob in zip(top5_indices[0], top5_probs[0])
                        ]
                    }
                    generation_details.append(detail)
                    
                    print(f"\n  Step {step}: Generated '{detail['token']}' (ID: {detail['token_id']})")
                    print(f"    Top-5 predictions:")
                    for pred in detail['top5_predictions']:
                        marker = " <-- selected" if pred['token_id'] == next_token.item() else ""
                        print(f"      - '{pred['token']}' (prob: {pred['probability']:.4f}){marker}")
                
                # 检查是否结束
                if next_token.item() == self.tokenizer.EOS_TOKEN:
                    if verbose:
                        print(f"\n[CodeGenerator] 遇到EOS token，生成结束")
                    break
                
                # 添加到生成序列
                generated_tokens.append(next_token.item())
                
                # 更新当前序列
                current_sequence = torch.cat(
                    [current_sequence, next_token.unsqueeze(0)], 
                    dim=1
                )
        
        # 4. 解码
        generated_text = self.tokenizer.decode(generated_tokens)
        
        # 5. 更新统计
        self.generation_stats['total_generations'] += 1
        self.generation_stats['total_tokens_generated'] += len(generated_tokens)
        self.generation_stats['avg_generation_length'] = (
            self.generation_stats['total_tokens_generated'] / 
            self.generation_stats['total_generations']
        )
        
        result = {
            'generated_code': generated_text,
            'token_count': len(generated_tokens),
            'generation_details': generation_details,
            'stats': self.generation_stats.copy()
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[CodeGenerator] 生成完成")
            print(f"{'='*60}")
            print(f"生成的Token数: {len(generated_tokens)}")
            print(f"\n生成的代码:\n{generated_text}")
            print(f"\n平均生成长度: {result['stats']['avg_generation_length']:.1f} tokens")
        
        return result
    
    def generate_multiple(self, prompt: str, num_samples: int = 3,
                         max_length: int = 100,
                         strategy: Optional[SamplingStrategy] = None) -> List[str]:
        """
        生成多个不同的样本（用于展示多样性）
        
        Args:
            prompt: 输入提示
            num_samples: 样本数量
            max_length: 最大长度
            strategy: 采样策略
            
        Returns:
            生成的代码列表
        """
        samples = []
        
        print(f"\n{'='*60}")
        print(f"[CodeGenerator] 生成 {num_samples} 个样本")
        print(f"{'='*60}\n")
        
        for i in range(num_samples):
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            result = self.generate(prompt, max_length, strategy, verbose=False)
            samples.append(result['generated_code'])
            print(f"Generated {result['token_count']} tokens")
        
        return samples
    
    def get_generation_report(self) -> Dict:
        """获取生成报告"""
        return {
            'stats': self.generation_stats,
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device)
            }
        }


# 测试代码
if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        print("="*60)
        print("测试代码生成器")
        print("="*60)
        
        # 创建模型和tokenizer
        vocab_size = 1000
        d_model = 128
        
        tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        # 创建生成器
        generator = CodeGenerator(model, tokenizer, device='cpu')
    
    # 测试不同的采样策略
    strategies = [
        ("Greedy", GreedySampling()),
        ("Temperature (0.5)", TemperatureSampling(temperature=0.5)),
        ("Temperature (1.0)", TemperatureSampling(temperature=1.0)),
        ("Top-K (k=20)", TopKSampling(top_k=20, temperature=0.7)),
        ("Top-P (p=0.9)", TopPSampling(top_p=0.9, temperature=0.7)),
    ]
    
    test_prompt = "public class UserService"
    
    for name, strategy in strategies:
        print(f"\n{'='*60}")
        print(f"测试采样策略: {name}")
        print(f"{'='*60}")
        
        try:
            result = generator.generate(
                test_prompt,
                max_length=30,
                strategy=strategy,
                verbose=True
            )
        except Exception as e:
            print(f"Error: {e}")
