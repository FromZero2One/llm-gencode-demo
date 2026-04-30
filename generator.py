"""
代码生成器模块 - 实现各种采样策略和代码生成逻辑
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from transformer import TransformerModel
from tokenizer import SimpleTokenizer


class ProbabilityAnalyzer:
    """
    概率分布分析器
    
    用于深入理解模型在每一步的预测不确定性、概率分布形态等
    帮助学习者理解为什么某些token被选中，以及temperature如何影响决策
    """
    
    @staticmethod
    def analyze_distribution(logits: torch.Tensor, selected_token_id: int) -> Dict:
        """
        分析概率分布的关键指标
        
        Args:
            logits: [batch_size, vocab_size] 模型的原始输出
            selected_token_id: 被选中的token ID
            
        Returns:
            dict包含各种概率分布指标
        """
        # 转换为概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 计算熵（不确定性）
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # 找到选中token的排名和概率
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 找出选中token在排序后的位置
        batch_size = logits.shape[0]
        ranks = []
        selected_probs = []
        
        for b in range(batch_size):
            rank_mask = (sorted_indices[b] == selected_token_id)
            rank = rank_mask.nonzero(as_tuple=True)[0][0].item()
            ranks.append(rank)
            selected_probs.append(sorted_probs[b, rank].item())
        
        # Top-k累积概率
        top5_cumulative = sorted_probs[:, :5].sum(dim=-1)
        top10_cumulative = sorted_probs[:, :10].sum(dim=-1)
        
        # 计算Gini系数（概率集中度）
        gini = ProbabilityAnalyzer._calculate_gini(probs)
        vocab_coverage = ProbabilityAnalyzer._vocab_coverage(probs, 0.9)
        
        return {
            'entropy': entropy.tolist() if batch_size > 1 else entropy.item(),
            'ranks': ranks if batch_size > 1 else ranks[0],
            'selected_probabilities': selected_probs,
            'top1_probabilities': sorted_probs[:, 0].tolist(),
            'top5_cumulative_probabilities': top5_cumulative.tolist(),
            'top10_cumulative_probabilities': top10_cumulative.tolist(),
            'gini_coefficients': gini.tolist() if batch_size > 1 else gini.item(),
            'vocab_coverage_90pct': vocab_coverage.tolist() if batch_size > 1 else vocab_coverage.item()
        }
    
    @staticmethod
    def _calculate_gini(probs: torch.Tensor) -> torch.Tensor:
        """
        计算Gini系数，衡量概率分布的不平等程度
        Gini接近1: 概率集中在少数token（确定性高）
        Gini接近0: 概率均匀分布（随机性高）
        """
        batch_size = probs.shape[0]
        ginis = []
        
        for b in range(batch_size):
            sorted_p = torch.sort(probs[b], descending=True)[0]
            n = sorted_p.shape[0]
            indices = torch.arange(1, n + 1, dtype=torch.float32, device=probs.device)
            gini = (2 * (sorted_p * indices).sum() / (n * sorted_p.sum())) - ((n + 1) / n)
            ginis.append(gini)
        
        return torch.tensor(ginis, device=probs.device)
    
    @staticmethod
    def _vocab_coverage(probs: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
        """
        计算累积概率达到threshold所需的最少token数量
        值越小说明概率越集中
        """
        batch_size = probs.shape[0]
        coverages = []
        
        for b in range(batch_size):
            sorted_probs = torch.sort(probs[b], descending=True)[0]
            cumulative = torch.cumsum(sorted_probs, dim=0)
            coverage = (cumulative >= threshold).nonzero(as_tuple=True)[0]
            if len(coverage) > 0:
                coverages.append(coverage[0].item() + 1)
            else:
                coverages.append(len(sorted_probs))
        
        return torch.tensor(coverages, device=probs.device)
    
    @staticmethod
    def format_analysis(analysis: Dict, step: int, selected_token: str) -> str:
        """格式化分析结果为可读字符串"""
        # 处理单值和列表的情况
        entropy = analysis['entropy'] if isinstance(analysis['entropy'], float) else analysis['entropy'][0]
        rank = analysis['ranks'] if isinstance(analysis['ranks'], int) else analysis['ranks'][0]
        sel_prob = analysis['selected_probabilities'][0]
        top1_prob = analysis['top1_probabilities'][0]
        top5_cum = analysis['top5_cumulative_probabilities'][0]
        gini = analysis['gini_coefficients'] if isinstance(analysis['gini_coefficients'], float) else analysis['gini_coefficients'][0]
        vocab_cov = analysis['vocab_coverage_90pct'] if isinstance(analysis['vocab_coverage_90pct'], int) else analysis['vocab_coverage_90pct'][0]
        
        lines = [
            f"\n  [Probability Analysis] Step {step}: Selected '{selected_token}'",
            f"    Entropy (uncertainty): {entropy:.4f}",
            f"    Token rank: #{rank}",
            f"    Selected probability: {sel_prob:.4f}",
            f"    Top-1 probability: {top1_prob:.4f}",
            f"    Top-5 cumulative: {top5_cum:.4f}",
            f"    Gini coefficient: {gini:.4f}",
            f"    Tokens for 90% prob: {vocab_cov}"
        ]
        return '\n'.join(lines)


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
                 verbose: bool = True,
                 enable_probability_analysis: bool = False) -> Dict:
        """
        生成代码的主函数
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            strategy: 采样策略（默认使用TemperatureSampling）
            verbose: 是否打印详细信息
            enable_probability_analysis: 是否启用概率分布分析
            
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
        probability_analyses = []
        
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
                    
                    # 概率分布分析
                    if enable_probability_analysis:
                        analysis = ProbabilityAnalyzer.analyze_distribution(logits, next_token.item())
                        probability_analyses.append(analysis)
                        
                        token_name = detail['token']
                        analysis_str = ProbabilityAnalyzer.format_analysis(analysis, step, token_name)
                        print(analysis_str)
                
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
            'probability_analyses': probability_analyses,
            'stats': self.generation_stats.copy()
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[CodeGenerator] 生成完成")
            print(f"{'='*60}")
            print(f"生成的Token数: {len(generated_tokens)}")
            print(f"\n生成的代码:\n{generated_text}")
            print(f"\n平均生成长度: {result['stats']['avg_generation_length']:.1f} tokens")
            
            if enable_probability_analysis and probability_analyses:
                print(f"\n{'='*60}")
                print(f"[Probability Analysis Summary]")
                print(f"{'='*60}")
                avg_entropy = sum(a['entropy'] for a in probability_analyses) / len(probability_analyses)
                avg_rank = sum(a['ranks'] for a in probability_analyses) / len(probability_analyses)
                print(f"Average entropy: {avg_entropy:.4f}")
                print(f"Average token rank: {avg_rank:.1f}")
                print(f"(Lower entropy = more confident, Lower rank = higher probability)")
        
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
