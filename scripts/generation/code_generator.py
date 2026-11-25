"""
代码生成器模块 - 简化的代码生成和后处理逻辑

教学优化:
- 移除复杂的SamplingStrategy抽象层,内嵌4种采样方法
  (greedy / temperature / top_k / top_p, 通过 strategy 参数选择)
- 内嵌后处理功能,减少类层次
- 简化接口: generate(prompt, strategy='temperature', ...)
- 架构感知: Decoder-Only 模型时 prompt 拼进生成序列(自回归续写)
"""

import sys
import re
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from scripts.core.transformer import TransformerModel
from scripts.core.tokenizer import SimpleTokenizer
from scripts.utils.logger import logging_context


class SimplePostProcessor:
    """简化的后处理器 - 只做基础的格式化和import管理"""
    
    def __init__(self):
        """初始化常见 Java 类 → import 语句的映射表（后处理自动补 import 用）"""
        # 常见类和对应的import
        self.import_map = {
            'List': 'java.util.List',
            'ArrayList': 'java.util.ArrayList',
            'Map': 'java.util.Map',
            'HashMap': 'java.util.HashMap',
            'Override': 'java.lang.Override',
            'Service': 'org.springframework.stereotype.Service',
            'Autowired': 'org.springframework.beans.factory.annotation.Autowired',
        }
    
    def simple_format(self, code: str) -> str:
        """基础格式化:调整缩进、清理空行"""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # 调整缩进
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            formatted_lines.append('    ' * indent_level + stripped)
            
            if stripped.endswith('{'):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
    
    def add_imports(self, code: str) -> str:
        """添加必要的import语句"""
        used_classes = set()
        for class_name in self.import_map.keys():
            if re.search(r'\b' + re.escape(class_name) + r'\b', code):
                used_classes.add(class_name)
        
        if not used_classes:
            return code
        
        imports = [f'import {self.import_map[c]};' for c in sorted(used_classes)]
        return '\n'.join(imports) + '\n\n' + code
    
    def process(self, code: str) -> str:
        """完整的后处理流程"""
        formatted = self.simple_format(code)
        with_imports = self.add_imports(formatted)
        return with_imports


class CodeGenerator:
    """
    简化版代码生成器 (Code Generator)
    
    ════════════════════════════════════════════════════════════
    📚 核心概念：Auto-regressive（自回归）生成
    ════════════════════════════════════════════════════════════
    
    【什么是Auto-regressive？】
    Auto-regressive = 逐步生成，每一步都基于之前生成的内容
    
    类比：就像人类写代码，每写一个token都要参考已经写好的部分。
    
    【生成流程示例】
    假设要生成: "public class UserService {}"
    
    Step 0: [BOS]                                    → 预测 "public"
    Step 1: [BOS, public]                            → 预测 "class"
    Step 2: [BOS, public, class]                     → 预测 "UserService"
    Step 3: [BOS, public, class, UserService]        → 预测 "{"
    Step 4: [BOS, public, class, UserService, {]     → 预测 "}"
    Step 5: [BOS, public, class, UserService, {, }]  → 预测 [EOS]
    
    【关键技术点】
    1. **Teacher Forcing** (训练时):
       - 输入完整的target序列，但每次只能看到前面的部分
       - 例如：target="ABC"，训练时会输入[A], [A,B], [A,B,C]
    
    2. **Greedy Decoding** (temperature=0):
       - 每次都选择概率最高的token
       - 优点：结果稳定可复现
       - 缺点：可能陷入重复模式
    
    3. **Temperature Sampling** (temperature>0):
       - 根据概率分布随机采样
       - temperature越高，随机性越强
       - 常见值：0.7(平衡), 0.9(创意), 1.2(高随机)
    
    【与真实模型的对比】
    ┌─────────────────────────────────────────────────────┐
    │  GPT-3/4:    Auto-regressive + KV Cache             │
    │  Llama-2:    Auto-regressive + Attention Mask       │
    │  Claude:     Auto-regressive + Constitutional AI    │
    │  本项目:     Auto-regressive + Temperature Sample   │
    └─────────────────────────────────────────────────────┘
    
    ════════════════════════════════════════════════════════════
    🎯 核心改进
    ════════════════════════════════════════════════════════════
    
    1. 移除SamplingStrategy抽象层,直接使用temperature参数
       - 学生可以直接理解temperature的作用,无需学习策略模式
       - 接口更简洁: generate(prompt, temperature=0.7)
    
    2. 内嵌简单的后处理功能
       - 减少类层次,代码行数减少约40%
       - 保留核心的auto-regressive生成逻辑
    
    3. 教学价值提升
       - 专注于理解生成过程,而非设计模式
       - 每个步骤都有详细的日志输出
    """
    
    def __init__(self, model: TransformerModel, tokenizer: SimpleTokenizer,
                 device: str = 'cpu', enable_post_process: bool = True):
        """
        Args:
            model: Transformer模型
            tokenizer: Tokenizer
            device: 计算设备
            enable_post_process: 是否启用后处理(默认True)
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.enable_post_process = enable_post_process
        self.post_processor = SimplePostProcessor() if enable_post_process else None
        self.kv_cache_manager = None  # KV Cache 管理器，跨步复用
        
        # 设置为评估模式
        self.model.eval()
        
        # 统计信息
        self.generation_stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'avg_generation_length': 0
        }
        
        print(f"[CodeGenerator] 初始化完成")
        print(f"  - Device: {device}")
        print(f"  - Post-processing: {'enabled' if enable_post_process else 'disabled'}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 0.7, verbose: bool = True,
                 strategy: str = 'temperature',
                 top_k: int = 50, top_p: float = 0.9) -> Dict:
        """
        生成代码的主函数 - 简化接口
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数(0.5-1.5),控制随机性
            verbose: 是否打印详细信息
            strategy: 采样策略,可选:
                      - 'greedy': 贪婪解码,每步选概率最高的token(确定性)
                      - 'temperature': 温度采样(默认),按概率分布随机采样
                      - 'top_k': 先保留概率最高的top_k个token,再采样
                      - 'top_p': 保留概率累积达到top_p的最小token集合(nucleus采样),再采样
            top_k: strategy='top_k'时生效,保留的候选token数
            top_p: strategy='top_p'时生效,累积概率阈值(0-1)
            
        Returns:
            dict包含生成的代码和统计信息
            
        Example:
            >>> generator.generate("public class UserService", temperature=0.7)
            {'code': '...', 'token_count': 50}
            >>> generator.generate("public class UserService", strategy='top_k', top_k=10)
            {'code': '...', 'token_count': 50}
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[CodeGenerator] 开始代码生成")
            print(f"{'='*60}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Max length: {max_length}, Temperature: {temperature}")
            print(f"Strategy: {strategy}" + 
                  (f" (top_k={top_k})" if strategy == 'top_k' else "") +
                  (f" (top_p={top_p})" if strategy == 'top_p' else ""))
        
        # 1. Tokenize prompt (使用较短的max_length为生成留出空间)
        prompt_tokens, prompt_mask = self.tokenizer.encode(prompt, max_length=32)
        prompt_tensor = torch.tensor([prompt_tokens]).to(self.device)
        prompt_mask_tensor = torch.tensor([prompt_mask]).unsqueeze(1).unsqueeze(2).to(self.device)
        
        # 架构感知：
        # - Encoder-Decoder: prompt 喂给 Encoder（src 参数），生成序列从 BOS 起步
        # - Decoder-Only:    没有 Encoder，prompt 直接拼进生成序列（自回归续写范式）
        is_decoder_only = getattr(self.model, 'architecture', 'encdec') == 'decoder'
        
        if verbose:
            print(f"\n[CodeGenerator] Prompt tokenized: {len(prompt_tokens)} tokens")
            print(f"[CodeGenerator] 架构: {'Decoder-Only' if is_decoder_only else 'Encoder-Decoder'}")
        
        # 2. 清除之前的缓存
        self.model.clear_cache()
        
        # 3. 逐步生成(auto-regressive)
        generated_tokens = []
        # 从BOS token开始
        bos_tensor = torch.tensor([[self.tokenizer.vocab[self.tokenizer.BOS_TOKEN]]]).to(self.device)
        if is_decoder_only:
            # Decoder-Only: 初始序列 = [BOS] + prompt tokens，模型续写后半部分
            current_sequence = torch.cat([bos_tensor, prompt_tensor], dim=1)
        else:
            current_sequence = bos_tensor
        
        generation_details = []
        
        for step in range(max_length):
            with torch.no_grad():
                # 前向传播获取logits
                logits, self.kv_cache_manager = self.model.generate_step(
                    prompt_tensor, 
                    current_sequence,
                    prompt_mask_tensor,
                    kv_cache_manager=self.kv_cache_manager
                )
                
                # 按策略采样(内嵌逻辑,无需策略对象)
                next_token = self._sample(
                    logits, strategy=strategy,
                    temperature=temperature, top_k=top_k, top_p=top_p
                )
                
                # 记录生成详情(只记录前10步)
                if verbose and step < 10:
                    probs = F.softmax(logits / temperature, dim=-1)
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
                
                # 检查是否遇到EOS
                if next_token.item() == self.tokenizer.EOS_TOKEN:
                    if verbose:
                        print(f"\n[CodeGenerator] 遇到EOS token,生成结束")
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
        
        # 5. 后处理(可选)
        if self.enable_post_process and self.post_processor:
            generated_text = self.post_processor.process(generated_text)
            if verbose:
                print(f"\n[CodeGenerator] 已应用后处理(格式化+import)")
        
        # 6. 更新统计
        self.generation_stats['total_generations'] += 1
        self.generation_stats['total_tokens_generated'] += len(generated_tokens)
        self.generation_stats['avg_generation_length'] = (
            self.generation_stats['total_tokens_generated'] / 
            self.generation_stats['total_generations']
        )
        
        result = {
            'code': generated_text,
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
    
    def _sample(self, logits: torch.Tensor, strategy: str = 'temperature',
                temperature: float = 0.7, top_k: int = 50,
                top_p: float = 0.9) -> torch.Tensor:
        """
        采样策略分发器 - 根据strategy选择对应的采样方法
        
        四种策略的确定性/多样性对比:
        ┌──────────────┬────────┬────────────────────────────────┐
        │ 策略          │ 确定性  │ 特点                           │
        ├──────────────┼────────┼────────────────────────────────┤
        │ greedy       │ ★★★★★  │ 每步取最大概率,结果完全可复现    │
        │ temperature  │ ★★★    │ 全词表按温度缩放后采样          │
        │ top_k        │ ★★★★   │ 只在概率最高的k个token中采样    │
        │ top_p        │ ★★★★   │ 只在累积概率达p的最小集合中采样  │
        └──────────────┴────────┴────────────────────────────────┘
        
        Args:
            logits: [batch_size, vocab_size]
            strategy: 'greedy' | 'temperature' | 'top_k' | 'top_p'
            temperature: 温度参数
            top_k: top_k采样的候选数
            top_p: top_p采样的累积概率阈值
            
        Returns:
            token_ids: [batch_size]
        """
        if strategy == 'greedy':
            return self._greedy_sample(logits)
        elif strategy == 'temperature':
            return self._temperature_sample(logits, temperature)
        elif strategy == 'top_k':
            return self._top_k_sample(logits, top_k, temperature)
        elif strategy == 'top_p':
            return self._top_p_sample(logits, top_p, temperature)
        else:
            raise ValueError(
                f"未知采样策略: '{strategy}', "
                f"可选: 'greedy' | 'temperature' | 'top_k' | 'top_p'"
            )
    
    def _greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        贪婪采样 - 每步选择概率最高的token
        
        数学形式: next_token = argmax(logits)
        
        特点:
        - 结果完全确定(相同输入必得相同输出)
        - 缺点: 容易陷入重复循环(如 "public class class class ...")
        
        Args:
            logits: [batch_size, vocab_size]
            
        Returns:
            token_ids: [batch_size]
        """
        return torch.argmax(logits, dim=-1)
    
    def _top_k_sample(self, logits: torch.Tensor, top_k: int,
                      temperature: float = 1.0) -> torch.Tensor:
        """
        Top-K采样 - 只保留概率最高的k个token,重新归一化后采样
        
        数学形式:
          1. 取logits最大的k个值: S = TopK(logits, k)
          2. 其余位置设为 -inf:    logits[i] = -inf, i ∉ S
          3. softmax后采样:        p = softmax(logits / T), sample ~ p
        
        直觉: 排除了长尾低概率token(如拼写错误的标识符),
             又保留了头部候选之间的随机性。
        k=1 时退化为greedy;k=词表大小时退化为temperature采样。
        
        Args:
            logits: [batch_size, vocab_size]
            top_k: 保留的候选token数(>=1)
            temperature: 温度参数
            
        Returns:
            token_ids: [batch_size]
        """
        if top_k < 1:
            raise ValueError(f"top_k必须>=1, 当前: {top_k}")
        # top_k超过词表大小时等价于不过滤
        top_k = min(top_k, logits.size(-1))
        
        # 取第k大的值作为阈值,低于阈值的全部置为-inf
        kth_value = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        filtered = logits.masked_fill(logits < kth_value, float('-inf'))
        
        probs = F.softmax(filtered / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _top_p_sample(self, logits: torch.Tensor, top_p: float,
                      temperature: float = 1.0) -> torch.Tensor:
        """
        Top-P采样(Nucleus Sampling / 核采样) - 保留累积概率达到p的最小token集合
        
        数学形式:
          1. 按概率降序排列:      p_(1) >= p_(2) >= ...
          2. 找最小n使:           p_(1) + p_(2) + ... + p_(n) >= top_p
          3. 只在 {token_(1)...token_(n)} 中重新归一化后采样
        
        与Top-K的区别:
        - Top-K的候选数固定(可能太少也可能太多)
        - Top-P的候选数自适应: 分布尖锐(确定性强)时候选少,
          分布平坦(不确定性强)时候选多,更符合直觉
        
        Args:
            logits: [batch_size, vocab_size]
            top_p: 累积概率阈值,(0, 1]
            temperature: 温度参数
            
        Returns:
            token_ids: [batch_size]
        """
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p必须在(0, 1]范围内, 当前: {top_p}")
        
        # 1. 计算概率并降序排序
        probs = F.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 2. 计算累积概率,找到累积超过top_p的位置
        cumsum_probs = sorted_probs.cumsum(dim=-1)
        # 第一个使累积概率超过top_p的位置之后的token全部移除
        # (保留第一个超过阈值的token本身,即移除"累积到p之后"的token)
        remove_mask = cumsum_probs > top_p
        remove_mask[:, 1:] = remove_mask[:, :-1].clone()  # 右移一位,保留临界token
        remove_mask[:, 0] = False  # 概率最高的token永远保留
        
        # 3. 把被移除token的概率置0,还原到原始顺序
        sorted_probs[remove_mask] = 0
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
        
        # 4. 归一化后采样
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    
    def _temperature_sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Temperature采样(内嵌实现)
        
        Args:
            logits: [batch_size, vocab_size]
            temperature: 温度参数
            
        Returns:
            token_ids: [batch_size]
        """
        # 应用温度缩放
        scaled_logits = logits / temperature
        
        # 转换为概率分布
        probs = F.softmax(scaled_logits, dim=-1)
        
        # 从分布中采样
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def generate_multiple(self, prompt: str, num_samples: int = 3,
                         max_length: int = 100, temperature: float = 0.7) -> List[str]:
        """
        生成多个不同的样本(展示多样性)
        
        Args:
            prompt: 输入提示
            num_samples: 样本数量
            max_length: 最大长度
            temperature: 温度参数
            
        Returns:
            生成的代码列表
        """
        samples = []
        
        print(f"\n{'='*60}")
        print(f"[CodeGenerator] 生成 {num_samples} 个样本 (temperature={temperature})")
        print(f"{'='*60}\n")
        
        for i in range(num_samples):
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            result = self.generate(prompt, max_length, temperature, verbose=False)
            samples.append(result['code'])
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
        print("测试简化版代码生成器")
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
        
        # 创建生成器(启用后处理)
        generator = CodeGenerator(model, tokenizer, device='cpu', enable_post_process=True)
    
    # 测试不同temperature的效果
    test_prompt = "public class UserService"
    
    temperatures = [0.5, 0.7, 1.0]
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"测试Temperature: {temp}")
        print(f"{'='*60}")
        
        try:
            result = generator.generate(
                test_prompt,
                max_length=30,
                temperature=temp,
                verbose=True
            )
        except Exception as e:
            print(f"Error: {e}")
    
    # ==================== 采样策略对比演示 ====================
    print("\n" + "="*60)
    print("采样策略对比演示 (greedy / temperature / top_k / top_p)")
    print("="*60)
    
    strategies = [
        ('greedy',      {}),
        ('temperature', {'temperature': 0.8}),
        ('top_k',       {'top_k': 10}),
        ('top_p',       {'top_p': 0.9}),
    ]
    
    for strat, kwargs in strategies:
        print(f"\n--- 策略: {strat} {kwargs} ---")
        try:
            result = generator.generate(
                test_prompt,
                max_length=15,
                verbose=False,
                strategy=strat,
                **kwargs
            )
            print(f"  生成 {result['token_count']} tokens")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n提示: greedy结果完全确定; temperature随机性最强;")
    print("      top_k/top_p在'候选质量'与'随机性'之间取得平衡")
