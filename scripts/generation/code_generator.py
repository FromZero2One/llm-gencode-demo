"""
代码生成器模块 - 简化的代码生成和后处理逻辑

教学优化:
- 移除复杂的SamplingStrategy抽象层,只保留Temperature采样
- 内嵌后处理功能,减少类层次
- 简化接口: generate(prompt, temperature=0.7)
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
                 temperature: float = 0.7, verbose: bool = True) -> Dict:
        """
        生成代码的主函数 - 简化接口
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数(0.5-1.5),控制随机性
            verbose: 是否打印详细信息
            
        Returns:
            dict包含生成的代码和统计信息
            
        Example:
            >>> generator.generate("public class UserService", temperature=0.7)
            {'code': '...', 'token_count': 50}
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[CodeGenerator] 开始代码生成")
            print(f"{'='*60}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Max length: {max_length}, Temperature: {temperature}")
        
        # 1. Tokenize prompt (使用较短的max_length为生成留出空间)
        prompt_tokens, prompt_mask = self.tokenizer.encode(prompt, max_length=32)
        prompt_tensor = torch.tensor([prompt_tokens]).to(self.device)
        prompt_mask_tensor = torch.tensor([prompt_mask]).unsqueeze(1).unsqueeze(2).to(self.device)
        
        if verbose:
            print(f"\n[CodeGenerator] Prompt tokenized: {len(prompt_tokens)} tokens")
        
        # 2. 清除之前的缓存
        self.model.clear_cache()
        
        # 3. 逐步生成(auto-regressive)
        generated_tokens = []
        # 从BOS token开始
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
                
                # Temperature采样(内嵌逻辑,无需策略对象)
                next_token = self._temperature_sample(logits, temperature)
                
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
