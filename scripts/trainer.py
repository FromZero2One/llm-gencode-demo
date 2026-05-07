"""
训练器模块 - 实现模型的训练功能

本模块提供完整的训练基础设施，包括：
1. 损失函数计算（CrossEntropy）
2. 优化器（AdamW）
3. 学习率调度器
4. 训练循环
5. 验证和评估
6. Checkpoint管理

教学要点：
- 理解模型如何通过梯度下降学习
- 掌握训练过程中的关键组件
- 学会监控训练进度和调试
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossEntropyLoss(nn.Module):
    """
    交叉熵损失函数
    
    用于语言建模任务，衡量预测分布与真实分布的差异。
    
    数学公式：
    Loss = -Σ log(P(target_token))
    
    在Transformer中：
    - logits: (batch_size, seq_len, vocab_size) - 模型对每个位置的预测
    - targets: (batch_size, seq_len) - 真实的token ID
    
    工作原理：
    1. 对logits应用softmax得到概率分布
    2. 取目标token位置的概率的对数
    3. 取负值并求平均
    
    Example:
        >>> loss_fn = CrossEntropyLoss()
        >>> logits = torch.randn(2, 10, 1000)  # batch=2, seq_len=10, vocab=1000
        >>> targets = torch.randint(0, 1000, (2, 10))
        >>> loss = loss_fn(logits, targets)
        >>> print(f"Loss: {loss.item():.4f}")
    """
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        """
        初始化交叉熵损失
        
        Args:
            ignore_index (int): 要忽略的target索引，默认为-100
                               常用于padding token，不参与损失计算
            
            label_smoothing (float): 标签平滑系数，范围[0, 1)
                                    0表示不使用平滑（标准交叉熵）
                                    0.1表示将10%的概率分配给非目标类别
                                    有助于防止过拟合和提高泛化能力
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算交叉熵损失
        
        Args:
            logits (torch.Tensor): 模型输出的未归一化分数
                                  形状: (batch_size, seq_len, vocab_size)
            
            targets (torch.Tensor): 真实的目标token ID
                                   形状: (batch_size, seq_len)
        
        Returns:
            torch.Tensor: 标量损失值
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 步骤1: 展平以便计算
        # 将(batch, seq_len, vocab)展平为(batch*seq_len, vocab)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # 步骤2: 应用标签平滑（如果启用）
        if self.label_smoothing > 0:
            # 标准交叉熵使用one-hot编码：目标位置为1，其他为0
            # 标签平滑后：目标位置为(1-smoothing)，其他位置均匀分配smoothing
            
            # 计算平滑后的目标分布
            smooth_targets = torch.full_like(logits_flat, self.label_smoothing / vocab_size)
            smooth_targets.scatter_(1, targets_flat.unsqueeze(1), 1 - self.label_smoothing + self.label_smoothing / vocab_size)
            
            # 计算log softmax
            log_probs = torch.log_softmax(logits_flat, dim=-1)
            
            # 计算KL散度（等价于交叉熵）
            loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            # 标准交叉熵
            log_probs = torch.log_softmax(logits_flat, dim=-1)
            
            # 收集目标位置的概率
            # gather操作：对于每个样本，取出targets对应位置的log_prob
            loss = -log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        
        # 步骤3: 处理ignore_index
        if self.ignore_index >= 0:
            mask = (targets_flat != self.ignore_index).float()
            loss = loss * mask
        
        # 步骤4: 求平均（只考虑有效位置）
        loss = loss.sum() / max(mask.sum(), 1) if self.ignore_index >= 0 else loss.mean()
        
        return loss


class AdamW(optim.Optimizer):
    """
    AdamW优化器
    
    Adam的改进版本，正确实现了权重衰减（Weight Decay）。
    
    为什么需要AdamW而不是Adam？
    - Adam: L2正则化和自适应学习率耦合，效果不佳
    - AdamW: 解耦权重衰减，先更新参数再衰减，效果更好
    
    数学公式：
    m_t = β1 * m_{t-1} + (1-β1) * g_t          # 一阶矩估计
    v_t = β2 * v_{t-1} + (1-β2) * g_t^2        # 二阶矩估计
    m_hat = m_t / (1-β1^t)                      # 偏差修正
    v_hat = v_t / (1-β2^t)                      # 偏差修正
    θ_t = θ_{t-1} - lr * (m_hat / (sqrt(v_hat) + ε) + λ * θ_{t-1})
    
    其中λ是权重衰减系数
    
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01):
        """
        初始化AdamW优化器
        
        Args:
            params: 模型参数迭代器
            lr (float): 学习率，控制每次更新的步长
            betas (tuple): (β1, β2)，动量和方差的衰减系数
                          β1控制一阶矩的历史权重（通常0.9）
                          β2控制二阶矩的历史权重（通常0.999）
            eps (float): 数值稳定性常数，防止除零
            weight_decay (float): 权重衰减系数（L2正则化强度）
                                 典型值：0.01或0.001
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """
        执行一步优化
        
        Args:
            closure (callable, optional): 重新计算损失并返回它的闭包
                                         主要用于某些高级优化算法
        
        Returns:
            None
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # 一阶矩
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # 二阶矩
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 更新一阶矩估计（梯度的指数移动平均）
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 更新二阶矩估计（梯度平方的指数移动平均）
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差修正（bias correction）
                # 初期矩估计偏向0，需要除以(1-β^t)进行修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算自适应学习率
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # 计算更新步长
                step_size = group['lr'] / bias_correction1
                
                # 更新参数：先应用自适应学习率，再应用权重衰减
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss


class WarmupLinearScheduler:
    """
    Warmup + 线性衰减学习率调度器
    
    Transformer训练中广泛使用的学习率策略。
    
    工作原理：
    1. Warmup阶段：学习率从0线性增加到最大值
       - 帮助模型稳定初始化
       - 避免初期大步长导致的不稳定
    
    2. 线性衰减阶段：学习率从最大值线性下降到0
       - 后期小步长精细调整
       - 帮助收敛到更好的局部最优
    
    数学公式：
    warmup阶段 (step < warmup_steps):
        lr = max_lr * (step / warmup_steps)
    
    衰减阶段 (step >= warmup_steps):
        lr = max_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    Example:
        >>> scheduler = WarmupLinearScheduler(
        ...     optimizer, 
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     max_lr=1e-4
        ... )
        >>> for step in range(total_steps):
        ...     train_step()
        ...     scheduler.step()
    """
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 max_lr: float = 1e-4, min_lr: float = 1e-6):
        """
        初始化学习率调度器
        
        Args:
            optimizer: PyTorch优化器实例
            warmup_steps (int): warmup阶段的步数
                               通常为总步数的10%-20%
            total_steps (int): 总训练步数
            max_lr (float): 最大学习率（warmup结束时的学习率）
            min_lr (float): 最小学习率（训练结束时的学习率）
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """
        更新学习率（每步调用）
        
        Returns:
            float: 当前学习率
        """
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup阶段：线性增加
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # 线性衰减阶段
            progress = (self.current_step - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps)
            lr = self.max_lr * (1 - progress) + self.min_lr * progress
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class TextDataset(Dataset):
    """
    文本数据集类
    
    将原始文本转换为训练所需的(input, target)对。
    
    工作原理：
    给定序列 [t1, t2, t3, t4, t5]
    - input:  [t1, t2, t3, t4]
    - target: [t2, t3, t4, t5]
    
    这样模型学习预测下一个token
    
    Example:
        >>> dataset = TextDataset(texts, tokenizer, max_len=128)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in dataloader:
        ...     inputs, targets = batch
    """
    
    def __init__(self, texts: List[str], tokenizer, max_len: int = 128):
        """
        初始化数据集
        
        Args:
            texts (List[str]): 原始文本列表
            tokenizer: Tokenizer实例，用于将文本转换为token IDs
            max_len (int): 最大序列长度，超过则截断
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 预处理所有文本
        logger.info(f"预处理 {len(texts)} 条文本...")
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) >= 2:  # 至少需要2个token才能形成input-target对
                self.examples.append(tokens[:max_len])
        
        logger.info(f"预处理完成，共 {len(self.examples)} 个样本")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            tuple: (input_ids, target_ids)
                  input_ids: 输入序列 [t1, t2, ..., t_{n-1}]
                  target_ids: 目标序列 [t2, t3, ..., t_n]
        """
        tokens = self.examples[idx]
        
        # 构建input和target
        input_ids = tokens[:-1]  # 去掉最后一个token
        target_ids = tokens[1:]  # 去掉第一个token
        
        # 转换为tensor
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        
        return input_tensor, target_tensor


class Trainer:
    """
    训练管理器
    
    封装完整的训练流程，包括：
    - 训练循环
    - 验证评估
    - 日志记录
    - Checkpoint保存/加载
    
    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataset=train_data,
        ...     val_dataset=val_data,
        ...     batch_size=32,
        ...     lr=1e-4,
        ...     epochs=10
        ... )
        >>> trainer.train()
    """
    
    def __init__(self, model, train_dataset: Dataset, val_dataset: Dataset = None,
                 batch_size: int = 32, lr: float = 1e-4, epochs: int = 10,
                 warmup_ratio: float = 0.1, weight_decay: float = 0.01,
                 save_dir: str = "checkpoints", device: str = None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_dataset (Dataset): 训练数据集
            val_dataset (Dataset, optional): 验证数据集
            batch_size (int): 批次大小
            lr (float): 最大学习率
            epochs (int): 训练轮数
            warmup_ratio (float): warmup占总步数的比例
            weight_decay (float): 权重衰减系数
            save_dir (str): checkpoint保存目录
            device (str, optional): 设备 ('cpu' 或 'cuda')
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Windows下建议设为0
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0
            )
        
        # 计算总步数
        steps_per_epoch = len(self.train_loader)
        self.total_steps = steps_per_epoch * epochs
        self.warmup_steps = int(self.total_steps * warmup_ratio)
        
        logger.info(f"训练配置:")
        logger.info(f"  - 总步数: {self.total_steps}")
        logger.info(f"  - Warmup步数: {self.warmup_steps}")
        logger.info(f"  - 每epoch步数: {steps_per_epoch}")
        
        # 创建优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 创建学习率调度器
        self.scheduler = WarmupLinearScheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            max_lr=lr
        )
        
        # 创建损失函数
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """
        训练一个epoch
        
        Args:
            epoch (int): 当前epoch编号
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # 将数据移到设备
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失
            loss = self.loss_fn(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率
            current_lr = self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"Epoch [{epoch+1}/{self.epochs}] | "
                    f"Batch [{batch_idx+1}/{len(self.train_loader)}] | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        在验证集上评估
        
        Returns:
            float: 平均验证损失
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float):
        """
        保存checkpoint
        
        Args:
            epoch (int): 当前epoch
            loss (float): 当前损失
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': {
                'current_step': self.scheduler.current_step
            },
            'loss': loss,
            'history': self.history
        }
        
        filepath = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载checkpoint
        
        Args:
            filepath (str): checkpoint文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_state_dict']['current_step']
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint已加载: {filepath}")
        logger.info(f"从epoch {checkpoint['epoch']}继续训练")
    
    def train(self):
        """
        开始训练
        
        Returns:
            dict: 训练历史记录
        """
        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(self.scheduler.get_lr())
            
            # 验证
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            logger.info(
                f"\n{'='*60}\n"
                f"Epoch [{epoch+1}/{self.epochs}] 完成\n"
                f"  训练损失: {train_loss:.4f}\n"
                f"  验证损失: {val_loss:.4f}\n"
                f"  当前学习率: {self.scheduler.get_lr():.6f}\n"
                f"{'='*60}\n"
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                logger.info(f"✓ 新的最佳模型! 验证损失: {val_loss:.4f}")
        
        logger.info("训练完成!")
        logger.info(f"最佳验证损失: {best_val_loss:.4f}")
        
        # 保存最终历史
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"训练历史已保存: {history_path}")
        
        return self.history
