"""
训练系统模块包
包含训练器、损失函数和优化器配置
"""

from .trainer import Trainer, TrainingConfig

__all__ = [
    'Trainer',
    'TrainingConfig'
]
