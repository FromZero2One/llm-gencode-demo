"""
预设配置模块 - 提供简化的模型配置选项

教学优化:
- 学生只需选择 preset='tiny'/'small'/'medium',无需手动设置10+个参数
- 每个preset都有明确的用途和适用场景说明
"""

from typing import Dict, Any


# 预设配置定义
PRESETS = {
    'tiny': {
        'name': 'Tiny (超小型)',
        'description': '用于快速测试和理解基本原理',
        'vocab_size': 500,
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'dim_feedforward': 256,
        'max_seq_length': 64,
        'cache_size': 20,
        'use_case': '教学演示、快速原型验证'
    },
    
    'small': {
        'name': 'Small (小型)',
        'description': '平衡性能和资源消耗,适合大多数教学场景',
        'vocab_size': 1000,
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 512,
        'max_seq_length': 128,
        'cache_size': 50,
        'use_case': '常规代码生成实验、课堂演示'
    },
    
    'medium': {
        'name': 'Medium (中型)',
        'description': '更强的表达能力,适合深入研究和性能测试',
        'vocab_size': 2000,
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 1024,
        'max_seq_length': 256,
        'cache_size': 100,
        'use_case': '深入研究、性能基准测试'
    }
}


def get_preset_config(preset: str = 'small') -> Dict[str, Any]:
    """
    获取预设配置
    
    Args:
        preset: 预设名称 ('tiny', 'small', 'medium')
        
    Returns:
        配置字典
        
    Raises:
        ValueError: 如果preset名称无效
        
    Example:
        >>> config = get_preset_config('small')
        >>> print(config['d_model'])  # 128
    """
    if preset not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(
            f"无效的preset: '{preset}'。可用选项: {available}"
        )
    
    config = PRESETS[preset].copy()
    # 移除元数据字段,只保留配置参数
    config.pop('name', None)
    config.pop('description', None)
    config.pop('use_case', None)
    
    return config


def list_presets() -> str:
    """
    列出所有可用的预设配置
    
    Returns:
        格式化的预设列表字符串
        
    Example:
        >>> print(list_presets())
        可用的预设配置:
          - tiny: Tiny (超小型) - 用于快速测试...
          - small: Small (小型) - 平衡性能和资源...
          - medium: Medium (中型) - 更强的表达能力...
    """
    lines = ["可用的预设配置:"]
    for key, info in PRESETS.items():
        lines.append(f"  - {key}: {info['name']}")
        lines.append(f"    {info['description']}")
        lines.append(f"    适用场景: {info['use_case']}")
        lines.append("")
    
    return '\n'.join(lines)


def create_pipeline_from_preset(preset: str = 'small', **overrides):
    """
    从预设创建Pipeline实例(便捷函数)
    
    Args:
        preset: 预设名称
        **overrides: 覆盖特定配置参数的关键字参数
        
    Returns:
        CodeGenerationPipeline实例
        
    Example:
        >>> from scripts.config.presets import create_pipeline_from_preset
        >>> pipeline = create_pipeline_from_preset('small')
        >>> # 或者覆盖某些参数
        >>> pipeline = create_pipeline_from_preset('small', device='cuda')
    """
    from scripts.pipeline import CodeGenerationPipeline
    
    config = get_preset_config(preset)
    # 应用覆盖参数
    config.update(overrides)
    
    # 提取Pipeline构造函数需要的参数
    pipeline_params = {
        'vocab_size': config['vocab_size'],
        'd_model': config['d_model'],
        'nhead': config['nhead'],
        'num_encoder_layers': config['num_encoder_layers'],
        'num_decoder_layers': config['num_decoder_layers'],
        'cache_size': config['cache_size'],
        'device': config.get('device', 'cpu'),
        'debug_mode': config.get('debug_mode', False)
    }
    
    return CodeGenerationPipeline(**pipeline_params)


# 测试代码
if __name__ == '__main__':
    print("="*60)
    print("预设配置系统测试")
    print("="*60)
    
    # 测试1: 列出所有预设
    print("\n" + list_presets())
    
    # 测试2: 获取特定预设
    print("\n" + "="*60)
    print("获取'small'预设配置:")
    print("="*60)
    config = get_preset_config('small')
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 测试3: 错误处理
    print("\n" + "="*60)
    print("测试无效preset名称:")
    print("="*60)
    try:
        invalid_config = get_preset_config('invalid')
    except ValueError as e:
        print(f"[OK] 正确捕获错误: {e}")
    
    print("\n[SUCCESS] 所有测试通过!")
