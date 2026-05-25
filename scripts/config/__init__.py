"""
[Config] 配置模块包
==================
包含预设配置和配置管理工具
"""

from scripts.config.presets import get_preset_config, list_presets, create_pipeline_from_preset

__all__ = [
    'get_preset_config',
    'list_presets',
    'create_pipeline_from_preset'
]
