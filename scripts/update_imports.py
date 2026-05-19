"""
批量更新导入路径的脚本
"""
import os
import re

# 定义导入映射规则
import_map = {
    # 核心模块
    r'^from tokenizer import': 'from .tokenizer import',
    r'^from attention import': 'from .attention import',
    r'^from transformer import': 'from .transformer import',
    
    # 生成模块
    r'^from generator import': 'from .generator import',
    r'^from postprocessor import': 'from .postprocessor import',
    
    # 优化模块
    r'^from cache import': 'from .cache import',
    r'^from kv_cache import': 'from .kv_cache import',
    
    # 训练模块
    r'^from trainer import': 'from .trainer import',
    
    # 工具模块
    r'^from logger import': 'from ..utils.logger import',
    r'^from visualizer import': 'from ..utils.visualizer import',
}

def update_imports_in_file(file_path):
    """更新单个文件的导入语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for pattern, replacement in import_map.items():
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """主函数"""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 需要更新的文件列表
    files_to_update = [
        os.path.join(scripts_dir, 'core', 'tokenizer.py'),
        os.path.join(scripts_dir, 'core', 'attention.py'),
        os.path.join(scripts_dir, 'core', 'transformer.py'),
        os.path.join(scripts_dir, 'generation', 'generator.py'),
        os.path.join(scripts_dir, 'generation', 'postprocessor.py'),
        os.path.join(scripts_dir, 'optimization', 'cache.py'),
        os.path.join(scripts_dir, 'optimization', 'kv_cache.py'),
        os.path.join(scripts_dir, 'training', 'trainer.py'),
        os.path.join(scripts_dir, 'utils', 'visualizer.py'),
        os.path.join(scripts_dir, 'pipeline.py'),
        os.path.join(scripts_dir, 'main.py'),
        os.path.join(scripts_dir, 'tests', 'test_all.py'),
        os.path.join(scripts_dir, 'tests', 'verify_math.py'),
    ]
    
    updated_count = 0
    for file_path in files_to_update:
        if os.path.exists(file_path):
            if update_imports_in_file(file_path):
                print(f"✓ Updated: {os.path.relpath(file_path, scripts_dir)}")
                updated_count += 1
            else:
                print(f"- No changes: {os.path.relpath(file_path, scripts_dir)}")
        else:
            print(f"✗ Not found: {os.path.relpath(file_path, scripts_dir)}")
    
    print(f"\n总计更新 {updated_count} 个文件")

if __name__ == '__main__':
    main()
