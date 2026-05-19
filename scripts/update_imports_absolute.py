"""
批量更新导入路径 - 使用绝对导入
"""
import os
import re

def update_file_imports(file_path, module_type):
    """更新单个文件的导入语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 根据模块类型替换导入
    if module_type == 'core':
        # core 模块内部互相引用
        content = re.sub(r'from logger import', 'from scripts.utils.logger import', content)
        content = re.sub(r'from attention import', 'from scripts.core.attention import', content)
        content = re.sub(r'from transformer import', 'from scripts.core.transformer import', content)
        
    elif module_type == 'generation':
        content = re.sub(r'from logger import', 'from scripts.utils.logger import', content)
        content = re.sub(r'from transformer import', 'from scripts.core.transformer import', content)
        content = re.sub(r'from tokenizer import', 'from scripts.core.tokenizer import', content)
        
    elif module_type == 'optimization':
        content = re.sub(r'from logger import', 'from scripts.utils.logger import', content)
        
    elif module_type == 'training':
        content = re.sub(r'from logger import', 'from scripts.utils.logger import', content)
        
    elif module_type == 'utils':
        # utils 内部不互相引用，只引用外部
        pass
        
    elif module_type == 'integration':
        content = re.sub(r'from tokenizer import', 'from scripts.core.tokenizer import', content)
        content = re.sub(r'from transformer import', 'from scripts.core.transformer import', content)
        content = re.sub(r'from generator import', 'from scripts.generation.generator import', content)
        content = re.sub(r'from postprocessor import', 'from scripts.generation.postprocessor import', content)
        content = re.sub(r'from cache import', 'from scripts.optimization.cache import', content)
        content = re.sub(r'from logger import', 'from scripts.utils.logger import', content)
        content = re.sub(r'from visualizer import', 'from scripts.utils.visualizer import', content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_update = [
        # Core modules
        (os.path.join(scripts_dir, 'core', 'tokenizer.py'), 'core'),
        (os.path.join(scripts_dir, 'core', 'attention.py'), 'core'),
        (os.path.join(scripts_dir, 'core', 'transformer.py'), 'core'),
        
        # Generation modules
        (os.path.join(scripts_dir, 'generation', 'generator.py'), 'generation'),
        (os.path.join(scripts_dir, 'generation', 'postprocessor.py'), 'generation'),
        
        # Optimization modules
        (os.path.join(scripts_dir, 'optimization', 'cache.py'), 'optimization'),
        (os.path.join(scripts_dir, 'optimization', 'kv_cache.py'), 'optimization'),
        
        # Training modules
        (os.path.join(scripts_dir, 'training', 'trainer.py'), 'training'),
        
        # Utils modules
        (os.path.join(scripts_dir, 'utils', 'visualizer.py'), 'utils'),
        
        # Integration modules
        (os.path.join(scripts_dir, 'pipeline.py'), 'integration'),
        (os.path.join(scripts_dir, 'main.py'), 'integration'),
        
        # Test files
        (os.path.join(scripts_dir, 'tests', 'test_all.py'), 'test'),
        (os.path.join(scripts_dir, 'tests', 'verify_math.py'), 'test'),
    ]
    
    updated_count = 0
    for file_path, module_type in files_to_update:
        if os.path.exists(file_path):
            if module_type == 'test':
                # 测试文件特殊处理
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original = content
                content = re.sub(r'from tokenizer import', 'from scripts.core.tokenizer import', content)
                content = re.sub(r'from attention import', 'from scripts.core.attention import', content)
                content = re.sub(r'from transformer import', 'from scripts.core.transformer import', content)
                content = re.sub(r'from generator import', 'from scripts.generation.generator import', content)
                content = re.sub(r'from postprocessor import', 'from scripts.generation.postprocessor import', content)
                content = re.sub(r'from cache import', 'from scripts.optimization.cache import', content)
                content = re.sub(r'from logger import', 'from scripts.utils.logger import', content)
                content = re.sub(r'from visualizer import', 'from scripts.utils.visualizer import', content)
                content = re.sub(r'from pipeline import', 'from scripts.pipeline import', content)
                
                if content != original:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✓ Updated: {os.path.relpath(file_path)}")
                    updated_count += 1
            else:
                if update_file_imports(file_path, module_type):
                    print(f"✓ Updated: {os.path.relpath(file_path)}")
                    updated_count += 1
                else:
                    print(f"- No changes: {os.path.relpath(file_path)}")
        else:
            print(f"✗ Not found: {os.path.relpath(file_path)}")
    
    print(f"\n总计更新 {updated_count} 个文件")

if __name__ == '__main__':
    main()
