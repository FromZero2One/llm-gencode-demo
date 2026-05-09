"""
后处理模块 - 对生成的代码进行格式化、验证和优化
"""

import sys
import re
from typing import List, Dict
from logger import logging_context


class SyntaxValidator:
    """简单的语法验证器"""
    
    def validate(self, code: str) -> Dict:
        """
        验证代码的基本语法
        
        Returns:
            dict包含验证结果和错误信息
        """
        errors = []
        warnings = []
        
        # 1. 检查括号匹配
        bracket_errors = self._check_brackets(code)
        errors.extend(bracket_errors)
        
        # 2. 检查分号（简化检查）
        # 这里只是示例，实际应该更复杂
        
        # 3. 检查常见错误模式
        if 'undefined' in code.lower():
            errors.append("包含'undefined'关键字")
        
        if code.count('class') > 0 and '{' not in code:
            errors.append("类定义缺少花括号")
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'error_count': len(errors),
            'warning_count': len(warnings)
        }
    
    def _check_brackets(self, code: str) -> List[str]:
        """检查括号匹配"""
        errors = []
        stack = []
        
        bracket_pairs = {
            '(': ')',
            '{': '}',
            '[': ']'
        }
        
        closing_brackets = set(bracket_pairs.values())
        
        for i, char in enumerate(code):
            if char in bracket_pairs:
                stack.append((char, i))
            elif char in closing_brackets:
                if not stack:
                    errors.append(f"位置 {i}: 未匹配的闭合括号 '{char}'")
                else:
                    last_open, last_pos = stack.pop()
                    if bracket_pairs[last_open] != char:
                        errors.append(
                            f"位置 {i}: 期望 '{bracket_pairs[last_open]}' "
                            f"但找到 '{char}'"
                        )
        
        # 检查是否有未闭合的括号
        for open_bracket, pos in stack:
            errors.append(f"位置 {pos}: 未闭合的括号 '{open_bracket}'")
        
        return errors


class JavaCodeFormatter:
    """Java代码格式化器（简化版）"""
    
    def format(self, code: str) -> str:
        """
        格式化Java代码
        
        包括：
        1. 调整缩进
        2. 规范化空格
        3. 整理import语句
        """
        print("[Formatter] 开始格式化代码")
        
        # 1. 标准化换行符
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # 2. 调整缩进
        code = self._fix_indentation(code)
        
        # 3. 清理多余空行
        code = self._remove_extra_blank_lines(code)
        
        # 4. 规范化空格
        code = self._normalize_spaces(code)
        
        print("[Formatter] 格式化完成")
        
        return code
    
    def _fix_indentation(self, code: str) -> str:
        """修复缩进"""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        indent_size = 4
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                formatted_lines.append('')
                continue
            
            # 减少缩进的情况（闭合括号）
            if stripped.startswith('}') or stripped.startswith(')'):
                indent_level = max(0, indent_level - 1)
            
            # 添加当前行（带缩进）
            formatted_lines.append(' ' * (indent_level * indent_size) + stripped)
            
            # 增加缩进的情况（开启括号）
            if stripped.endswith('{') or stripped.endswith('('):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
    
    def _remove_extra_blank_lines(self, code: str) -> str:
        """移除多余的空白行（保留最多2个连续空行）"""
        lines = code.split('\n')
        result = []
        blank_count = 0
        
        for line in lines:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 2:
                    result.append(line)
            else:
                blank_count = 0
                result.append(line)
        
        return '\n'.join(result)
    
    def _normalize_spaces(self, code: str) -> str:
        """规范化空格"""
        # 运算符周围的空格
        code = re.sub(r'\s*=\s*', ' = ', code)
        code = re.sub(r'\s*\+\s*', ' + ', code)
        code = re.sub(r'\s*-\s*', ' - ', code)
        
        # 逗号后的空格
        code = re.sub(r',\s*', ', ', code)
        
        # 分号前的空格
        code = re.sub(r'\s+;', ';', code)
        
        # 清理多余空格
        code = re.sub(r'  +', ' ', code)
        
        return code


class ImportManager:
    """管理import语句"""
    
    def __init__(self):
        # 常见类和对应的import
        self.import_map = {
            'List': 'java.util.List',
            'ArrayList': 'java.util.ArrayList',
            'Map': 'java.util.Map',
            'HashMap': 'java.util.HashMap',
            'Set': 'java.util.Set',
            'HashSet': 'java.util.HashSet',
            'Optional': 'java.util.Optional',
            'Stream': 'java.util.stream.Stream',
            'Collectors': 'java.util.stream.Collectors',
            'CompletableFuture': 'java.util.concurrent.CompletableFuture',
            'Service': 'org.springframework.stereotype.Service',
            'Component': 'org.springframework.stereotype.Component',
            'Autowired': 'org.springframework.beans.factory.annotation.Autowired',
            'Override': 'java.lang.Override',
            'RestController': 'org.springframework.web.bind.annotation.RestController',
            'GetMapping': 'org.springframework.web.bind.annotation.GetMapping',
            'PostMapping': 'org.springframework.web.bind.annotation.PostMapping',
            'RequestBody': 'org.springframework.web.bind.annotation.RequestBody',
            'ResponseBody': 'org.springframework.web.bind.annotation.ResponseBody',
            'PathVariable': 'org.springframework.web.bind.annotation.PathVariable',
            'Transactional': 'org.springframework.transaction.annotation.Transactional',
            'Slf4j': 'lombok.extern.slf4j.Slf4j',
            'Log': 'lombok.extern.slf4j.Slf4j',
        }
    
    def analyze_and_add_imports(self, code: str) -> str:
        """
        分析代码中使用的类并添加相应的import语句
        
        Args:
            code: Java代码
            
        Returns:
            添加了import的代码
        """
        print("[ImportManager] 分析并添加import语句")
        
        # 提取package声明
        package_match = re.search(r'package\s+([\w.]+)\s*;', code)
        package_line = package_match.group(0) if package_match else ''
        
        # 找出代码中使用的所有类名
        used_classes = set()
        for class_name in self.import_map.keys():
            # 使用单词边界匹配
            pattern = r'\b' + re.escape(class_name) + r'\b'
            if re.search(pattern, code):
                used_classes.add(class_name)
        
        # 生成import语句
        imports = []
        for class_name in sorted(used_classes):
            import_path = self.import_map[class_name]
            imports.append(f'import {import_path};')
        
        if not imports:
            return code
        
        # 构建完整的import部分
        import_section = '\n'.join(imports)
        
        # 在package声明后插入import
        if package_line:
            code = code.replace(package_line, package_line + '\n\n' + import_section)
        else:
            # 如果没有package声明，在开头添加
            code = import_section + '\n\n' + code
        
        print(f"[ImportManager] 添加了 {len(imports)} 个import语句")
        
        return code


class CodeOptimizer:
    """代码优化器"""
    
    def optimize(self, code: str) -> str:
        """
        优化代码结构
        
        包括：
        1. 移除未使用的变量
        2. 简化常见的代码模式
        3. 添加必要的注释
        """
        print("[Optimizer] 优化代码结构")
        
        # 这里可以添加更多优化逻辑
        # 目前只做基本的清理
        
        # 移除末尾的多余空白
        code = code.rstrip()
        
        # 确保文件以换行符结尾
        if not code.endswith('\n'):
            code += '\n'
        
        print("[Optimizer] 优化完成")
        
        return code


class CodePostProcessor:
    """
    代码后处理器
    
    整合所有后处理步骤：
    1. 语法验证
    2. 代码格式化
    3. Import管理
    4. 代码优化
    """
    
    def __init__(self):
        self.validator = SyntaxValidator()
        self.formatter = JavaCodeFormatter()
        self.import_manager = ImportManager()
        self.optimizer = CodeOptimizer()
        
        print("[PostProcessor] 初始化完成")
    
    def process(self, generated_code: str) -> Dict:
        """
        完整的后处理流程
        
        Args:
            generated_code: 生成的原始代码
            
        Returns:
            dict包含处理后的代码和验证报告
        """
        print(f"\n{'='*60}")
        print(f"[PostProcessor] 开始后处理")
        print(f"{'='*60}")
        print(f"原始代码长度: {len(generated_code)} 字符")
        
        result = {
            'original_code': generated_code,
            'processed_code': generated_code,
            'validation_report': None,
            'steps_applied': []
        }
        
        try:
            # 步骤1: 语法验证
            print("\n[Step 1] 语法验证")
            validation = self.validator.validate(generated_code)
            result['validation_report'] = validation
            
            if validation['is_valid']:
                print("  [OK] 语法验证通过")
                result['steps_applied'].append('validation_passed')
            else:
                print(f"  [ERROR] 发现 {validation['error_count']} 个错误:")
                for error in validation['errors']:
                    print(f"    - {error}")
                result['steps_applied'].append('validation_failed')
            
            # 步骤2: 代码格式化
            print("\n[Step 2] 代码格式化")
            formatted_code = self.formatter.format(result['processed_code'])
            result['processed_code'] = formatted_code
            result['steps_applied'].append('formatting')
            print(f"  格式化后长度: {len(formatted_code)} 字符")
            
            # 步骤3: 添加import语句
            print("\n[Step 3] 添加import语句")
            code_with_imports = self.import_manager.analyze_and_add_imports(formatted_code)
            result['processed_code'] = code_with_imports
            result['steps_applied'].append('import_management')
            print(f"  添加import后长度: {len(code_with_imports)} 字符")
            
            # 步骤4: 代码优化
            print("\n[Step 4] 代码优化")
            optimized_code = self.optimizer.optimize(code_with_imports)
            result['processed_code'] = optimized_code
            result['steps_applied'].append('optimization')
            print(f"  优化后长度: {len(optimized_code)} 字符")
            
            print(f"\n{'='*60}")
            print(f"[PostProcessor] 后处理完成")
            print(f"{'='*60}")
            print(f"应用的步骤: {', '.join(result['steps_applied'])}")
            print(f"最终代码长度: {len(result['processed_code'])} 字符")
            
        except Exception as e:
            print(f"\n[PostProcessor] 后处理出错: {e}")
            result['error'] = str(e)
        
        return result
    
    def quick_process(self, code: str) -> str:
        """
        快速处理（只进行格式化和import管理）
        
        Args:
            code: 原始代码
            
        Returns:
            处理后的代码
        """
        # 格式化
        formatted = self.formatter.format(code)
        
        # 添加import
        with_imports = self.import_manager.analyze_and_add_imports(formatted)
        
        return with_imports


# 测试代码
if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        print("="*60)
        print("测试后处理模块")
        print("="*60)
        
        # 创建后处理器
        post_processor = CodePostProcessor()
    
    # 测试代码
    test_code = """
public class UserService { public User findById(Long id) { return userRepository.findById(id).orElse(null); } public List<User> findAll() { return userRepository.findAll(); } }
"""
    
    print(f"\n原始代码:\n{test_code}")
    
    # 完整后处理
    result = post_processor.process(test_code)
    
    print(f"\n处理后代码:\n{result['processed_code']}")
    
    print(f"\n验证报告:")
    print(f"  是否有效: {result['validation_report']['is_valid']}")
    print(f"  错误数: {result['validation_report']['error_count']}")
    print(f"  警告数: {result['validation_report']['warning_count']}")
    
    print(f"\n应用的步骤: {result['steps_applied']}")
