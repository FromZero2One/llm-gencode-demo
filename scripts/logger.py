"""
统一日志管理工具 - 解决Windows控制台缓冲区限制问题

功能：
1. 双重输出：同时输出到控制台和日志文件
2. 自动命名：根据脚本名称自动生成日志文件名
3. 时间戳：日志文件包含时间戳，避免覆盖
4. 编码支持：使用UTF-8编码，支持中文

使用示例：
    from logger import setup_logger
    
    # 在脚本开头设置logger
    logger = setup_logger(__file__)
    
    # 所有print()会自动输出到控制台和日志文件
    print("这条消息会同时显示在控制台和日志文件中")
"""

import sys
import os
import time
from datetime import datetime


class DualOutputLogger:
    """
    双重输出Logger - 同时输出到控制台和文件
    
    解决Windows控制台缓冲区限制问题，确保完整日志可追溯
    """
    
    def __init__(self, log_file=None):
        """
        初始化Logger
        
        Args:
            log_file: 日志文件路径，如果为None则只输出到控制台
        """
        self.log_file = log_file
        self.file_handle = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        if log_file:
            try:
                # 确保目录存在
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                # 以UTF-8编码打开文件
                self.file_handle = open(log_file, 'w', encoding='utf-8')
                
                # 写入文件头信息
                self._write_header()
                
            except Exception as e:
                print(f"[WARNING] 无法创建日志文件 {log_file}: {e}")
                self.file_handle = None
    
    def _write_header(self):
        """写入日志文件头部信息"""
        if self.file_handle:
            header = f"""{'='*80}
日志文件: {os.path.basename(self.log_file)}
创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Python版本: {sys.version}
工作目录: {os.getcwd()}
{'='*80}

"""
            self.file_handle.write(header)
            self.file_handle.flush()
    
    def write(self, text):
        """
        写入文本到控制台和文件
        
        Args:
            text: 要写入的文本
        """
        # 输出到原始控制台（避免递归）
        if self.original_stdout:
            try:
                self.original_stdout.write(text)
                self.original_stdout.flush()
            except Exception:
                pass
        
        # 输出到文件
        if self.file_handle:
            try:
                self.file_handle.write(text)
                self.file_handle.flush()
            except Exception:
                pass
    
    def flush(self):
        """刷新缓冲区"""
        if self.original_stdout:
            try:
                self.original_stdout.flush()
            except Exception:
                pass
        
        if self.file_handle:
            try:
                self.file_handle.flush()
            except Exception:
                pass
    
    def close(self):
        """关闭文件句柄并恢复标准输出"""
        if self.file_handle:
            try:
                # 写入尾部信息
                footer = f"\n{'='*80}\n日志结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
                self.file_handle.write(footer)
                self.file_handle.close()
            except Exception:
                pass
            finally:
                self.file_handle = None
        
        # 恢复标准输出
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动关闭"""
        self.close()


def generate_log_filename(script_path, log_dir='logs'):
    """
    根据脚本路径生成日志文件名
    
    Args:
        script_path: 脚本的文件路径（通常是 __file__）
        log_dir: 日志目录，默认为'logs'
    
    Returns:
        完整的日志文件路径
    
    Example:
        >>> generate_log_filename('scripts/pipeline.py')
        'logs/pipeline_20260509_155630.log'
    """
    # 获取脚本文件名（不含扩展名）
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 组合文件名
    log_filename = f"{script_name}_{timestamp}.log"
    
    # 构建完整路径
    if not os.path.isabs(log_dir):
        # 如果是相对路径，基于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(project_root, log_dir)
    
    log_path = os.path.join(log_dir, log_filename)
    
    return log_path


def setup_logger(script_path=None, log_dir='logs', enable_logging=True):
    """
    设置双重输出Logger
    
    Args:
        script_path: 脚本路径（通常传入 __file__），用于生成日志文件名
        log_dir: 日志目录，默认为'logs'
        enable_logging: 是否启用日志文件输出
    
    Returns:
        DualOutputLogger 实例
    
    Example:
        # 在脚本开头调用
        logger = setup_logger(__file__)
        
        # 所有print()会自动输出到控制台和日志文件
        print("测试消息")
        
        # 脚本结束时关闭logger
        logger.close()
    """
    if not enable_logging:
        # 返回一个空的logger，只输出到控制台
        return DualOutputLogger(log_file=None)
    
    if script_path is None:
        # 如果没有提供脚本路径，尝试从调用栈获取
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            if caller_frame:
                script_path = caller_frame.f_code.co_filename
            else:
                script_path = 'unknown_script.py'
        finally:
            del frame
    
    # 生成日志文件路径
    log_file = generate_log_filename(script_path, log_dir)
    
    # 创建并返回logger
    logger = DualOutputLogger(log_file=log_file)
    
    # 重定向标准输出
    sys.stdout = logger
    sys.stderr = logger
    
    # 打印启动信息
    print(f"\n[INFO] 日志系统已启动")
    print(f"[INFO] 日志文件: {log_file}")
    print(f"[INFO] 所有输出将同时显示在控制台和保存到日志文件\n")
    
    return logger


def cleanup_logger(logger):
    """
    清理Logger，恢复标准输出
    
    Args:
        logger: DualOutputLogger 实例
    """
    if logger:
        logger.close()


# 便捷函数：作为上下文管理器使用
class logging_context:
    """
    日志上下文管理器
    
    使用with语句自动管理日志生命周期
    
    Example:
        with logging_context(__file__):
            print("这条消息会被记录到日志文件")
        # 退出with块时自动关闭日志
    """
    
    def __init__(self, script_path=None, log_dir='logs', enable_logging=True):
        self.script_path = script_path
        self.log_dir = log_dir
        self.enable_logging = enable_logging
        self.logger = None
    
    def __enter__(self):
        self.logger = setup_logger(
            script_path=self.script_path,
            log_dir=self.log_dir,
            enable_logging=self.enable_logging
        )
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_logger(self.logger)
        return False  # 不抑制异常
