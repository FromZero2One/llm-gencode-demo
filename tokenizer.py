"""
Token化模块 - 将文本转换为token序列
用于理解和调试大模型的输入处理过程
"""

import re
import logging
from typing import List, Dict, Tuple

# 配置日志 - 如果没有配置过,设置默认级别为 DEBUG
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# 配置日志
logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """
    简化的代码Token化器
    模拟真实LLM的BPE分词过程
    
    Example:
        >>> tokenizer = SimpleTokenizer(vocab_size=1000)
        >>> ids, mask = tokenizer.encode("public class User")
        >>> text = tokenizer.decode(ids)
    """
    
    # 类变量：缓存不同大小的词汇表
    _vocab_cache = {}
    
    def __init__(self, vocab_size: int = 1000, debug_mode: bool = True):
        self.vocab_size = vocab_size
        self.debug_mode = debug_mode
        self.PAD_TOKEN = '<PAD>'
        self.EOS_TOKEN = '<EOS>'
        self.BOS_TOKEN = '<BOS>'
        self.UNK_TOKEN = '<UNK>'
        
        # 使用缓存或构建新词汇表
        if vocab_size not in SimpleTokenizer._vocab_cache:
            SimpleTokenizer._vocab_cache[vocab_size] = self._build_vocabulary()
        
        self.vocab = SimpleTokenizer._vocab_cache[vocab_size]
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        if self.debug_mode:
            logger.info(f"词汇表大小: {len(self.vocab)}")
            logger.info(f"PAD token ID: {self.vocab[self.PAD_TOKEN]}")
            logger.info(f"EOS token ID: {self.vocab[self.EOS_TOKEN]}")
            logger.info(f"BOS token ID: {self.vocab[self.BOS_TOKEN]}")
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """构建简化的词汇表"""
        vocab = {}
        idx = 0
        
        # 特殊token
        vocab[self.PAD_TOKEN] = idx; idx += 1
        vocab[self.EOS_TOKEN] = idx; idx += 1
        vocab[self.BOS_TOKEN] = idx; idx += 1
        vocab[self.UNK_TOKEN] = idx; idx += 1
        
        # Java关键字
        java_keywords = [
            'public', 'private', 'protected', 'class', 'interface', 'enum',
            'static', 'final', 'void', 'int', 'long', 'double', 'float',
            'boolean', 'String', 'return', 'if', 'else', 'for', 'while',
            'do', 'switch', 'case', 'break', 'continue', 'try', 'catch',
            'finally', 'throw', 'throws', 'new', 'this', 'super', 'import',
            'package', 'extends', 'implements', 'abstract', 'synchronized',
            'volatile', 'transient', 'native', 'strictfp', 'assert', 'instanceof'
        ]
        for kw in java_keywords:
            vocab[kw] = idx; idx += 1
        
        # 常用标识符和符号
        common_tokens = [
            '{', '}', '(', ')', '[', ']', ';', ',', '.', '=', '<', '>',
            '!', '&', '|', '+', '-', '*', '/', '%', '@', '#', '$',
            'Service', 'Controller', 'Repository', 'Autowired', 'Override',
            'List', 'Map', 'Set', 'ArrayList', 'HashMap', 'HashSet',
            'Optional', 'Stream', 'Collectors', 'CompletableFuture',
            'System', 'out', 'println', 'log', 'logger', 'INFO', 'ERROR',
            'param', 'result', 'data', 'user', 'id', 'name', 'value',
            'get', 'set', 'create', 'update', 'delete', 'find', 'save',
            'userRepository', 'userService', 'userController',
            'findById', 'findAll', 'save', 'delete', 'exists',
            'GetMapping', 'PostMapping', 'PutMapping', 'DeleteMapping',
            'RequestBody', 'ResponseBody', 'PathVariable', 'RequestParam',
            'RestController', 'RequestMapping', 'Transactional',
            'Slf4j', 'Component', 'Bean', 'Configuration',
            'Exception', 'RuntimeException', 'IllegalArgumentException',
            'NullPointerException', 'IOException', 'SQLException',
            'Collections', 'Objects', 'Arrays', 'StringUtils',
            'true', 'false', 'null', '==', '!=', '<=', '>=', '&&', '||'
        ]
        for token in common_tokens:
            if token not in vocab:
                vocab[token] = idx; idx += 1
        
        # 常见变量名模式（简化）
        for i in range(100):
            vocab[f'var{i}'] = idx; idx += 1
        
        return vocab
    
    def tokenize(self, text: str) -> List[str]:
        """
        将文本分割为tokens
        使用简单的正则表达式分词
        """
        # 匹配标识符、数字、运算符、标点等
        pattern = r'[a-zA-Z_]\w*|[{}();,.=\[\]<>!&|+\-*/%@#$]|\S+'
        tokens = re.findall(pattern, text)
        
        if self.debug_mode:
            logger.debug(f"原始文本长度: {len(text)} 字符")
            logger.debug(f"分词结果数量: {len(tokens)} tokens")
            if len(tokens) <= 20:
                logger.debug(f"Tokens: {tokens}")
        
        return tokens
    
    def encode(self, text: str, max_length: int = 128) -> Tuple[List[int], List[int]]:
        """
        将文本编码为token IDs
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            
        Returns:
            token_ids: token ID列表
            attention_mask: 注意力掩码（1表示有效token，0表示padding）
        """
        if self.debug_mode:
            logger.debug("=" * 60)
            logger.debug("开始编码文本")
            logger.debug("=" * 60)
        
        # 1. 分词
        tokens = self.tokenize(text)
        
        # 2. 添加特殊token
        tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        
        # 3. 转换为IDs
        token_ids = []
        unk_count = 0
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.UNK_TOKEN])
                unk_count += 1
        
        if self.debug_mode:
            logger.debug(f"UNK token数量: {unk_count}")
            logger.debug(f"编码前长度: {len(token_ids)}")
        
        # 4. 截断或填充
        original_length = len(token_ids)
        if len(token_ids) > max_length:
            # 截断（保留BOS，移除EOS前的部分）
            token_ids = token_ids[:max_length-1] + [self.vocab[self.EOS_TOKEN]]
            if self.debug_mode:
                logger.debug(f"截断到 {max_length} tokens")
        elif len(token_ids) < max_length:
            # 填充
            pad_id = self.vocab[self.PAD_TOKEN]
            padding_length = max_length - len(token_ids)
            token_ids.extend([pad_id] * padding_length)
            if self.debug_mode:
                logger.debug(f"填充 {padding_length} 个PAD tokens")
        
        # 5. 创建注意力掩码
        attention_mask = [1] * original_length + [0] * (max_length - original_length)
        
        if self.debug_mode:
            logger.debug(f"最终序列长度: {len(token_ids)}")
            logger.debug(f"有效tokens: {sum(attention_mask)}")
            logger.debug("=" * 60)
        
        return token_ids, attention_mask
    
    def decode(self, token_ids: List[int]) -> str:
        """
        将token IDs解码为文本
        
        Args:
            token_ids: token ID列表
            
        Returns:
            解码后的文本
        """
        tokens = []
        for tid in token_ids:
            if tid in self.reverse_vocab:
                token = self.reverse_vocab[tid]
                # 跳过特殊token
                if token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.UNK_TOKEN)
        
        # 简单的拼接逻辑（实际应该更复杂）
        text = ' '.join(tokens)
        # 清理空格（标点符号前不加空格）
        text = re.sub(r'\s+([{}();,.])', r'\1', text)
        text = re.sub(r'([{}();,.])\s+', r'\1 ', text)
        
        return text.strip()
    
    def get_token_info(self, token_id: int) -> Dict:
        """获取token的详细信息"""
        token = self.reverse_vocab.get(token_id, self.UNK_TOKEN)
        return {
            'id': token_id,
            'token': token,
            'is_special': token in [self.PAD_TOKEN, self.EOS_TOKEN, self.BOS_TOKEN, self.UNK_TOKEN],
            'is_keyword': token in ['public', 'private', 'class', 'return', 'if', 'else', 'for', 'while']
        }


# 测试代码
if __name__ == '__main__':
    # 创建tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)
    
    # 测试文本
    test_text = "public class UserService { public User findById(Long id) { return userRepository.findById(id).orElse(null); } }"
    
    print("\n" + "="*60)
    print("测试Token化过程")
    print("="*60 + "\n")
    
    # 编码
    token_ids, attention_mask = tokenizer.encode(test_text, max_length=64)
    
    print(f"\nToken IDs (前20个): {token_ids[:20]}")
    print(f"Attention Mask (前20个): {attention_mask[:20]}")
    
    # 解码
    decoded_text = tokenizer.decode(token_ids)
    print(f"\n解码结果:\n{decoded_text}")
    
    # 查看特定token信息
    print(f"\nToken详细信息示例:")
    for i in range(min(10, len(token_ids))):
        info = tokenizer.get_token_info(token_ids[i])
        print(f"  Position {i}: ID={info['id']}, Token='{info['token']}', "
              f"Special={info['is_special']}, Keyword={info['is_keyword']}")
