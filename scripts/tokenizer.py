"""
Token化模块 - 将文本转换为token序列
用于理解和调试大模型的输入处理过程
"""

import sys
import re
import logging
from typing import List, Dict, Tuple
from logger import logging_context

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
    模拟真实LLM的BPE分词过程，用于将源代码转换为模型可处理的数值序列。
    
    主要功能：
    - 将Java代码文本分割为token序列
    - 将token映射为数值ID（编码）
    - 将数值ID还原为可读文本（解码）
    - 支持固定长度序列的填充和截断
    - 提供注意力掩码 (Attention Mask) 用于Transformer模型
    
    Attention Mask 说明：
    - 作用：标记哪些位置是有效内容(1)，哪些是padding(0)
    - 原理：在Self-Attention中，mask=0的位置会被加上极大负数，使模型忽略padding
    - 必要性：批量处理不同长度文本时，确保模型只关注有效tokens
    - 示例：text="return" → mask=[1,1,1,0,0,0...] (3个有效+多个padding)
    
    Attributes:
        vocab_size (int): 词汇表大小，决定能识别的token数量
        debug_mode (bool): 是否启用调试日志输出
        PAD_TOKEN (str): 填充token，用于补齐序列到固定长度 (ID=0)
        EOS_TOKEN (str): 结束标记，表示序列结尾
        BOS_TOKEN (str): 开始标记，表示序列开头
        UNK_TOKEN (str): 未知标记，用于词汇表外的token
        vocab (Dict[str, int]): token到ID的映射字典
        reverse_vocab (Dict[int, str]): ID到token的反向映射字典
    
    Example:
        >>> tokenizer = SimpleTokenizer(vocab_size=1000)
        >>> ids, mask = tokenizer.encode("public class User")
        >>> # ids: [2, 5, 7, 49, 1, 0, 0, ...]  (BOS + tokens + EOS + PAD)
        >>> # mask: [1, 1, 1, 1, 1, 0, 0, ...]  (有效=1, padding=0)
        >>> text = tokenizer.decode(ids)
    """
    
    # 类变量：缓存不同大小的词汇表，避免重复构建
    _vocab_cache = {}
    
    def __init__(self, vocab_size: int = 1000, debug_mode: bool = True):
        """
        初始化SimpleTokenizer实例
        
        Args:
            vocab_size (int): 词汇表大小，默认为1000。
                             较大的词汇表可以识别更多token，但会增加内存占用。
            debug_mode (bool): 是否启用调试模式，默认为True。
                              启用后会输出详细的分词和编码日志信息。
        
        Note:
            - 使用类级别的词汇表缓存，相同大小的词汇表只构建一次
            - 自动创建反向映射字典用于解码操作
        """
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
        """
        构建简化的词汇表
        
        按顺序添加以下类型的token：
        1. 特殊token（PAD, EOS, BOS, UNK）
        2. Java关键字（public, class, return等）
        3. 常用符号和标识符（运算符、Spring注解、常见类名等）
        4. 变量名模式（var0-var99）
        
        Returns:
            Dict[str, int]: token到ID的映射字典，ID从0开始递增
        
        Note:
            - 词汇表大小由vocab_size参数控制
            - 如果token数量超过vocab_size，超出的部分会被截断
        """
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
        
        使用正则表达式对输入文本进行分词，识别以下类型的token：
        - 标识符（变量名、方法名、类名等）
        - 运算符和标点符号
        - 其他非空白字符序列
        
        Args:
            text (str): 待分词的原始文本，通常是Java代码
            
        Returns:
            List[str]: 分词后的token列表，按出现顺序排列
            
        Example:
            >>> tokenizer = SimpleTokenizer(vocab_size=1000)
            >>> tokenizer.tokenize("public class User { }")
            ['public', 'class', 'User', '{', '}']
        
        Note:
            - 使用正则模式：[a-zA-Z_]\\w*|[{}();,.=\\[\\]<>!&|+\\-*/%@#$]|\\S+
            - 第一个模式匹配标识符（字母或下划线开头，后跟字母数字下划线）
            - 第二个模式匹配常见运算符和标点
            - 第三个模式匹配其他非空白字符（兜底规则）
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
        将文本编码为token IDs序列
        
        完整的编码流程：
        1. 分词：将输入文本分割为token列表
        2. 添加特殊标记：在开头添加BOS，结尾添加EOS
        3. ID转换：将每个token映射为对应的数值ID（未知token使用UNK）
        4. 长度调整：截断过长序列或填充过短序列到max_length
        5. 生成掩码：创建注意力掩码标识有效token位置
        
        Args:
            text (str): 待编码的输入文本，通常是Java代码
            max_length (int): 输出序列的最大长度，默认为128。
                             - 如果token数超过此值，会截断多余部分
                             - 如果token数不足此值，会用PAD token填充
            
            💡 max_length 与上下文长度 (Context Length) 的关系：
            
            max_length 本质上就是大模型中的"上下文长度"概念，但这里是简化版本：
            
            【相同点】
            - 都限制了模型能处理的最大 token 数量
            - 超出限制的内容需要特殊处理（截断或分块）
            - 都需要 attention mask 来区分有效内容和padding
            
            【不同点】
            - 本代码：教学简化版，max_length=128（便于观察和调试）
            - 真实模型：GPT-3(2K), GPT-4(8K-128K), Claude(100K), Llama 3(8K)
            
            【为什么真实模型有固定的上下文长度？】
            1. 位置编码限制：Transformer需要知道每个token的位置，位置编码范围有限
            2. 注意力复杂度：Self-Attention计算复杂度是O(n²)，序列越长计算量呈平方增长
               - 1K tokens → 1M次计算
               - 10K tokens → 100M次计算  
               - 100K tokens → 10B次计算！
            3. 训练配置：模型在训练时就确定了最大序列长度
            
            【现代模型的长上下文技术】
            - RoPE (Rotary Positional Embedding): Llama、PaLM使用
            - ALiBi (Attention with Linear Biases): 支持extrapolate到更长序列
            - Sliding Window Attention: LongFormer、BigBird，只关注局部窗口
            - Context Compression: 将长文本压缩成关键信息
            
            【本代码的设计选择】
            - 使用较小的max_length(默认128)是为了教学目的
            - 便于观察tokenization过程和attention mask的作用
            - 减少内存占用，加快测试速度
            - 实际应用中应根据目标模型的架构设置合适的值
            
        Returns:
            Tuple[List[int], List[int]]: 包含两个列表的元组
                - token_ids (List[int]): 编码后的token ID序列，长度固定为max_length
                  示例：[2, 5, 7, 49, 1, 0, 0, 0, 0, 0]  (BOS=2, PAD=0)
                
                - attention_mask (List[int]): 注意力掩码序列，长度与token_ids相同
                  * 1 表示该位置是有效token（模型应该关注）
                  * 0 表示该位置是padding（模型应该忽略）
                  示例：[1, 1, 1, 1,  1, 0, 0, 0, 0, 0]
                        ↑^^^有效内容^^^^↑↑^^^padding^^^↑
                  
                  Attention Mask 在 Transformer 中的作用：
                  - Self-Attention 计算时，mask=0的位置会被加上极大负数(-1e9)
                  - Softmax后，这些位置的权重趋近于0，模型不会关注padding
                  - 确保批量处理不同长度文本时，模型只关注有效内容
        
        Example:
            >>> tokenizer.encode("public class User", max_length=10)
            # Tokens: [BOS, public, class, User, EOS] + 5个PAD
            # 返回: ([2, 5, 7, 49, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
            #       ↑^^^^^token IDs^^^^^^^^^^^^^^↑ ↑^^^^attention mask^^^^^↑
        
        Note:
            - BOS (Begin of Sequence) 标记始终位于序列开头
            - EOS (End of Sequence) 标记在截断时会被保留在末尾
            - 未知token会被替换为UNK_TOKEN的ID
            - attention_mask用于Transformer模型区分有效token和padding
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
        
        # 4. 截断或填充（确保所有序列长度一致，便于批量处理）
        original_length = len(token_ids)
        if len(token_ids) > max_length:
            # 截断：保留BOS和前面的tokens，确保EOS在末尾
            # 原因：Transformer要求固定长度的输入张量
            token_ids = token_ids[:max_length-1] + [self.vocab[self.EOS_TOKEN]]
            if self.debug_mode:
                logger.debug(f"截断到 {max_length} tokens")
        elif len(token_ids) < max_length:
            # 填充：用PAD token补齐到max_length
            # 原因：批量处理时，不同长度的文本需要统一长度
            pad_id = self.vocab[self.PAD_TOKEN]
            padding_length = max_length - len(token_ids)
            token_ids.extend([pad_id] * padding_length)
            if self.debug_mode:
                logger.debug(f"填充 {padding_length} 个PAD tokens")
        
        # 5. 创建注意力掩码 (Attention Mask)
        # 
        # Attention Mask 是 Transformer 模型的关键组件，用于区分有效内容和填充内容：
        # - 1 表示该位置是有效token（模型应该关注）
        # - 0 表示该位置是padding（模型应该忽略）
        #
        # 工作原理：
        #   在 Self-Attention 计算中，mask 会被应用到注意力分数上：
        #   masked_scores = scores + (1 - mask) * -1e9
        #   这样 padding 位置的分数会变成极大负数，softmax 后权重趋近于0
        #   模型就不会关注无意义的 padding tokens
        #
        # 示例：
        #   text = "public class User" → 5个tokens (含BOS/EOS)
        #   max_length = 10
        #   token_ids =      [2, 4, 7, 49, 1, 0, 0, 0, 0, 0]
        #                    ↑^^^^有效内容^^^^^↑↑^^padding^^↑
        #   attention_mask = [1, 1, 1, 1,  1, 0, 0, 0, 0, 0]
        #                    ↑^^^^有效=1^^^^^↑↑^^padding=0^↑
        #
        # 为什么需要 Attention Mask？
        #   1. 批量处理：同时处理多个不同长度的文本时，需要统一长度
        #   2. 提高效率：避免模型浪费计算资源在 padding 上
        #   3. 保证准确性：防止 padding 影响模型的注意力分布
        attention_mask = [1] * original_length + [0] * (max_length - original_length)
        #                ↑^^^^有效tokens^^^^^↑↑^^^^padding^^^^^↑
        
        if self.debug_mode:
            logger.debug(f"最终序列长度: {len(token_ids)}")
            logger.debug(f"有效tokens: {sum(attention_mask)}")
            logger.debug("=" * 60)
        
        return token_ids, attention_mask
    
    def decode(self, token_ids: List[int]) -> str:
        """
        将token IDs序列解码为可读文本
        
        解码流程：
        1. ID转换：将每个数值ID通过反向映射转换为对应的token字符串
        2. 过滤特殊标记：跳过PAD、BOS、EOS等特殊token
        3. 拼接文本：用空格连接所有token
        4. 清理格式：调整标点符号周围的空格，使输出更符合代码规范
        
        Args:
            token_ids (List[int]): 待解码的token ID序列，通常来自encode方法的输出
            
        Returns:
            str: 解码后的文本字符串，已去除特殊标记并格式化
            
        Example:
            >>> tokenizer.decode([3, 5, 6, 7, 4])
            'public class User { }'
        
        Note:
            - PAD、BOS、EOS token会被自动过滤，不出现在输出中
            - UNK token会保留在输出中，表示无法识别的token
            - 使用正则表达式清理标点符号周围的空格：
              * 删除标点前的空格："{ }" → "{}"
              * 在标点后添加空格："}{" → "} {"
            - 最终结果会去除首尾空白字符
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
        """
        获取指定token ID的详细信息
        
        用于调试和分析，返回token的属性和分类信息。
        
        Args:
            token_id (int): 要查询的token ID，通常在0到vocab_size-1范围内
            
        Returns:
            Dict: 包含token详细信息的字典，包含以下键：
                - 'id' (int): token的数值ID
                - 'token' (str): token对应的字符串表示
                - 'is_special' (bool): 是否为特殊token（PAD/EOS/BOS/UNK）
                - 'is_keyword' (bool): 是否为Java关键字
                
        Example:
            >>> tokenizer.get_token_info(5)
            {'id': 5, 'token': 'public', 'is_special': False, 'is_keyword': True}
            
            >>> tokenizer.get_token_info(0)
            {'id': 0, 'token': '<PAD>', 'is_special': True, 'is_keyword': False}
        
        Note:
            - 如果token_id不在词汇表中，会返回UNK_TOKEN的信息
            - is_keyword仅检查部分常见Java关键字，不是完整列表
        """
        token = self.reverse_vocab.get(token_id, self.UNK_TOKEN)
        return {
            'id': token_id,
            'token': token,
            'is_special': token in [self.PAD_TOKEN, self.EOS_TOKEN, self.BOS_TOKEN, self.UNK_TOKEN],
            'is_keyword': token in ['public', 'private', 'class', 'return', 'if', 'else', 'for', 'while']
        }


# 测试代码
if __name__ == '__main__':
    # 使用日志上下文管理器
    with logging_context(__file__):
        # 创建tokenizer
        tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)
        
        # 测试文本
        test_text = "public class UserService { public User findById(Long id) { return userRepository.findById(id).orElse(null); } }"
        
        print("\n" + "="*60)
        print("测试Token化过程")
        print("="*60 + "\n")
        
        # 编码
        token_ids, attention_mask = tokenizer.encode(test_text, max_length=64)
    
    print(f"\nToken IDs (前30个): {token_ids[:30]}")
    print(f"Attention Mask (前30个): {attention_mask[:30]}")
    
    # 解码
    decoded_text = tokenizer.decode(token_ids)
    print(f"\n解码结果:\n{decoded_text}")
    
    # 查看特定token信息
    print(f"\nToken详细信息示例:")
    for i in range(min(10, len(token_ids))):
        info = tokenizer.get_token_info(token_ids[i])
        print(f"  Position {i}: ID={info['id']}, Token='{info['token']}', "
              f"Special={info['is_special']}, Keyword={info['is_keyword']}")
