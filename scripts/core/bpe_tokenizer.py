"""
BPE 分词器 - 现代大模型标准分词方案（对标 minimind trainer/train_tokenizer.py）

╔═══════════════════════════════════════════════════════════╗
║  📚 为什么需要 BPE？(Byte Pair Encoding, 字节对编码)       ║
╚═══════════════════════════════════════════════════════════╝

【旧方案 SimpleTokenizer 的缺陷】
    正则切词 + 按频率编号，未登录词 → <UNK>。
    - "量子波动编程" → [<UNK>, <UNK>, <UNK>]：模型永远学不到没见过的词
    - 这就是本项目模型生成结果大量 <UNK> 的根本原因

【BPE 的思路】
    1. 初始词表 = 所有单字节/单字符（覆盖一切输入，永远不 OOV）
    2. 统计语料中相邻 token 对的出现频率
    3. 把最高频的对合并成新 token，加入词表
    4. 重复 2-3 直到词表达到目标大小

    效果：常见词被合并成完整 token（压缩率高），
          罕见词被拆成子词（仍然可表示）——这正是 GPT/Qwen 的做法。

【minimind 的配置（本文件对齐）】
    - 词表 6400（含 36 个特殊 token 位）
    - ByteLevel 预分词：任何字节序列都能表示（真正的"不 OOV"）
    - 中文按字拆分后逐步合并

实际应用：GPT-4 (cl100k)、Qwen、Llama 全部是 BPE 系（ByteLevel BPE 或变体）。

使用方法：
    # 训练（在语料上学习合并规则）
    python -m scripts.core.bpe_tokenizer --train --data_path corpus.txt

    # Pipeline 中使用
    pipeline = CodeGenerationPipeline(preset='small', tokenizer_type='bpe')
"""

import os
import json
import argparse
from typing import List, Tuple, Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# 项目根目录（从文件位置向上两级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BPETokenizer:
    """
    BPE 分词器 - 接口兼容 SimpleTokenizer

    ════════════════════════════════════════════════════════════
    与 SimpleTokenizer 的接口对应关系（保证 Pipeline/Trainer 零改动切换）：
        encode(text, max_length) -> (token_ids, attention_mask)   ✔ 相同
        decode(token_ids) -> str                                   ✔ 相同
        vocab / reverse_vocab / vocab_size                         ✔ 相同
        BOS_TOKEN / EOS_TOKEN / PAD_TOKEN / UNK_TOKEN              ✔ 相同
    额外能力：
        train()           在语料上训练 BPE 合并规则
        save()/load()     持久化到目录（tokenizer.json + config.json）
        char_to_token_ratio  压缩率统计（chars/token，越高说明词表学得越好）
    ════════════════════════════════════════════════════════════

    Args:
        vocab_size: 目标词表大小（默认 6400，与 minimind 一致）
        special_tokens_num: 预留特殊 token 数量（含 4 个基础位 + chat 模板角色位）
        tokenizer_dir: 持久化目录（load 时使用）
    """

    # 特殊 token（ID 顺序与 SimpleTokenizer 对齐：PAD=0, EOS=1, BOS=2, UNK=3）
    PAD_TOKEN = '<PAD>'
    EOS_TOKEN = '<EOS>'
    BOS_TOKEN = '<BOS>'
    UNK_TOKEN = '<UNK>'
    # Chat 模板角色位（P8 SFT 使用；现在预留 ID，模板文档见 scripts/core/chat_template.py）
    USER_TOKEN = '<|user|>'
    ASSISTANT_TOKEN = '<|assistant|>'
    SYSTEM_TOKEN = '<|system|>'
    IM_START_TOKEN = '<|im_start|>'
    IM_END_TOKEN = '<|im_end|>'

    def __init__(self, vocab_size: int = 6400, special_tokens_num: int = 36,
                 tokenizer_dir: Optional[str] = None, debug_mode: bool = False):
        self.vocab_size = vocab_size
        self.special_tokens_num = special_tokens_num
        self.tokenizer_dir = tokenizer_dir or os.path.join(PROJECT_ROOT, 'tokenizer_bpe')
        self.debug_mode = debug_mode

        # 特殊 token 列表：前 4 个与 SimpleTokenizer 的 ID 顺序严格对齐
        self._special_tokens = [self.PAD_TOKEN, self.EOS_TOKEN, self.BOS_TOKEN, self.UNK_TOKEN]
        # 扩展到 special_tokens_num 个（chat 角色位 + 保留位）
        extra = [self.USER_TOKEN, self.ASSISTANT_TOKEN, self.SYSTEM_TOKEN,
                 self.IM_START_TOKEN, self.IM_END_TOKEN]
        self._special_tokens += extra
        while len(self._special_tokens) < special_tokens_num:
            self._special_tokens.append(f'<|reserved_{len(self._special_tokens)}|>')

        # 尝试从磁盘加载；没有已训练的分词器则构建未训练的空壳（必须先 train 才能用）
        if os.path.exists(os.path.join(self.tokenizer_dir, 'tokenizer.json')):
            self.load()
        else:
            self._build_untrained()

    # ------------------------------------------------------------------
    # 构建 / 训练
    # ------------------------------------------------------------------
    def _build_untrained(self):
        """构建未训练的 BPE（仅初始字节词表，无合并规则）"""
        self.tokenizer = Tokenizer(models.BPE(unk_token=None))
        # ByteLevel 预分词：把文本映射到字节级表示，保证任何输入都可编码（不 OOV 的关键）
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, data_path: str):
        """
        在语料上训练 BPE 合并规则

        Args:
            data_path: 训练语料路径（.txt 每行一段，或 .jsonl 每行一个 JSON 文本字段）

        详解：
            - initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
              保证 256 个字节字符全部进初始词表（minimind 同款做法），
              这样任何未登录字符也能拆到字节级，永远不 OOV
            - min_frequency=2：出现少于 2 次的字节对不值得合并（防噪声）
        """
        # 收集语料文本行
        corpus_lines = self._read_corpus(data_path)
        if not corpus_lines:
            raise ValueError(f"语料为空或无法读取: {data_path}")

        # 临时语料文件（tokenizers 库需要文件迭代器）
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                                encoding='utf-8')
        for line in corpus_lines:
            tmp_file.write(line + '\n')
        tmp_file.close()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._special_tokens,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 字节全量打底
            show_progress=False,
        )

        self._build_untrained()
        self.tokenizer.train([tmp_file.name], trainer)
        os.unlink(tmp_file.name)

        self.save()
        print(f"[BPETokenizer] 训练完成: {len(self.tokenizer.get_vocab())} tokens "
              f"→ {self.tokenizer_dir}")

    def _read_corpus(self, data_path: str) -> List[str]:
        """
        读取语料文件

        支持：
            - .txt:  每行一段文本
            - .jsonl: 每行一个 JSON，取所有字符串字段拼接（兼容 sft_*.jsonl 格式）
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"语料不存在: {data_path}")

        lines = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                if data_path.endswith('.jsonl'):
                    try:
                        obj = json.loads(raw)
                        # 递归收集所有字符串值（兼容 {"messages": [{"content": ...}]} 嵌套）
                        texts = []
                        def _collect(x):
                            if isinstance(x, str):
                                texts.append(x)
                            elif isinstance(x, list):
                                for i in x:
                                    _collect(i)
                            elif isinstance(x, dict):
                                for v in x.values():
                                    _collect(v)
                        _collect(obj)
                        lines.extend(texts)
                    except json.JSONDecodeError:
                        continue
                else:
                    lines.append(raw)
        return lines

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self):
        """保存分词器（tokenizer.json 是 tokenizers 标准格式，config.json 记录本项目配置）"""
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(self.tokenizer_dir, 'tokenizer.json'))
        with open(os.path.join(self.tokenizer_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'special_tokens_num': self.special_tokens_num,
                'type': 'bytelevel_bpe',
                'special_tokens': self._special_tokens,
            }, f, ensure_ascii=False, indent=2)

    def load(self):
        """从磁盘加载分词器"""
        path = os.path.join(self.tokenizer_dir, 'tokenizer.json')
        self.tokenizer = Tokenizer.from_file(path)
        # 恢复配置（容忍缺省）
        cfg_path = os.path.join(self.tokenizer_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self._special_tokens = cfg.get('special_tokens', self._special_tokens)
            self.vocab_size = cfg.get('vocab_size', self.vocab_size)
        if self.debug_mode:
            print(f"[BPETokenizer] 已从 {path} 加载")

    # ------------------------------------------------------------------
    # 核心接口（与 SimpleTokenizer 对齐）
    # ------------------------------------------------------------------
    @property
    def vocab(self) -> dict:
        """token -> id 映射（兼容 SimpleTokenizer.vocab 属性）"""
        return self.tokenizer.get_vocab()

    @property
    def reverse_vocab(self) -> dict:
        """id -> token 映射"""
        return {v: k for k, v in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    def encode(self, text: str, max_length: int = 128) -> Tuple[List[int], List[int]]:
        """
        文本 → token IDs + attention mask（接口与 SimpleTokenizer.encode 完全一致）

        流程：
            1. BPE 编码（自动加 BOS/EOS，由后处理手动添加以对齐 SimpleTokenizer 行为）
            2. 截断/填充到 max_length
            3. 生成 attention mask（1=有效, 0=padding）

        Args:
            text: 输入文本
            max_length: 输出序列长度

        Returns:
            (token_ids, attention_mask)
        """
        # BPE 编码（不带特殊 token，手动加，与 SimpleTokenizer 的 [BOS]...[EOS] 行为一致）
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = [self.bos_id] + enc.ids + [self.eos_id]

        # 截断（保留 BOS，末尾放 EOS）
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length - 1] + [self.eos_id]

        # 填充到 max_length
        original_length = len(token_ids)
        pad_needed = max_length - original_length
        if pad_needed > 0:
            token_ids.extend([self.pad_id] * pad_needed)

        # attention mask：1=有效 token，0=padding（语义与 SimpleTokenizer 一致）
        attention_mask = [1] * original_length + [0] * (max_length - original_length)
        return token_ids, attention_mask

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        token IDs → 文本

        Args:
            token_ids: token 序列
            skip_special_tokens: 是否跳过特殊 token（PAD/BOS/EOS/角色位）

        Returns:
            解码后的文本
        """
        special_ids = {self.vocab[t] for t in self._special_tokens if t in self.vocab}
        ids = [i for i in token_ids
               if not (skip_special_tokens and i in special_ids)]
        return self.tokenizer.decode(ids)

    def char_to_token_ratio(self, texts: List[str]) -> float:
        """
        压缩率统计：平均每个 token 对应多少字符

        压缩率越高 → 词表学习得越好（常见词合并越充分）。
        教学参考：英文 GPT-4 约 4.4，中文 Qwen 约 1.5-2.0。
        """
        total_chars = sum(len(t) for t in texts)
        total_tokens = sum(len(self.tokenizer.encode(t).ids) for t in texts)
        return total_chars / total_tokens if total_tokens else 0.0


# ══════════════════════════════════════════════════════════════════
# 自验证区
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BPE 分词器训练与验证")
    parser.add_argument('--train', action='store_true', help='训练 BPE')
    parser.add_argument('--data_path', default=None, type=str, help='训练语料路径')
    parser.add_argument('--vocab_size', default=6400, type=int, help='目标词表大小')
    args = parser.parse_args()

    if args.train:
        data_path = args.data_path or os.path.join(PROJECT_ROOT, 'scripts', 'data',
                                                   'corpus_demo.txt')
        if not os.path.exists(data_path):
            # 语料不存在时先生成 demo 语料（P7 会正式提供，这里内置最小版保证自洽）
            from scripts.data.corpus_demo import generate_demo_corpus
            generate_demo_corpus(data_path)
        tok = BPETokenizer(vocab_size=args.vocab_size)
        tok.train(data_path)

    # ===== 验证流程 =====
    tok = BPETokenizer(debug_mode=True)
    if not hasattr(tok, 'tokenizer') or len(tok.tokenizer.get_vocab()) <= 300:
        print("[BPETokenizer] 尚未训练（找不到 tokenizer.json），先训练一个 demo 分词器...")
        data_path = os.path.join(PROJECT_ROOT, 'scripts', 'data', 'corpus_demo.txt')
        if not os.path.exists(data_path):
            from scripts.data.corpus_demo import generate_demo_corpus
            generate_demo_corpus(data_path)
        tok = BPETokenizer(vocab_size=args.vocab_size)
        tok.train(data_path)

    print("\n" + "=" * 60)
    print("BPE 分词器验证")
    print("=" * 60)

    # 1. 特殊 token ID 对齐验证（与 SimpleTokenizer 一致）
    assert tok.vocab[tok.PAD_TOKEN] == 0, "PAD 应为 ID 0"
    assert tok.vocab[tok.EOS_TOKEN] == 1, "EOS 应为 ID 1"
    assert tok.vocab[tok.BOS_TOKEN] == 2, "BOS 应为 ID 2"
    assert tok.vocab[tok.UNK_TOKEN] == 3, "UNK 应为 ID 3"
    print(f"[OK] 特殊 token ID 对齐: PAD=0, EOS=1, BOS=2, UNK=3")

    # 2. 编解码 roundtrip（正常文本）
    for text in ["public class UserService {",
                 "创建一个用户服务类",
                 "def fibonacci(n): return n"]:
        ids, mask = tok.encode(text, max_length=32)
        decoded = tok.decode(ids)
        assert decoded == text, f"roundtrip 失败: {text!r} -> {decoded!r}"
    print(f"[OK] 编解码 roundtrip 3 组全部无损")

    # 3. 生造词不再 <UNK>（BPE 核心价值！）
    novel = "量子波动编程法"
    ids, _ = tok.encode(novel, max_length=16)
    decoded = tok.decode(ids)
    assert decoded == novel, f"生造词 roundtrip 失败: {decoded!r}"
    print(f"[OK] 生造词 '{novel}' 编码为 {len(ids)} 个 token，无 <UNK>（旧方案全是 UNK）")

    # 4. 填充与截断
    ids, mask = tok.encode("public class", max_length=10)
    assert len(ids) == len(mask) == 10
    assert 0 in mask, "短文本应有 padding"
    ids, mask = tok.encode("x " * 50, max_length=10)
    assert ids[-1] == tok.eos_id, "长文本截断后应以 EOS 结尾"
    print(f"[OK] 填充/截断行为与 SimpleTokenizer 一致")

    # 5. 压缩率统计
    samples = ["public class UserService {",
               "public static void main(String[] args)",
               "创建一个用户服务类包含增删改查方法"]
    ratio = tok.char_to_token_ratio(samples)
    print(f"[OK] 压缩率: {ratio:.2f} chars/token（词表越大该值越高）")

    print(f"\n词表大小: {len(tok.vocab)}")
    print("[OK] BPE 分词器全部验证通过")
