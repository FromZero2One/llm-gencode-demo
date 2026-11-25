"""
预训练数据集 - 把原始语料转成自回归训练样本（对标 minimind dataset/lm_dataset.py）

╔═══════════════════════════════════════════════════════════╗
║  📚 预训练样本是怎么构造的？                              ║
╚═══════════════════════════════════════════════════════════╝

给定一段文本的 token 序列：
    [t1, t2, t3, t4, t5]

自回归语言建模的目标是"预测下一个 token"：
    input:  [t1, t2, t3, t4]
    label:  [t2, t3, t4, t5]
            ↑ label 只是 input 右移一位

两种实现方式：
    1. TextDataset（旧）：每条文本独立成样本，短文本浪费严重
    2. PretrainDataset（本文件，minimind 做法）：
       先把所有文本拼接成长流，再按 max_length 滑窗切块 →
       每个样本都是满长度，训练效率高、上下文连续

 labels 的特殊处理：
    - padding 位置 label = -100（CrossEntropy 的 ignore_index，不计 loss）
    - Encoder-Decoder 训练时 label 即 decoder 输入右移一位
"""

import os
import json
import torch
from typing import List
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集 - 滑窗切块的自回归样本

    处理流程：
        1. 读取语料（txt 每行一段 / jsonl）
        2. BPE 编码每行并拼接成一条长 token 流（行间插入换行）
        3. 按 max_length 滑窗切块（步长 = max_length，不重叠）
        4. 每个块构造 (input, label) 对：label = input 右移一位，
           最后一个 label 用 EOS（或 PAD=-100）填充

    Args:
        data_path: 语料路径（.txt 或 .jsonl）
        tokenizer: BPETokenizer 实例
        max_length: 样本序列长度（即训练上下文长度）
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 读取所有文本行
        lines = self._read_lines(data_path)
        if not lines:
            raise ValueError(f"语料为空: {data_path}")

        # 2. 全部编码并拼接成一条长流
        pad_id = tokenizer.vocab[tokenizer.PAD_TOKEN]
        eos_id = tokenizer.vocab[tokenizer.EOS_TOKEN]
        bos_id = tokenizer.vocab[tokenizer.BOS_TOKEN]

        stream: List[int] = []
        for line in lines:
            ids, _ = tokenizer.encode(line, max_length=max_length)  # 带 BOS/EOS
            stream.extend(ids)

        # 3. 滑窗切块（不重叠），丢弃不足 2 个 token 的尾巴
        n_chunks = len(stream) // max_length
        self.samples = []
        for i in range(n_chunks):
            chunk = stream[i * max_length:(i + 1) * max_length]
            self.samples.append(chunk)

        print(f"[PretrainDataset] 语料 {data_path}")
        print(f"  - 文本行数: {len(lines)}, 总 token 数: {len(stream):,}")
        print(f"  - 样本数: {len(self.samples)} (每样本 {max_length} tokens)")

    def _read_lines(self, data_path: str) -> List[str]:
        """读取 txt/jsonl 语料行（复用 BPETokenizer 的解析逻辑，jsonl 展开所有字符串字段）"""
        from scripts.core.bpe_tokenizer import BPETokenizer
        # 直接复用 BPETokenizer._read_corpus 的解析（它是实例方法但只依赖参数）
        tmp = BPETokenizer.__new__(BPETokenizer)  # 不走 __init__，避免加载分词器
        return tmp._read_corpus(data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        返回 (input_ids, labels)

        labels = input 右移一位 + 末尾 EOS：
            input: [t1, t2, t3, t4]
            label: [t2, t3, t4, EOS]
        Encoder-Decoder 训练时 inputs 作 src、labels 作 tgt（teacher forcing）；
        Decoder-Only 训练时把 labels 同时喂给 forward(labels=...) 即可。
        """
        chunk = self.samples[idx]
        eos_id = self.tokenizer.vocab[self.tokenizer.EOS_TOKEN]

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:] + [eos_id], dtype=torch.long)
        return input_ids, labels
