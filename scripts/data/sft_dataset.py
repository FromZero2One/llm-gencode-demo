"""
SFT 数据集 - 指令微调样本（label 掩码版，对标 minimind train_full_sft.py 的数据处理）

SFT 样本格式（jsonl，每行一条对话）：
    {"messages": [
        {"role": "system", "content": "你是编程助手"},   # 可选
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"}
    ]}

处理流程：
    1. apply_chat_template 拼模板
    2. build_labels 编码 + loss mask（只有 assistant 部分算 loss）
    3. padding 到 max_length（label 的 padding 也是 -100）

Decoder-Only 训练：forward(input_ids, labels) 直接可用。
Encoder-Decoder 不适用本数据集（SFT 是 Decoder-Only 范式，minimind 亦如此）。
"""

import os
import json
import torch
from typing import List, Dict
from torch.utils.data import Dataset

from scripts.core.chat_template import apply_chat_template, build_labels


class SFTDataset(Dataset):
    """
    SFT 指令微调数据集

    Args:
        data_path: jsonl 数据路径
        tokenizer: BPETokenizer（需已训练）
        max_length: 最大序列长度
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.vocab[tokenizer.PAD_TOKEN]

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"SFT 数据不存在: {data_path}")

        self.samples = []
        n_skipped = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    messages = obj.get('messages', [])
                    if not any(m.get('role') == 'assistant' for m in messages):
                        n_skipped += 1
                        continue
                    self.samples.append(messages)
                except json.JSONDecodeError:
                    n_skipped += 1

        if not self.samples:
            raise ValueError(f"SFT 数据为空（{n_skipped} 行无效）: {data_path}")
        print(f"[SFTDataset] 加载 {len(self.samples)} 条对话 (跳过 {n_skipped} 条无效)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        返回 (input_ids, labels)，均已 padding 到 max_length

        labels 中：assistant 回答 = token id，其余（模板/prompt/padding）= -100
        """
        messages = self.samples[idx]
        text = apply_chat_template(messages)
        ids, labels = build_labels(text, self.tokenizer, max_length=self.max_length)

        # padding（input 补 PAD，label 补 -100）
        pad_len = self.max_length - len(ids)
        ids = ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
