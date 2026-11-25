"""
DPO 训练脚本 - Direct Preference Optimization 直接偏好优化（对标 minimind train_dpo.py）

╔═══════════════════════════════════════════════════════════╗
║  📚 DPO 是什么？为什么替代 RLHF/PPO？                     ║
╚═══════════════════════════════════════════════════════════╝

SFT 之后的模型"会回答"，但不知道"哪种回答更好"。
对齐（Alignment）就是教模型人类偏好：chosen（好回答）> rejected（差回答）。

经典路线 RLHF：
    1. 训练奖励模型 RM 给回答打分
    2. PPO 强化学习让策略最大化奖励
    → 复杂：4 个模型同时在线（policy/ref/reward/value），训练不稳

DPO（Rafailov 2023）的洞察：
    奖励可以用策略与参考模型的 logp 之差隐式表示，
    于是 RL 目标可以化为一个简单的分类损失：

    L = -log σ( β·[ (logp_chosen - logp_ref_chosen)
                 - (logp_rejected - logp_ref_rejected) ] )

    只需 2 个模型（policy + 冻结的 ref，ref 常直接用 SFT 底座），
    无 RM、无采样循环，稳定性远超 PPO —— 这正是它成为当前主流的原因。

    β（beta）控制偏离参考模型的强度：太小学不动，太大遗忘 SFT。

数据格式（jsonl 每行一条偏好对）：
    {"prompt": "...", "chosen": "好回答", "rejected": "差回答"}

使用：
    python -m scripts.training.train_dpo --preset tiny --sft_checkpoint checkpoints/sft_tiny_epoch2.pt
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.core.transformer import DecoderOnlyTransformer
from scripts.core.bpe_tokenizer import BPETokenizer
from scripts.core.chat_template import apply_chat_template
from scripts.optional.training.trainer import AdamW


class DPODataset(Dataset):
    """
    DPO 偏好对数据集

    每条样本返回 (prompt_ids, chosen_ids, rejected_ids)，
    三段分开编码：prompt 部分是共享条件，两个回答分别与 prompt 拼接计分。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"DPO 数据不存在: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if obj.get('chosen') and obj.get('rejected'):
                    self.samples.append(obj)
        print(f"[DPODataset] 加载 {len(self.samples)} 条偏好对")

    def __len__(self):
        return len(self.samples)

    def _encode_pair(self, prompt_text: str, answer_text: str) -> list:
        """prompt+回答 拼模板编码（回答部分即待计分序列）"""
        text = apply_chat_template([
            {'role': 'user', 'content': prompt_text},
            {'role': 'assistant', 'content': answer_text},
        ])
        ids = self.tokenizer.tokenizer.encode(text, add_special_tokens=False).ids
        return ids[:self.max_length]

    def __getitem__(self, idx):
        obj = self.samples[idx]
        return {
            'chosen': self._encode_pair(obj['prompt'], obj['chosen']),
            'rejected': self._encode_pair(obj['prompt'], obj['rejected']),
        }


def collate_dpo(batch, pad_id: int):
    """把变长序列 pad 成 batch 张量"""
    def pad_batch(seqs):
        max_len = max(len(s) for s in seqs)
        tensors = [torch.tensor(s + [pad_id] * (max_len - len(s))) for s in seqs]
        return torch.stack(tensors)
    return pad_batch([b['chosen'] for b in batch]), pad_batch([b['rejected'] for b in batch])


def sequence_logp(model, sequences: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    计算整条序列的 log P(sequence)（对回答部分逐 token 求和）

    用 Decoder-Only 的下一 token 预测：input[:-1] 预测 labels[1:]。
    padding 位置不计。
    """
    labels = sequences.clone()
    # shift: logits[t] 预测 tokens[t+1]
    logits, _ = model(sequences[:, :-1], labels=sequences[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tgt = sequences[:, 1:]
    # 逐 token logp
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [batch, seq-1]
    # padding/EOS 之后不计
    mask = (tgt != pad_id).float()
    return (token_logp * mask).sum(dim=1)  # [batch]


def train(args):
    """DPO 主流程"""
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n{'='*60}")
    print(f"DPO 直接偏好优化 (beta={args.beta})")
    print(f"{'='*60}")

    tokenizer = BPETokenizer()
    pad_id = tokenizer.vocab[tokenizer.PAD_TOKEN]
    vocab_size = len(tokenizer.vocab)

    # 1. 数据
    data_path = args.data_path or os.path.join(PROJECT_ROOT, 'scripts', 'data', 'dpo_demo.jsonl')
    if not os.path.exists(data_path):
        generate_dpo_demo(data_path)
    dataset = DPODataset(data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=lambda b: collate_dpo(b, pad_id))

    # 2. policy 模型（要训练的）与 ref 模型（冻结参考）
    from scripts.config.presets import get_preset_config
    config = get_preset_config(args.preset)

    def build():
        m = DecoderOnlyTransformer(
            vocab_size=vocab_size, d_model=config['d_model'], nhead=config['nhead'],
            num_layers=config['num_decoder_layers'], max_seq_length=args.max_length)
        return m

    policy = build().to(device)
    if args.sft_checkpoint:
        ckpt = torch.load(args.sft_checkpoint, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        # pos_encoder.pe 是确定性正弦表，max_length 不同则形状不同，跳过让其自动重建
        if 'pos_encoder.pe' in sd and sd['pos_encoder.pe'].shape != policy.pos_encoder.pe.shape:
            full_sd = policy.state_dict()
            sd = {k: v for k, v in sd.items() if k != 'pos_encoder.pe'}
            full_sd.update(sd)
            sd = full_sd
        policy.load_state_dict(sd)
        print(f"[policy] 已加载 SFT 底座: {args.sft_checkpoint}")

    ref = build().to(device)
    ref.load_state_dict(policy.state_dict())  # ref 初始化为 policy 的副本（SFT 底座）
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False  # ref 全程冻结
    print(f"[ref] 已冻结（与 policy 初始权重相同）")

    # 3. DPO 训练循环
    optimizer = AdamW(policy.parameters(), lr=args.lr)
    history = {'dpo_loss': [], 'margin': []}

    policy.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_margin, n = 0.0, 0.0, 0
        t0 = time.time()
        for chosen_seqs, rejected_seqs in loader:
            chosen_seqs, rejected_seqs = chosen_seqs.to(device), rejected_seqs.to(device)

            with torch.no_grad():
                ref_logp_c = sequence_logp(ref, chosen_seqs, pad_id)
                ref_logp_r = sequence_logp(ref, rejected_seqs, pad_id)
            pol_logp_c = sequence_logp(policy, chosen_seqs, pad_id)
            pol_logp_r = sequence_logp(policy, rejected_seqs, pad_id)

            # DPO 损失：让 policy 相对 ref 更偏好 chosen
            logits = args.beta * ((pol_logp_c - ref_logp_c) - (pol_logp_r - ref_logp_r))
            loss = -F.logsigmoid(logits).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_margin += logits.mean().item()  # margin>0 说明偏好已被拉开
            n += 1

        print(f"[epoch {epoch+1}/{args.epochs}] dpo_loss={epoch_loss/n:.4f} "
              f"margin={epoch_margin/n:.4f} ({time.time()-t0:.1f}s)")
        history['dpo_loss'].append(epoch_loss / n)
        history['margin'].append(epoch_margin / n)

        save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        torch.save({'model_state_dict': policy.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'dpo_{args.preset}_epoch{epoch}.pt'))

    with open(os.path.join(PROJECT_ROOT, 'checkpoints', 'dpo_history.json'), 'w') as f:
        json.dump(history, f)

    # 4. 验收：margin 应为正（chosen 相对得分 > rejected）
    final_margin = history['margin'][-1]
    print(f"\n[验收] 最终 margin = {final_margin:.4f} "
          f"({'已学会偏好 chosen' if final_margin > 0 else '偏好未拉开，可增大 epochs/beta'})")


def generate_dpo_demo(output_path: str):
    """生成 demo 偏好数据（好回答 vs 差回答）"""
    pairs = [
        ("写一个 Hello World", "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}", "你好你好你好"),
        ("怎么创建列表", "List<String> items = new ArrayList<>();", "嗯这个嗯大概可以吧"),
        ("写一个保存方法", "public void save(User user) {\n    this.user = user;\n}", "好的好的知道了谢谢"),
        ("你好", "你好！我是编程助手，可以帮你写代码。", "你好"),
        ("写一个 for 循环", "for (int i = 0; i < size; i++) {\n}", "循环就是循环的意思"),
        ("怎么判断空值", "if (value == null) {\n}", "空的就是没有的"),
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt, chosen, rejected in pairs:
            f.write(json.dumps({'prompt': prompt, 'chosen': chosen, 'rejected': rejected},
                               ensure_ascii=False) + '\n')
    print(f"[dpo_demo] 偏好数据已生成: {output_path} ({len(pairs)} 条)")


def main():
    parser = argparse.ArgumentParser(description="DPO 训练脚本")
    parser.add_argument('--preset', default='tiny', type=str)
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--sft_checkpoint', default=None, type=str, help='SFT 底座（作为 ref/policy 初始）')
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_length', default=192, type=int)
    parser.add_argument('--beta', default=0.1, type=float, help='DPO 温度系数')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
