"""
SFT 训练脚本 - 指令微调（对标 minimind trainer/train_full_sft.py）

╔═══════════════════════════════════════════════════════════╗
║  📚 SFT 在 LLM 流程中的位置                               ║
╚═══════════════════════════════════════════════════════════╝

    预训练 (train_pretrain) → SFT (本脚本) → DPO 对齐 (P10)
    "学会语言规律"             "学会听指令回答"

SFT = Supervised Fine-Tuning（监督微调）：
    在"指令 → 回答"格式的对话数据上继续训练预训练模型，
    只对回答部分计算 loss（loss mask 见 scripts/core/chat_template.py）。

训练后模型行为变化：
    - 预训练模型：续写任何文本（你问它问题，它可能继续"写问题"）
    - SFT 模型：  看到模板里 <|im_start|>assistant 后开始"回答"

典型配置（minimind mini 档）：lr 1e-4 ~ 5e-5（比预训练小一个量级，避免灾难性遗忘）、
epochs 1~3。

使用方法：
    python -m scripts.training.train_full_sft --preset tiny
    python -m scripts.training.train_full_sft --pretrain_checkpoint checkpoints/pretrain_tiny_decoder_epoch2.pt
"""

import os
import sys
import json
import argparse
import time

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.core.transformer import DecoderOnlyTransformer
from scripts.core.bpe_tokenizer import BPETokenizer
from scripts.data.sft_dataset import SFTDataset
from scripts.optional.training.trainer import AdamW, WarmupLinearScheduler


def sft_forward_loss(model, batch, device):
    """
    SFT 前向 + loss

    Decoder-Only: input_ids 与 labels 同时喂给 forward，
    labels 里的 -100（prompt/padding 位置）在 CrossEntropyLoss 中被忽略。
    """
    input_ids, labels = [t.to(device) for t in batch]
    logits, loss = model(input_ids, labels=labels)
    return loss, input_ids.size(0)


def train(args):
    """SFT 主流程"""
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n{'='*60}")
    print(f"SFT 指令微调")
    print(f"  preset: {args.preset} | device: {device} | epochs: {args.epochs}")
    print(f"{'='*60}")

    # 1. 分词器（BPE，chat 模板依赖其特殊 token）
    tokenizer = BPETokenizer()
    vocab_size = len(tokenizer.vocab)
    print(f"\n[1] BPE 词表大小: {vocab_size}")

    # 2. SFT 数据
    data_path = args.data_path or os.path.join(PROJECT_ROOT, 'scripts', 'data', 'sft_demo.jsonl')
    if not os.path.exists(data_path):
        from scripts.data.sft_demo import generate_sft_demo
        generate_sft_demo(data_path)
    dataset = SFTDataset(data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 3. 模型（Decoder-Only）+ 可选加载预训练权重
    from scripts.config.presets import get_preset_config
    config = get_preset_config(args.preset)
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_decoder_layers'],
        max_seq_length=args.max_length,
    ).to(device)

    if args.pretrain_checkpoint:
        ckpt = torch.load(args.pretrain_checkpoint, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        # pos_encoder.pe 是确定性正弦表，其形状取决于预训练时的 max_length；
        # 若与本次 SFT 的 max_length 不同，跳过加载（模型会按新长度自动重建）
        pe_key = 'pos_encoder.pe'
        if pe_key in state_dict and state_dict[pe_key].shape != model.pos_encoder.pe.shape:
            print(f"[加载] 跳过 {pe_key} (预训练 {tuple(state_dict[pe_key].shape[:2])} "
                  f"≠ 当前 {tuple(model.pos_encoder.pe.shape[:2])})，正弦表按新长度重建")
            state_dict = {k: v for k, v in state_dict.items() if k != pe_key}
            # 缺 pe 时：先取当前模型完整 state_dict 再覆盖（pe 按新 max_length 自动重建）
            full_sd = model.state_dict()
            full_sd.update(state_dict)
            model.load_state_dict(full_sd)
        else:
            model.load_state_dict(state_dict)
        print(f"[2] 已加载预训练权重: {args.pretrain_checkpoint} "
              f"(epoch={ckpt.get('epoch')}, loss={ckpt.get('loss')})")
    else:
        print(f"[2] 从随机初始化开始（建议先跑 train_pretrain）")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    模型参数量: {n_params:,}")

    # 4. 优化器（SFT 用更小学习率）
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=int(total_steps * 0.1),
                                      total_steps=total_steps)

    # 5. 训练循环
    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    history = {'sft_loss': []}

    model.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_items = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(loader):
            loss, n_items = sft_forward_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * n_items
            epoch_items += n_items
            if (step + 1) % 10 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(loader)} "
                      f"loss={epoch_loss/epoch_items:.4f}")

        avg_loss = epoch_loss / epoch_items
        history['sft_loss'].append(avg_loss)
        print(f"[epoch {epoch+1}/{args.epochs}] sft_loss={avg_loss:.4f} ({time.time()-t0:.1f}s)")

        ckpt_path = os.path.join(save_dir, f'sft_{args.preset}_epoch{epoch}.pt')
        torch.save({'model_state_dict': model.state_dict(),
                    'epoch': epoch, 'loss': avg_loss}, ckpt_path)
        print(f"  checkpoint → {ckpt_path}")

    with open(os.path.join(save_dir, 'sft_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f)

    # 6. 训练后对话演示（走真实 chat template）
    demo_chat(model, tokenizer, args, device)


@torch.no_grad()
def demo_chat(model, tokenizer, args, device):
    """SFT 后的对话演示：模板 → 贪心生成 → 停在 <|im_end|>"""
    from scripts.core.chat_template import apply_chat_template
    from scripts.core.chat_template import IM_START, IM_END

    # Windows GBK 控制台无法打印部分生成字符，用 replace 防崩溃
    safe_print = lambda s: print(s.encode('gbk', errors='replace').decode('gbk'))

    model.eval()
    print(f"\n{'='*60}")
    print("SFT 后对话演示")
    print(f"{'='*60}")

    im_start_id = tokenizer.vocab[IM_START]
    im_end_id = tokenizer.vocab[IM_END]

    for question in args.demo_questions.split('|'):
        messages = [{'role': 'user', 'content': question}]
        prompt = apply_chat_template(messages, add_generation_prompt=True)
        ids = tokenizer.tokenizer.encode(prompt, add_special_tokens=False).ids
        generated = torch.tensor([ids], device=device)

        # 贪心生成，直到 <|im_end|> 或达到长度上限
        for _ in range(args.gen_length):
            logits, _ = model.generate_step(None, generated)
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if next_tok.item() == im_end_id:
                break

        # 提取回答部分（生成 prompt 之后的 token）
        answer_ids = generated[0].tolist()[len(ids):]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        safe_print(f"\n  问: {question}")
        safe_print(f"  答: {answer.strip()[:200]}")


def main():
    parser = argparse.ArgumentParser(description="SFT 指令微调脚本")
    parser.add_argument('--preset', default='tiny', type=str,
                        choices=['tiny', 'small', 'medium'])
    parser.add_argument('--data_path', default=None, type=str, help='SFT jsonl 路径')
    parser.add_argument('--pretrain_checkpoint', default=None, type=str,
                        help='预训练 checkpoint（建议传入）')
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='SFT 学习率（比预训练小）')
    parser.add_argument('--gen_length', default=40, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--demo_questions', default='你好|写一个 Hello World|怎么创建一个列表',
                        type=str, help='演示问题，| 分隔')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
