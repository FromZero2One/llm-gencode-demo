"""
LoRA 微调训练脚本（对标 minimind trainer/train_lora.py）

与全参数 SFT 的区别：
    - 底座模型完全冻结（本实现额外冻结 embedding/FFN/norm，只训 LoRA 参数）
    - 只保存 LoRA 权重（几十 KB vs 完整模型几 MB）
    - 同一个底座 + 不同 LoRA = 不同行为的模型（"插件式"定制）

使用方法：
    # 在预训练/SFT 底座上做 LoRA
    python -m scripts.training.train_lora --preset tiny --base_checkpoint checkpoints/sft_tiny_epoch2.pt

    # 合并 LoRA 到底座（部署形态）
    python -m scripts.training.train_lora --merge_only --lora_path checkpoints/lora_tiny.pt ...
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
from scripts.core.lora import apply_lora, save_lora, load_lora, merge_lora
from scripts.data.sft_dataset import SFTDataset
from scripts.optional.training.trainer import AdamW, WarmupLinearScheduler
from scripts.training.train_full_sft import sft_forward_loss, demo_chat


def train(args):
    """LoRA 微调主流程"""
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n{'='*60}")
    print(f"LoRA 微调 (r={args.r}, alpha={args.alpha})")
    print(f"  preset: {args.preset} | device: {device} | epochs: {args.epochs}")
    print(f"{'='*60}")

    tokenizer = BPETokenizer()
    vocab_size = len(tokenizer.vocab)

    # 1. 数据
    data_path = args.data_path or os.path.join(PROJECT_ROOT, 'scripts', 'data', 'sft_demo.jsonl')
    if not os.path.exists(data_path):
        from scripts.data.sft_demo import generate_sft_demo
        generate_sft_demo(data_path)
    dataset = SFTDataset(data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 2. 底座模型（加载 checkpoint 并整体冻结）
    from scripts.config.presets import get_preset_config
    config = get_preset_config(args.preset)
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size, d_model=config['d_model'], nhead=config['nhead'],
        num_layers=config['num_decoder_layers'], max_seq_length=args.max_length,
    ).to(device)

    if args.base_checkpoint:
        ckpt = torch.load(args.base_checkpoint, map_location='cpu')
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"[底座] 已加载: {args.base_checkpoint}")
    else:
        print(f"[底座] 随机初始化（建议传入 --base_checkpoint）")

    # 3. 整体冻结 → 套 LoRA（只对注意力投影）
    for p in model.parameters():
        p.requires_grad = False
    apply_lora(model, target_modules=['W_q', 'W_k', 'W_v', 'W_o'],
               r=args.r, alpha=args.alpha)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[冻结] 可训练参数: {n_train:,}/{n_total:,} ({n_train/n_total*100:.1f}%)")

    # 4. 训练（只优化 LoRA 参数）
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=0.01)
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=int(len(loader)*args.epochs*0.1),
                                      total_steps=len(loader)*args.epochs)

    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_items = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(loader):
            loss, n_items = sft_forward_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * n_items
            epoch_items += n_items

        avg = epoch_loss / epoch_items
        print(f"[epoch {epoch+1}/{args.epochs}] lora_loss={avg:.4f} ({time.time()-t0:.1f}s)")

        lora_path = os.path.join(save_dir, f'lora_{args.preset}_epoch{epoch}.pt')
        save_lora(model, lora_path)

    # 5. 演示：加载 LoRA 后对话
    print(f"\n[演示] 使用最终 LoRA:")
    demo_chat(model, tokenizer, args, device)

    # 6. 可选：合并到底座并保存完整模型
    if args.merge:
        merge_lora(model)
        merged_path = os.path.join(save_dir, f'merged_{args.preset}.pt')
        torch.save({'model_state_dict': model.state_dict()}, merged_path)
        print(f"[合并] 完整模型 → {merged_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调脚本")
    parser.add_argument('--preset', default='tiny', type=str)
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--base_checkpoint', default=None, type=str, help='底座模型 checkpoint')
    parser.add_argument('--r', default=8, type=int, help='LoRA 秩')
    parser.add_argument('--alpha', default=16, type=int, help='LoRA 缩放')
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--gen_length', default=40, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--merge', action='store_true', help='训练后合并 LoRA 到底座')
    parser.add_argument('--demo_questions', default='你好|写一个 Hello World', type=str)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
