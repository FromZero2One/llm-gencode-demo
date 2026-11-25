"""
预训练训练脚本 - 从零训练语言模型（对标 minimind trainer/train_pretrain.py）

╔═══════════════════════════════════════════════════════════╗
║  📚 预训练在 LLM 流程中的位置                             ║
╚═══════════════════════════════════════════════════════════╝

    预训练 (本脚本)  →  SFT 指令微调 (P8)  →  DPO 对齐 (P10)
    "学会语言规律"      "学会听指令回答"      "学会人类偏好"

预训练任务：给定前文预测下一个 token（自回归语言建模）。
模型没有任何先验知识，全靠语料统计规律学习。

梯度累积（Gradient Accumulation）：
    显存不够大 batch 时，把多个小 batch 的梯度累加后再更新一次：
        effective_batch = batch_size × accumulation_steps
    这是单卡训练大模型的标准技巧（minimind 同款）。

使用方法：
    python -m scripts.training.train_pretrain --preset tiny --epochs 3
    python -m scripts.training.train_pretrain --preset small --data_path my_corpus.txt
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

from scripts.core.transformer import TransformerModel, DecoderOnlyTransformer
from scripts.core.bpe_tokenizer import BPETokenizer
from scripts.data.lm_dataset import PretrainDataset
from scripts.optional.training.trainer import CrossEntropyLoss, AdamW, WarmupLinearScheduler


def build_model(args, vocab_size: int):
    """按架构和 preset 构建模型（vocab_size 以 BPE 词表为准）"""
    from scripts.config.presets import get_preset_config
    config = get_preset_config(args.preset)

    if args.architecture == 'decoder':
        return DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_decoder_layers'],
            max_seq_length=args.max_length,
        )
    return TransformerModel(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
    )


def forward_loss(model, batch, architecture: str, loss_fn):
    """
    单 batch 前向 + loss（按架构分发）

    Returns:
        (loss, num_items) num_items 用于梯度累积的加权平均
    """
    inputs, targets = batch

    if architecture == 'decoder':
        # Decoder-Only: forward(labels=) 内部右移对齐并计算 loss
        logits, loss = model(targets, labels=targets)
        return loss, inputs.size(0)
    else:
        # Encoder-Decoder: src=inputs, tgt=targets（teacher forcing）
        tgt_mask = model.generate_square_subsequent_mask(targets.size(1))
        logits, _, _ = model(inputs, targets, tgt_mask=tgt_mask)
        # labels 中 -100 的位置不计 loss（CrossEntropy ignore_index）
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, inputs.size(0)


def train(args):
    """预训练主流程"""
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto'
                          else args.device)
    print(f"\n{'='*60}")
    print(f"预训练配置")
    print(f"  preset: {args.preset} | architecture: {args.architecture}")
    print(f"  device: {device} | epochs: {args.epochs} | batch: {args.batch_size}×{args.accumulation_steps}")
    print(f"{'='*60}")

    # 1. 分词器（必须用已训练的 BPE）
    tokenizer = BPETokenizer()
    vocab_size = len(tokenizer.vocab)
    print(f"\n[1] BPE 词表大小: {vocab_size}")

    # 2. 数据集
    data_path = args.data_path or os.path.join(PROJECT_ROOT, 'scripts', 'data', 'corpus_demo.txt')
    if not os.path.exists(data_path):
        from scripts.data.corpus_demo import generate_demo_corpus
        generate_demo_corpus(data_path)
    dataset = PretrainDataset(data_path, tokenizer, max_length=args.max_length)
    if args.max_samples and args.max_samples < len(dataset):
        # 快速验证用：截取部分样本
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"  [max_samples] 截取前 {len(dataset)} 个样本")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 3. 模型 / 优化器 / 调度器 / 损失
    model = build_model(args, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] 模型参数量: {n_params:,} ({args.architecture})")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) // args.accumulation_steps) * args.epochs
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=int(total_steps * 0.1),
                                      total_steps=total_steps)
    loss_fn = CrossEntropyLoss(ignore_index=-100)

    # 4. 训练循环（含梯度累积）
    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    history = {'train_loss': [], 'lr': []}

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss, epoch_items = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(loader):
            batch = [t.to(device) for t in batch]
            loss, n_items = forward_loss(model, batch, args.architecture, loss_fn)

            # 梯度累积：loss 除以累积步数，梯度先不清零
            (loss / args.accumulation_steps).backward()

            epoch_loss += loss.item() * n_items
            epoch_items += n_items

            # 达到累积步数 → 执行一次真正的参数更新
            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg = epoch_loss / epoch_items
                    print(f"  epoch {epoch+1} step {global_step}/{total_steps} "
                          f"loss={avg:.4f} lr={scheduler.optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = epoch_loss / epoch_items
        history['train_loss'].append(avg_loss)
        history['lr'].append(scheduler.optimizer.param_groups[0]['lr'])
        print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f} "
              f"({time.time()-t0:.1f}s)")

        # 5. 每个 epoch 存 checkpoint（兼容 Pipeline.load_checkpoint 格式）
        ckpt_path = os.path.join(save_dir, f'pretrain_{args.preset}_{args.architecture}_epoch{epoch}.pt')
        torch.save({'model_state_dict': model.state_dict(),
                    'epoch': epoch, 'loss': avg_loss}, ckpt_path)
        print(f"  checkpoint → {ckpt_path}")

    # 6. 保存训练曲线数据
    history_path = os.path.join(save_dir, f'pretrain_history_{args.preset}_{args.architecture}.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f)
    print(f"\n[完成] loss 曲线: {history_path}")

    # 7. 训练后即时推理对比（训练前 vs 训练后）
    demo_inference(model, tokenizer, args, device)


@torch.no_grad()
def demo_inference(model, tokenizer, args, device):
    """用训练好的模型做贪心续写，直观看到预训练效果"""
    model.eval()
    print(f"\n{'='*60}")
    print("训练后贪心续写演示")
    print(f"{'='*60}")

    for prompt_text in ["public class UserService", "创建一个"]:
        if args.architecture == 'decoder':
            # Decoder-Only: prompt 拼进生成序列
            ids, _ = tokenizer.encode(prompt_text, max_length=args.max_length)
            generated = torch.tensor([ids], device=device)
            for _ in range(args.gen_length):
                logits, _ = model.generate_step(None, generated)
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)
            out_ids = generated[0].tolist()
        else:
            # Encoder-Decoder: prompt 喂 encoder，从 BOS 生成
            src_ids, _ = tokenizer.encode(prompt_text, max_length=args.max_length)
            src = torch.tensor([src_ids], device=device)
            generated = torch.tensor([[tokenizer.vocab[tokenizer.BOS_TOKEN]]], device=device)
            for _ in range(args.gen_length):
                logits, _ = model.generate_step(src, generated)
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)
            out_ids = generated[0].tolist()

        text = tokenizer.decode(out_ids)
        print(f"\n  prompt: {prompt_text}")
        print(f"  续写:   {text[:150]}")


def main():
    parser = argparse.ArgumentParser(description="预训练脚本")
    parser.add_argument('--preset', default='tiny', type=str,
                        choices=['tiny', 'small', 'medium'])
    parser.add_argument('--architecture', default='encdec', type=str,
                        choices=['encdec', 'decoder'])
    parser.add_argument('--data_path', default=None, type=str, help='语料路径（默认 demo 语料）')
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--accumulation_steps', default=2, type=int,
                        help='梯度累积步数（有效 batch = batch_size × 此值）')
    parser.add_argument('--max_length', default=128, type=int, help='序列长度/上下文窗口')
    parser.add_argument('--max_samples', default=None, type=int,
                        help='截取样本数（快速验证用，None=全量）')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gen_length', default=30, type=int, help='演示续写长度')
    parser.add_argument('--device', default='cpu', type=str, help="'cpu' | 'cuda' | 'auto'")
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
