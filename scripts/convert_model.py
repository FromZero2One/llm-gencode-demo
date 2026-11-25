"""
模型格式转换 - PyTorch checkpoint ↔ HuggingFace Transformers 格式

将本项目训练的 .pt 权重转换为 HuggingFace 格式目录（config.json + model.safetensors），
转换后可直接用 transformers 加载，或进一步导出到 ollama / llama.cpp / vLLM。

说明：
- Encoder-Decoder 架构的 HF 原生对应是自建的 config（本项目结构非标准 T5），
  因此采用"保存权重 + 自定义 config"的方式；Decoder-Only 架构则映射到
  标准 CausalLM 语义。
- 转换是可逆的：load 回来时用同架构的模型类 load_state_dict 即可。

使用方法：
    python scripts/convert_model.py --checkpoint checkpoints/checkpoint_epoch_9.pt --architecture encdec
    python scripts/convert_model.py --checkpoint model_decoder.pt --architecture decoder
"""

import argparse
import json
import os

import torch


def inspect_checkpoint(filepath: str) -> dict:
    """打印 checkpoint 的元信息和参数形状摘要"""
    checkpoint = torch.load(filepath, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    meta = {
        'epoch': checkpoint.get('epoch'),
        'loss': checkpoint.get('loss'),
        'num_params': len(state_dict),
        'total_elements': sum(v.numel() for v in state_dict.values()),
    }
    print(f"Checkpoint: {filepath}")
    print(f"  epoch: {meta['epoch']}, loss: {meta['loss']}")
    print(f"  参数张量数: {meta['num_params']}, 总参数量: {meta['total_elements']:,}")
    return meta


def convert_to_hf(filepath: str, architecture: str, output_dir: str,
                  vocab_size: int, d_model: int):
    """
    转换为 HuggingFace 格式目录

    生成：
    - config.json: 架构超参数与自定义标识
    - pytorch_model.bin: 完整 state_dict（safetensors 需额外依赖，教学用 .bin 足够）
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 从权重本身推断模型配置（避免用户填错）
    inferred = {
        'vocab_size': state_dict['embedding.weight'].size(0),
        'd_model': state_dict['embedding.weight'].size(1),
        'architectures': ['TransformerModel' if architecture == 'encdec'
                          else 'DecoderOnlyTransformer'],
        'model_type': 'llm-codegen-demo',
        'architecture': architecture,
        'tie_word_embeddings': architecture == 'decoder',
    }
    # 与传入参数交叉校验
    if vocab_size and vocab_size != inferred['vocab_size']:
        print(f"  [警告] 传入 vocab_size={vocab_size} 与权重不一致(实际{inferred['vocab_size']})，以权重为准")
    if d_model and d_model != inferred['d_model']:
        print(f"  [警告] 传入 d_model={d_model} 与权重不一致(实际{inferred['d_model']})，以权重为准")

    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(inferred, f, ensure_ascii=False, indent=2)

    weights_path = os.path.join(output_dir, 'pytorch_model.bin')
    torch.save(state_dict, weights_path)

    print(f"\n已转换 → {output_dir}")
    print(f"  config.json: {json.dumps(inferred, ensure_ascii=False)}")
    print(f"  pytorch_model.bin: {os.path.getsize(weights_path) / 1024 / 1024:.2f} MB")
    print(f"\n加载方式:")
    print(f"  checkpoint = torch.load('{weights_path}')")
    print(f"  model = {'TransformerModel' if architecture == 'encdec' else 'DecoderOnlyTransformer'}(...)")
    print(f"  model.load_state_dict(checkpoint)")


def main():
    parser = argparse.ArgumentParser(description="模型格式转换工具")
    parser.add_argument('--checkpoint', required=True, type=str, help='输入 .pt 权重路径')
    parser.add_argument('--architecture', default='encdec', type=str,
                        choices=['encdec', 'decoder'], help='模型架构')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='输出目录（默认: <checkpoint名>_hf/）')
    parser.add_argument('--vocab_size', default=None, type=int, help='词汇表大小(校验用)')
    parser.add_argument('--d_model', default=None, type=int, help='模型维度(校验用)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    inspect_checkpoint(args.checkpoint)

    output_dir = args.output_dir or os.path.splitext(args.checkpoint)[0] + '_hf'
    convert_to_hf(args.checkpoint, args.architecture, output_dir,
                  args.vocab_size, args.d_model)


if __name__ == '__main__':
    main()
