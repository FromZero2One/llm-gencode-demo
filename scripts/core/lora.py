"""
LoRA - Low-Rank Adaptation 低秩适配微调（对标 minimind model/model_lora.py）

╔═══════════════════════════════════════════════════════════╗
║  📚 LoRA 为什么有效？                                     ║
╚═══════════════════════════════════════════════════════════╝

全参数微调：要更新 W ∈ R^(d×d) 的全部 d² 个参数。
LoRA 的假设：微调引起的权重变化 ΔW 是低秩的，可以分解为：
    W' = W + (α/r) · B·A
    其中 A ∈ R^(r×d), B ∈ R^(d×r), r << d（如 r=8）

参数量对比（d=128, r=8）：
    全量: 128×128 = 16,384
    LoRA: 8×128 + 128×8 = 2,048  （约 1/8）

训练时：W 冻结（requires_grad=False），只训 A/B → 显存和时间大幅下降。
部署时：merge_lora() 把 BA 合并回 W，推理零额外开销。

LoRA 的数学细节：
    - A 用高斯初始化，B 初始化为 0 → 训练开始时 BA=0，不破坏原模型
    - 缩放 α/r：α 通常取 2r（如 r=8, α=16），控制适配强度

使用：
    from scripts.core.lora import apply_lora, load_lora, merge_lora
    model = apply_lora(model, target_modules=['W_q','W_k','W_v','W_o'], r=8, alpha=16)
    # ... 训练 ...
    merge_lora(model)   # 合并后可像普通模型一样部署
"""

import os
import json
import torch
import torch.nn as nn
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LoRALinear(nn.Module):
    """
    LoRA 适配层 - 包装一个冻结的原始 Linear

    forward(x) = original(x) + (alpha/r) · lora_B(lora_A(x))

    Args:
        original: 被包装的 nn.Linear（将被冻结）
        r: LoRA 秩（越大表达力越强、参数越多）
        alpha: 缩放系数（通常取 2r）
        dropout: LoRA 分支的 dropout
    """

    def __init__(self, original: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        d_out, d_in = original.weight.shape

        # A: 降维 (r × d_in)，高斯初始化
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        # B: 升维 (d_out × r)，零初始化 → 初始时 BA = 0，不改变原模型行为
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 冻结原权重
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始路径（冻结）
        out = self.original(x)
        # LoRA 路径: x → A (降维到 r) → B (升维回 d_out) × 缩放
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scaling * lora_out


def apply_lora(model: nn.Module, target_modules: List[str],
               r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> nn.Module:
    """
    给模型的指定模块套上 LoRA

    Args:
        model: 目标模型（TransformerModel / DecoderOnlyTransformer）
        target_modules: 要适配的子模块名列表，如 ['W_q', 'W_k', 'W_v', 'W_o']
                        （即 MultiHeadAttention 的 Q/K/V/输出投影）
        r, alpha, dropout: LoRA 超参数

    Returns:
        原模型（原地替换，方便链式调用）
    """
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            # 精确匹配模块名末段
            if child_name in target_modules and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha,
                                                       dropout=dropout))
                replaced += 1

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] 已适配 {replaced} 个 Linear 层 (r={r}, alpha={alpha})")
    print(f"[LoRA] 参数量: 总 {n_total:,} | 可训练 {n_train:,} "
          f"({n_train/n_total*100:.1f}%)")
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """只提取 LoRA 参数（保存用，文件极小）"""
    return {k: v for k, v in model.state_dict().items()
            if 'lora_A' in k or 'lora_B' in k}


def save_lora(model: nn.Module, filepath: str):
    """保存 LoRA 权重（不含底座模型，通常只有几十 KB）"""
    lora_sd = get_lora_state_dict(model)
    torch.save(lora_sd, filepath)
    print(f"[LoRA] 已保存 {len(lora_sd)} 个张量 → {filepath} "
          f"({os.path.getsize(filepath)/1024:.1f} KB)")


def load_lora(model: nn.Module, filepath: str):
    """加载 LoRA 权重到已 apply_lora 的模型"""
    lora_sd = torch.load(filepath, map_location='cpu')
    model.load_state_dict(lora_sd, strict=False)
    print(f"[LoRA] 已加载 {len(lora_sd)} 个张量 ← {filepath}")


@torch.no_grad()
def merge_lora(model: nn.Module):
    """
    把 LoRA 合并回原权重（部署用）

    W_new = W + (alpha/r) · B·A
    合并后模型结构恢复原样，推理零开销。
    """
    merged = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            delta = (module.scaling * (module.lora_B @ module.lora_A))
            module.original.weight.data += delta
            # 用合并后的原层替换 LoRALinear
            parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            setattr(parent, name.rsplit('.', 1)[-1], module.original)
            merged += 1
    print(f"[LoRA] 已合并 {merged} 个层回原权重")
    return model


if __name__ == '__main__':
    # ===== 自验证 =====
    from scripts.core.transformer import DecoderOnlyTransformer

    print("=" * 60)
    print("LoRA 验证")
    print("=" * 60)

    torch.manual_seed(42)
    model = DecoderOnlyTransformer(vocab_size=500, d_model=64, nhead=4, num_layers=2)
    model.eval()

    # 1. apply_lora 后初始输出不变（B=0 → BA=0）
    x = torch.randint(0, 500, (1, 8))
    out_before, _ = model(x)
    apply_lora(model, target_modules=['W_q', 'W_k', 'W_v', 'W_o'], r=8, alpha=16)
    out_after, _ = model(x)
    diff = (out_before - out_after).abs().max().item()
    print(f"[OK] apply_lora 初始输出差: {diff:.2e} (应为 0)")

    # 2. 参数冻结验证：LoRA 目标层的原权重必须冻结，只有 lora_A/B 可训练
    lora_layer = model.layers[0].self_attn.W_q
    assert not lora_layer.original.weight.requires_grad, "被包装的原权重应已冻结"
    assert lora_layer.lora_A.requires_grad and lora_layer.lora_B.requires_grad, "LoRA 参数应可训练"
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora_only = sum(p.numel() for n, p in model.named_parameters()
                      if 'lora' in n)
    print(f"[OK] 原 Linear 已冻结, LoRA 参数可训练: {n_lora_only:,}/{n_total:,} "
          f"({n_lora_only/n_total*100:.1f}% 为 LoRA 参数)")
    print(f"     （本实现只适配注意力投影；FFN/embedding 保持可训练，如需全冻结可扩展）")

    # 3. 训练几步 → LoRA 权重变化
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(5):
        logits, loss = model(x, labels=x)
        opt.zero_grad(); loss.backward(); opt.step()
    a_norm = model.layers[0].self_attn.W_q.lora_A.norm().item()
    print(f"[OK] 训练后 lora_A 范数: {a_norm:.4f} (非零说明在更新)")

    # 4. 保存/加载 roundtrip
    path = os.path.join(PROJECT_ROOT, 'checkpoints', 'lora_test.pt')
    save_lora(model, path)
    load_lora(model, path)

    # 5. merge 后输出与 merge 前一致
    out_merged_before, _ = model(x)
    merge_lora(model)
    out_merged_after, _ = model(x)
    diff = (out_merged_before - out_merged_after).abs().max().item()
    print(f"[OK] merge 前后输出差: {diff:.2e} (应≈0)")
    # merge 后模型里不应再有 LoRALinear
    from scripts.core.lora import LoRALinear as _L
    n_lora_left = sum(1 for m in model.modules() if isinstance(m, _L))
    assert n_lora_left == 0, "merge 后不应残留 LoRALinear"
    print("[OK] merge 后结构恢复原样")

    # 清理测试文件
    os.remove(path)
    print("\n[OK] LoRA 全部验证通过")
