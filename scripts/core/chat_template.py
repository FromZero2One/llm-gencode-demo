"""
Chat Template - 对话模板与 loss mask 生成（对标 Qwen ChatML 风格）

╔═══════════════════════════════════════════════════════════╗
║  📚 SFT 的核心细节：为什么只有"回答部分"计算 loss？        ║
╚═══════════════════════════════════════════════════════════╝

一条 SFT 样本：
    <|im_start|>user
    写一个 Hello World<|im_end|>
    <|im_start|>assistant
    class HelloWorld { ... }<|im_end|>

训练时如果把整段都算 loss，模型会把精力浪费在"学习复述用户问题"上。
SFT 的正确做法：只在 assistant 的回答部分计算 loss，
prompt 部分 label 置 -100（CrossEntropy 忽略）——这就是 loss mask。

模板格式（与 Qwen ChatML 一致，简化版）：
    <|im_start|>role\ncontent<|im_end|>\n
推理时模型看到 <|im_start|>assistant\n 后开始生成，直到输出 <|im_end|>。

使用：
    from scripts.core.chat_template import apply_chat_template, build_labels
    text = apply_chat_template(messages)          # 拼模板
    ids, labels = build_labels(text, tokenizer)   # 编码 + loss mask
"""

from typing import List, Dict, Tuple

# 模板控制符（与 BPETokenizer 预留的特殊 token 对齐）
IM_START = '<|im_start|>'
IM_END = '<|im_end|>'

# 各角色名（模板内的 role 文本）
ROLES = ('system', 'user', 'assistant')


def apply_chat_template(messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """
    把 messages 列表拼成模型输入文本

    Args:
        messages: [{'role': 'user'|'assistant'|'system', 'content': '...'}, ...]
        add_generation_prompt: True 时在末尾追加 '<|im_start|>assistant\n'，
            供推理时引导模型开始回答（训练时不加）

    Returns:
        拼接后的模板字符串

    Example:
        >>> msgs = [{'role': 'user', 'content': '你好'},
        ...         {'role': 'assistant', 'content': '你好！有什么可以帮你？'}]
        >>> text = apply_chat_template(msgs)
        >>> '<|im_start|>user' in text and '<|im_end|>' in text
        True
    """
    parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role not in ROLES:
            raise ValueError(f"未知角色: {role}, 可选: {ROLES}")
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")

    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return ''.join(parts)


def build_labels(text: str, tokenizer, max_length: int = 512) -> Tuple[List[int], List[int]]:
    """
    编码模板文本并生成 SFT loss mask

    规则：
        - input_ids: 整段模板的 BPE 编码（含模板控制符 token）
        - labels: 与 input_ids 等长；assistant 回答区段 = 原 id，
          其余（system/user 区段 + 模板控制符）= -100（不参与 loss）

    实现方式：按区段切分文本，逐段编码，记录每段是否属于 assistant 回答。

    Args:
        text: apply_chat_template 的输出
        tokenizer: BPETokenizer（需已训练）
        max_length: 最大长度（超长截断）

    Returns:
        (input_ids, labels)
    """
    input_ids = []
    labels = []

    # 按行解析区段：'<|im_start|>role\ncontent<|im_end|>\n'
    segments = _split_segments(text)

    for seg_role, seg_content, is_assistant in segments:
        # 区段头（模板控制符 + 角色名）：不算 loss
        header = f"{IM_START}{seg_role}\n"
        header_ids = tokenizer.tokenizer.encode(header, add_special_tokens=False).ids
        input_ids.extend(header_ids)
        labels.extend([-100] * len(header_ids))

        # 内容：assistant 段算 loss，其余不算
        content_ids = tokenizer.tokenizer.encode(seg_content, add_special_tokens=False).ids
        input_ids.extend(content_ids)
        if is_assistant:
            labels.extend(content_ids)
        else:
            labels.extend([-100] * len(content_ids))

        # 区段尾 <|im_end|>\n：assistant 段的结束符算 loss（模型要学会"停下来"）
        tail_ids = tokenizer.tokenizer.encode(f"{IM_END}\n", add_special_tokens=False).ids
        input_ids.extend(tail_ids)
        if is_assistant:
            labels.extend(tail_ids)
        else:
            labels.extend([-100] * len(tail_ids))

    # 截断
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return input_ids, labels


def _split_segments(text: str) -> List[Tuple[str, str, bool]]:
    """
    解析模板文本为区段列表

    Returns:
        [(role, content, is_assistant), ...]
    """
    segments = []
    # 简单状态机解析（模板是我们自己拼的，格式可控）
    blocks = text.split(IM_START)
    for block in blocks:
        if not block:
            continue
        role_line, _, rest = block.partition('\n')
        role = role_line.strip()
        content, _, _ = rest.partition(IM_END)
        segments.append((role, content, role == 'assistant'))
    return segments


def messages_to_prompt_legacy(messages: List[Dict[str, str]]) -> str:
    """旧版无模板拼接（保留给 regex 分词器 / 未 SFT 的模型使用）"""
    parts = [m.get('content', '') for m in messages
             if m.get('role') in ('system', 'user')]
    return '\n'.join(parts) if parts else 'public class'


if __name__ == '__main__':
    # ===== 自验证 =====
    from scripts.core.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    assert len(tok.vocab) > 300, "BPE 分词器尚未训练，先运行 python -m scripts.core.bpe_tokenizer"

    print("=" * 60)
    print("Chat Template 验证")
    print("=" * 60)

    # 1. 模板拼接
    messages = [
        {'role': 'system', 'content': '你是编程助手'},
        {'role': 'user', 'content': '写一个 Hello World'},
        {'role': 'assistant', 'content': 'public class Hello { }'},
    ]
    text = apply_chat_template(messages)
    print(text)
    assert text.count(IM_START) == 3 and text.count(IM_END) == 3

    # 2. 推理 prompt（add_generation_prompt）
    gen_prompt = apply_chat_template(messages[:2], add_generation_prompt=True)
    assert gen_prompt.endswith(f"{IM_START}assistant\n")
    print("[OK] add_generation_prompt 正确追加 assistant 开头")

    # 3. loss mask：只有 assistant 部分有有效 label
    ids, labels = build_labels(text, tok)
    valid = [l for l in labels if l != -100]
    print(f"总 token: {len(ids)}, 计算 loss 的 token: {len(valid)}")
    assert len(valid) > 0, "assistant 段应有有效 label"
    # 验证有效 label 对应的文本正是回答内容
    answer_text = tok.decode(valid)
    assert 'Hello' in answer_text, f"loss 应只覆盖回答部分, 实际: {answer_text!r}"
    assert '编程助手' not in answer_text, "system 部分不应计算 loss"
    assert 'Hello World' not in answer_text.replace('Hello { }', ''), "user 问题不应计算 loss"
    print(f"[OK] loss mask 正确：仅覆盖回答 '{answer_text}'")

    # 4. 非法角色报错
    try:
        apply_chat_template([{'role': 'robot', 'content': 'x'}])
        assert False, "应抛出 ValueError"
    except ValueError:
        print("[OK] 非法角色正确报错")

    print("\n[OK] Chat Template 全部验证通过")
