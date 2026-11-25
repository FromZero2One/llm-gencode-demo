"""
Demo SFT 数据 - 编程助手指令对（教学用小数据集，让 tiny/small 模型过拟合出对话行为）

真实 SFT 请替换为大规模指令数据（如 Alpaca/Belle 格式的 jsonl）。
"""

import os
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = "你是编程助手，用简洁的代码和说明回答问题。"

# (问题, 回答) 对——围绕 demo 语料里的代码模式设计，模型学得会
QA_PAIRS = [
    ("写一个 Hello World", "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}"),
    ("写一个 UserService 类", "public class UserService {\n    public User findById(Long id) {\n        return user;\n    }\n}"),
    ("写一个空的 Java 类", "public class Empty {\n}"),
    ("怎么创建一个列表", "List<String> items = new ArrayList<>();"),
    ("怎么创建一个 Map", "Map<String, Integer> map = new HashMap<>();"),
    ("写一个 for 循环", "for (int i = 0; i < size; i++) {\n}"),
    ("怎么判断空值", "if (value == null) {\n}"),
    ("写一个遍历方法", "public List<String> findAll() {\n    return items;\n}"),
    ("怎么抛出异常", "throw new IllegalArgumentException(\"invalid\");"),
    ("写一个 try catch", "try {\n} catch (Exception e) {\n    logger.info(e.getMessage());\n}"),
    ("写一个保存方法", "public void save(User user) {\n    this.user = user;\n}"),
    ("写一个删除方法", "public void delete(Long id) {\n    repository.delete(id);\n}"),
    ("你好", "你好！我是编程助手，可以帮你写代码。"),
    ("你能做什么", "我可以帮你编写和解释 Java 代码。"),
    ("什么是单例模式", "单例模式保证一个类只有一个实例，通过私有构造方法和静态方法实现。"),
    ("什么是多态", "多态是同一个方法调用在不同对象上有不同行为，是面向对象三大特性之一。"),
    ("写一个快速排序", "public void quickSort(int[] a, int l, int r) {\n    if (l >= r) return;\n}"),
    ("写一个订单类", "public class Order {\n    private Long id;\n    private String name;\n}"),
    ("怎么打印日志", "logger.info(\"processing \" + value);"),
    ("写一个用户控制器", "public class UserController {\n    private UserService service;\n}"),
]


def generate_sft_demo(output_path: str, repeat: int = 20, seed: int = 42):
    """
    生成 demo SFT 数据（jsonl）

    Args:
        output_path: 输出路径
        repeat: 每条 QA 重复次数（教学小数据需要重复才能过拟合出行为）
    """
    rng = random.Random(seed)
    lines = []
    for q, a in QA_PAIRS:
        for _ in range(repeat):
            lines.append(json.dumps({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': q},
                    {'role': 'assistant', 'content': a},
                ]
            }, ensure_ascii=False))
    rng.shuffle(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[sft_demo] SFT 数据已生成: {output_path} ({len(lines)} 条)")
    return output_path


if __name__ == '__main__':
    default_path = os.path.join(PROJECT_ROOT, 'scripts', 'data', 'sft_demo.jsonl')
    generate_sft_demo(default_path)
