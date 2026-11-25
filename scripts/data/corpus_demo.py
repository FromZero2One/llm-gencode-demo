"""
Demo 语料生成器 - 内置训练语料，保证项目零依赖可运行

提供两类语料（均为内置文本，无需网络下载）：
    1. 代码语料：Java 代码片段（贴合本项目的代码生成场景）
    2. 自然语言语料：中英文技术问答短句（为 P8 SFT 铺垫通用性）

语料规模：几百 KB 级别 —— 对 tiny/small preset 的 BPE 词表（6400）刚好够用，
真实大规模预训练请用 --data_path 指向自己的语料文件。
"""

import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------
# Java 代码语料模板
# ----------------------------------------------------------------------
JAVA_CLASS_TEMPLATES = [
    "public class {cls} {{",
    "public class {cls} {{",
    "public interface {cls} {{",
]

JAVA_METHOD_TEMPLATES = [
    "    public {ret} {name}({params}) {{",
    "    private {ret} {name}({params}) {{",
    "    protected {ret} {name}({params}) {{",
]

JAVA_STATEMENTS = [
    "        return result;",
    "        return null;",
    "        return this.{field};",
    "        this.{field} = {value};",
    "        int count = 0;",
    "        for (int i = 0; i < size; i++) {{",
    "        if (value == null) {{",
    "        throw new IllegalArgumentException(\"invalid\");",
    "        List<String> items = new ArrayList<>();",
    "        Map<String, Integer> map = new HashMap<>();",
    "        System.out.println(\"debug: \" + value);",
    "        try {{",
    "        }} catch (Exception e) {{",
    "            logger.info(\"processing \" + e.getMessage());",
    "        }}",
    "    }}",
]

JAVA_CLASSES = [
    "UserService", "OrderService", "PaymentService", "UserController",
    "OrderController", "ProductService", "InventoryService", "UserRepository",
    "OrderRepository", "BaseEntity", "ResponseWrapper", "DataAccessException",
    "ShoppingCart", "CustomerProfile", "NotificationService", "AuditLogger",
]

JAVA_TYPES = ["void", "int", "long", "boolean", "String", "List<String>",
              "Map<String, Object>", "Optional<User>", "BigDecimal"]

JAVA_METHOD_NAMES = ["findById", "findAll", "save", "delete", "update",
                     "process", "validate", "execute", "handle", "query",
                     "calculate", "transform", "convert", "apply"]

JAVA_PARAMS = ["", "Long id", "String name", "User user", "Order order",
               "Long id, String name", "int page, int size"]


def _gen_java_snippet(rng: random.Random) -> str:
    """生成一个完整的 Java 类片段"""
    cls = rng.choice(JAVA_CLASSES)
    lines = [f"public class {cls} {{"]
    n_methods = rng.randint(1, 3)
    for _ in range(n_methods):
        ret = rng.choice(JAVA_TYPES)
        name = rng.choice(JAVA_METHOD_NAMES)
        params = rng.choice(JAVA_PARAMS)
        lines.append(f"    public {ret} {name}({params}) {{")
        for _ in range(rng.randint(1, 4)):
            lines.append(rng.choice(JAVA_STATEMENTS).replace("{{", "{").replace("}}", "}"))
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 中英文自然语言语料
# ----------------------------------------------------------------------
NATURAL_SENTENCES = [
    "创建一个用户服务类，包含增删改查方法。",
    "编写一个函数计算斐波那契数列的第 n 项。",
    "如何用 Java 实现单例模式？",
    "什么是依赖注入？为什么要使用它？",
    "写一个快速排序算法的实现。",
    "解释一下 HTTP 状态码 404 和 500 的区别。",
    "面向对象的三大特性是封装、继承和多态。",
    "数据库事务的 ACID 特性分别指什么？",
    "Write a function to reverse a linked list.",
    "Explain the difference between == and equals in Java.",
    "How does garbage collection work in the JVM?",
    "What is the time complexity of binary search?",
    "A hash map stores key-value pairs for fast lookup.",
    "Recursion is a function that calls itself until a base case is reached.",
    "线程和进程的区别是什么？",
    "使用缓存可以显著提高重复请求的响应速度。",
    "代码审查有助于发现潜在的缺陷和安全问题。",
    "单元测试应该覆盖正常路径和边界条件。",
    "The transformer architecture uses self-attention to process sequences.",
    "BPE tokenization breaks rare words into subword units.",
    "预训练模型通过大规模语料学习语言的统计规律。",
    "微调让预训练模型适应特定领域的任务。",
    "注意力机制让模型关注输入中最重要的部分。",
    "大语言模型的参数量通常达到数十亿级别。",
]


def generate_demo_corpus(output_path: str, lines: int = 8000, seed: int = 42):
    """
    生成 demo 训练语料文件

    组成：约 60% Java 代码片段 + 40% 中英自然语言句子（重复采样至目标行数）

    Args:
        output_path: 输出 .txt 路径
        lines: 目标行数
        seed: 随机种子（保证可复现）
    """
    rng = random.Random(seed)

    # 种子素材
    java_snippets = [_gen_java_snippet(rng) for _ in range(400)]
    natural = list(NATURAL_SENTENCES)

    # 生成到目标行数
    corpus = []
    while len(corpus) < lines:
        r = rng.random()
        if r < 0.6:
            corpus.append(rng.choice(java_snippets))
        else:
            corpus.append(rng.choice(natural))
    rng.shuffle(corpus)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(corpus))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[corpus_demo] 语料已生成: {output_path} ({lines} 行, {size_kb:.1f} KB)")
    return output_path


if __name__ == '__main__':
    default_path = os.path.join(PROJECT_ROOT, 'scripts', 'data', 'corpus_demo.txt')
    generate_demo_corpus(default_path)
    # 预览
    with open(default_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(line[:80])
