"""
快速测试脚本 - 验证所有模块是否正常工作
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入日志系统
from scripts.utils.logger import logging_context


def test_tokenizer():
    """测试Token化模块"""
    print("\n" + "=" * 60)
    print("测试1: Tokenizer")
    print("=" * 60)

    try:
        from scripts.core.tokenizer import SimpleTokenizer

        tokenizer = SimpleTokenizer(vocab_size=1000, debug_mode=True)

        test_text = "public class UserService"
        token_ids, mask = tokenizer.encode(test_text, max_length=32)

        decoded = tokenizer.decode(token_ids)

        print(f"[OK] Tokenizer测试通过")
        print(f"  原始文本: {test_text}")
        print(f"  Token数量: {sum(mask)}")
        print(f"  解码结果: {decoded}")

        return True
    except Exception as e:
        print(f"[ERROR] Tokenizer测试失败: {e}")
        traceback.print_exc()
        return False


def test_attention():
    """测试注意力机制"""
    print("\n" + "=" * 60)
    print("测试2: Attention Mechanism")
    print("=" * 60)

    try:
        import torch
        from scripts.core.attention import MultiHeadAttention, PositionalEncoding

        # 测试多头注意力
        d_model = 128
        nhead = 8
        batch_size = 1
        seq_len = 10

        attention = MultiHeadAttention(d_model=d_model, nhead=nhead)

        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        output, weights = attention(query, key, value)

        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape[0] == batch_size

        # 测试位置编码
        pos_encoder = PositionalEncoding(d_model=d_model)
        embedded = torch.randn(batch_size, seq_len, d_model)
        encoded = pos_encoder(embedded)

        assert encoded.shape == embedded.shape

        print(f"[OK] Attention测试通过")
        print(f"  Output shape: {output.shape}")
        print(f"  Attention weights shape: {weights.shape}")

        return True
    except Exception as e:
        print(f"[ERROR] Attention测试失败: {e}")
        traceback.print_exc()
        return False


def test_transformer():
    """测试Transformer模型"""
    print("\n" + "=" * 60)
    print("测试3: Transformer Model")
    print("=" * 60)

    try:
        import torch
        from scripts.core.transformer import TransformerModel

        vocab_size = 1000
        d_model = 128

        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 1
        src_seq_len = 20
        tgt_seq_len = 15

        src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

        logits, enc_weights, dec_weights = model(src, tgt)

        assert logits.shape == (batch_size, tgt_seq_len, vocab_size)

        print(f"[OK] Transformer测试通过")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Encoder layers: {len(enc_weights)}")
        print(f"  Decoder layers: {len(dec_weights)}")

        return True
    except Exception as e:
        print(f"[ERROR] Transformer测试失败: {e}")
        traceback.print_exc()
        return False


def test_generator():
    """测试代码生成器"""
    print("\n" + "=" * 60)
    print("测试4: Code Generator")
    print("=" * 60)

    try:
        from scripts.core.tokenizer import SimpleTokenizer
        from scripts.core.transformer import TransformerModel
        from scripts.generation.code_generator import CodeGenerator

        vocab_size = 1000
        d_model = 128

        tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=8,
            num_encoder_layers=1,
            num_decoder_layers=1
        )

        generator = CodeGenerator(model, tokenizer, device='cpu')

        result = generator.generate(
            prompt="public class",
            max_length=20,
            temperature=0.7,
            verbose=False
        )

        assert 'code' in result
        assert 'token_count' in result

        print(f"[OK] Generator测试通过")
        print(f"  生成Token数: {result['token_count']}")
        print(f"  生成代码长度: {len(result['code'])} 字符")

        return True
    except Exception as e:
        print(f"[ERROR] Generator测试失败: {e}")
        traceback.print_exc()
        return False


def test_postprocessor():
    """测试后处理器(已合并到CodeGenerator)"""
    print("\n" + "=" * 60)
    print("测试5: Post Processor (merged into CodeGenerator)")
    print("=" * 60)

    try:
        from scripts.generation.code_generator import CodeGenerator
        from scripts.core.tokenizer import SimpleTokenizer
        from scripts.core.transformer import TransformerModel

        # 创建带后处理的生成器
        tokenizer = SimpleTokenizer(vocab_size=1000)
        model = TransformerModel(
            vocab_size=1000,
            d_model=128,
            nhead=8,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        post_processor = CodeGenerator(model, tokenizer, device='cpu', enable_post_process=True)

        test_code = "public class Test { public void method() { } }"

        # 测试后处理功能
        formatted = post_processor.post_processor.simple_format(test_code)
        with_imports = post_processor.post_processor.add_imports(formatted)

        assert len(with_imports) > 0

        print(f"[OK] PostProcessor测试通过(内嵌在CodeGenerator中)")
        print(f"  格式化后长度: {len(formatted)}")
        print(f"  添加import后长度: {len(with_imports)}")

        return True
    except Exception as e:
        print(f"[ERROR] PostProcessor测试失败: {e}")
        traceback.print_exc()
        return False


def test_cache():
    """测试缓存"""
    print("\n" + "=" * 60)
    print("测试6: Cache")
    print("=" * 60)

    try:
        from scripts.optimization.cache import GenerationCache

        cache = GenerationCache(max_size=10)

        # 测试put和get
        prompt = "test prompt"
        result = "test result"

        cache.put(prompt, result)
        cached = cache.get(prompt)

        assert cached == result

        # 测试统计
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 0

        print(f"[OK] Cache测试通过")
        print(f"  命中率: {stats['hit_rate']}")
        print(f"  缓存大小: {stats['size']}")

        return True
    except Exception as e:
        print(f"[ERROR] Cache测试失败: {e}")
        traceback.print_exc()
        return False


def test_pipeline():
    """测试完整管道(使用preset配置)"""
    print("\n" + "=" * 60)
    print("测试7: Complete Pipeline (with preset)")
    print("=" * 60)

    try:
        from scripts.pipeline import CodeGenerationPipeline

        # 使用tiny preset进行测试(更快)
        pipeline = CodeGenerationPipeline(
            preset='tiny',
            device='cpu'
        )

        result = pipeline.generate(
            prompt="public class Test",
            max_length=20,
            verbose=False
        )

        assert result['success'] == True
        assert 'processed_code' in result

        print(f"[OK] Pipeline测试通过")
        print(f"  Preset: tiny")
        print(f"  来源: {result['source']}")
        print(f"  耗时: {result['processing_time']:.4f}s")
        print(f"  Token数: {result['token_count']}")

        return True
    except Exception as e:
        print(f"[ERROR] Pipeline测试失败: {e}")
        traceback.print_exc()
        return False


def test_presets():
    """测试预设配置系统"""
    print("\n" + "=" * 60)
    print("测试8: Preset Configuration System")
    print("=" * 60)

    try:
        from scripts.config.presets import get_preset_config, list_presets

        # 测试1: 列出所有预设
        presets_str = list_presets()
        assert 'tiny' in presets_str
        assert 'small' in presets_str
        assert 'medium' in presets_str
        print(f"[OK] 预设列表功能正常")

        # 测试2: 获取特定预设
        config = get_preset_config('tiny')
        assert config['vocab_size'] == 500
        assert config['d_model'] == 64
        assert config['nhead'] == 4
        print(f"[OK] Tiny preset配置正确")

        config = get_preset_config('small')
        assert config['vocab_size'] == 1000
        assert config['d_model'] == 128
        print(f"[OK] Small preset配置正确")

        # 测试3: 错误处理
        try:
            get_preset_config('invalid')
            assert False, "应该抛出ValueError"
        except ValueError:
            print(f"[OK] 错误处理正常")

        print(f"\n[SUCCESS] Preset系统测试通过")
        return True
    except Exception as e:
        print(f"[ERROR] Preset测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    # 使用日志上下文管理器
    with logging_context(__file__):
        print("=" * 60)
        print("LLM代码生成演示 - 完整测试")
        print("=" * 60)

        tests = [
            ("Tokenizer", test_tokenizer),
            ("Attention", test_attention),
            ("Transformer", test_transformer),
            ("Generator", test_generator),
            ("PostProcessor", test_postprocessor),
            ("Cache", test_cache),
            ("Pipeline", test_pipeline),
            ("Presets", test_presets),
        ]

        results = []

        for name, test_func in tests:
            try:
                success = test_func()
                results.append((name, success))
            except Exception as e:
                print(f"\n[ERROR] {name}测试异常: {e}")
                traceback.print_exc()
                results.append((name, False))

        # 汇总结果
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)

        passed = sum(1 for _, success in results if success)
        total = len(results)

        for name, success in results:
            status = "[OK] 通过" if success else "[ERROR] 失败"
            print(f"  {status} - {name}")

        print(f"\n总计: {passed}/{total} 测试通过")

        if passed == total:
            print("\n[SUCCESS] 所有测试通过！系统工作正常。")
            print("\n下一步:")
            print("  1. 运行 python main.py 查看完整演示")
            print("  2. 阅读 README.md 了解详细用法")
            print("  3. 修改参数进行实验")
            return 0
        else:
            print(f"\n[WARNING] {total - passed} 个测试失败，请检查错误信息")
            return 1


if __name__ == '__main__':
    exit(main())
