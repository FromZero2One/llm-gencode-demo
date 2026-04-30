"""
快速测试脚本 - 验证所有模块是否正常工作
"""

import sys
import traceback


def test_tokenizer():
    """测试Token化模块"""
    print("\n" + "="*60)
    print("测试1: Tokenizer")
    print("="*60)
    
    try:
        from tokenizer import SimpleTokenizer
        
        tokenizer = SimpleTokenizer(vocab_size=1000)
        
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
    print("\n" + "="*60)
    print("测试2: Attention Mechanism")
    print("="*60)
    
    try:
        import torch
        from attention import MultiHeadAttention, PositionalEncoding
        
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
    print("\n" + "="*60)
    print("测试3: Transformer Model")
    print("="*60)
    
    try:
        import torch
        from transformer import TransformerModel
        
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
    print("\n" + "="*60)
    print("测试4: Code Generator")
    print("="*60)
    
    try:
        from tokenizer import SimpleTokenizer
        from transformer import TransformerModel
        from generator import CodeGenerator, TemperatureSampling
        
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
        
        strategy = TemperatureSampling(temperature=0.7)
        
        result = generator.generate(
            prompt="public class",
            max_length=20,
            strategy=strategy,
            verbose=False
        )
        
        assert 'generated_code' in result
        assert 'token_count' in result
        
        print(f"[OK] Generator测试通过")
        print(f"  生成Token数: {result['token_count']}")
        print(f"  生成代码长度: {len(result['generated_code'])} 字符")
        
        return True
    except Exception as e:
        print(f"[ERROR] Generator测试失败: {e}")
        traceback.print_exc()
        return False


def test_postprocessor():
    """测试后处理器"""
    print("\n" + "="*60)
    print("测试5: Post Processor")
    print("="*60)
    
    try:
        from postprocessor import CodePostProcessor
        
        post_processor = CodePostProcessor()
        
        test_code = "public class Test { public void method() { } }"
        
        result = post_processor.process(test_code)
        
        assert 'processed_code' in result
        assert 'validation_report' in result
        
        print(f"[OK] PostProcessor测试通过")
        print(f"  验证结果: {'有效' if result['validation_report']['is_valid'] else '无效'}")
        print(f"  应用步骤: {len(result['steps_applied'])}")
        
        return True
    except Exception as e:
        print(f"[ERROR] PostProcessor测试失败: {e}")
        traceback.print_exc()
        return False


def test_cache():
    """测试缓存"""
    print("\n" + "="*60)
    print("测试6: Cache")
    print("="*60)
    
    try:
        from cache import GenerationCache
        
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
    """测试完整管道"""
    print("\n" + "="*60)
    print("测试7: Complete Pipeline")
    print("="*60)
    
    try:
        from pipeline import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline(
            vocab_size=1000,
            d_model=128,
            nhead=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            cache_size=5,
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
        print(f"  来源: {result['source']}")
        print(f"  耗时: {result['processing_time']:.4f}s")
        print(f"  Token数: {result['token_count']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Pipeline测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("LLM代码生成演示 - 完整测试")
    print("="*60)
    
    tests = [
        ("Tokenizer", test_tokenizer),
        ("Attention", test_attention),
        ("Transformer", test_transformer),
        ("Generator", test_generator),
        ("PostProcessor", test_postprocessor),
        ("Cache", test_cache),
        ("Pipeline", test_pipeline),
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
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
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
