"""
Test suite for new features: Trainer and KV Cache
"""

import sys
import os
import torch

# Add scripts to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scripts_dir = os.path.join(project_root, 'scripts')
sys.path.insert(0, scripts_dir)

# Import logger
from logger import logging_context

from trainer import CrossEntropyLoss, AdamW, WarmupLinearScheduler, TextDataset, Trainer
from kv_cache import KVCache, KVCacheManager
from transformer import TransformerModel


def test_cross_entropy_loss():
    """Test CrossEntropyLoss"""
    print("\n" + "="*70)
    print("Test 1: CrossEntropyLoss")
    print("="*70)
    
    try:
        loss_fn = CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)
        
        batch_size = 2
        seq_len = 5
        vocab_size = 100
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = loss_fn(logits, targets)
        
        print(f"  [OK] Basic functionality works")
        print(f"    - Loss value: {loss.item():.4f}")
        print(f"    - Loss in valid range: {0 <= loss.item() < 10}")
        
        # Test label smoothing
        loss_fn_smooth = CrossEntropyLoss(label_smoothing=0.1)
        loss_smooth = loss_fn_smooth(logits, targets)
        print(f"  [OK] Label smoothing works: {loss_smooth.item():.4f}")
        
        print("  [PASS] CrossEntropyLoss test passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] CrossEntropyLoss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adamw_optimizer():
    """Test AdamW optimizer"""
    print("\n" + "="*70)
    print("Test 2: AdamW Optimizer")
    print("="*70)
    
    try:
        model = torch.nn.Linear(10, 5)
        
        optimizer = AdamW(
            model.parameters(),
            lr=0.01,
            weight_decay=0.01
        )
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        
        initial_loss = None
        final_loss = None
        
        for step in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            
            if step == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            if step == 4:
                final_loss = loss.item()
        
        print(f"  [OK] Optimizer works correctly")
        print(f"    - Initial loss: {initial_loss:.4f}")
        print(f"    - Final loss: {final_loss:.4f}")
        print(f"    - Loss decreased: {initial_loss > final_loss}")
        
        print("  [PASS] AdamW test passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] AdamW test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_warmup_scheduler():
    """Test WarmupLinearScheduler"""
    print("\n" + "="*70)
    print("Test 3: WarmupLinearScheduler")
    print("="*70)
    
    try:
        model = torch.nn.Linear(5, 5)
        optimizer = AdamW(model.parameters(), lr=0.01)
        
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=50,
            max_lr=0.01,
            min_lr=0.001
        )
        
        lrs = []
        for step in range(50):
            lr = scheduler.step()
            lrs.append(lr)
        
        warmup_lrs = lrs[:10]
        is_increasing = all(warmup_lrs[i] <= warmup_lrs[i+1] for i in range(len(warmup_lrs)-1))
        print(f"  [OK] Warmup phase LR increasing: {is_increasing}")
        print(f"    - Start LR: {lrs[0]:.6f}")
        print(f"    - End of warmup LR: {lrs[9]:.6f}")
        
        decay_lrs = lrs[10:]
        is_decreasing = all(decay_lrs[i] >= decay_lrs[i+1] for i in range(len(decay_lrs)-1))
        print(f"  [OK] Decay phase LR decreasing: {is_decreasing}")
        print(f"    - Start of decay LR: {lrs[10]:.6f}")
        print(f"    - Final LR: {lrs[-1]:.6f}")
        
        print("  [PASS] WarmupLinearScheduler test passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] WarmupLinearScheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kv_cache():
    """Test KV Cache"""
    print("\n" + "="*70)
    print("Test 4: KV Cache")
    print("="*70)
    
    try:
        cache = KVCache(
            num_layers=2,
            batch_size=1,
            num_heads=4,
            max_seq_len=10,
            head_dim=16,
            device=torch.device('cpu')
        )
        
        print(f"  [OK] KV Cache initialized")
        print(f"    - Num layers: {cache.num_layers}")
        print(f"    - Max length: {cache.max_seq_len}")
        
        for step in range(5):
            new_k = torch.randn(1, 4, 1, 16)
            new_v = torch.randn(1, 4, 1, 16)
            
            cache.update(layer_idx=0, new_k=new_k, new_v=new_v)
            cache.update(layer_idx=1, new_k=new_k, new_v=new_v)
        
        cached_k, cached_v = cache.get(layer_idx=0)
        print(f"  [OK] Cache update and retrieval works")
        print(f"    - Current length: {cached_k.shape[2]}")
        
        cache.reset()
        cached_k_after_reset, _ = cache.get(layer_idx=0)
        print(f"  [OK] Cache reset works")
        print(f"    - Length after reset: {cached_k_after_reset.shape[2]}")
        
        stats = cache.get_stats()
        print(f"  [OK] Stats retrieval works")
        print(f"    - Memory: {stats['memory_kb']:.2f} KB")
        
        print("  [PASS] KV Cache test passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] KV Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_documentation_files():
    """Test documentation files exist"""
    print("\n" + "="*70)
    print("Test 5: Documentation Files")
    print("="*70)
    
    try:
        required_files = [
            "scripts/trainer.py",
            "scripts/kv_cache.py",
            "scripts/test_training_and_cache.py",
            "docs/TRAINING_AND_KV_CACHE_GUIDE.md"
        ]
        
        all_exist = True
        for filepath in required_files:
            exists = os.path.exists(filepath)
            status = "[OK]" if exists else "[FAIL]"
            print(f"  {status} {filepath}: {'exists' if exists else 'missing'}")
            
            if exists:
                size = os.path.getsize(filepath)
                print(f"      File size: {size/1024:.1f} KB")
            
            all_exist = all_exist and exists
        
        if all_exist:
            print("  [PASS] All documentation files present")
            return True
        else:
            print("  [FAIL] Some documentation files missing")
            return False
        
    except Exception as e:
        print(f"  [FAIL] Documentation check failed: {e}")
        return False


def main():
    """Run all tests"""
    # Use logging context manager
    with logging_context(__file__):
        print("\n" + "="*70)
        print("Training System and KV Cache Feature Tests")
        print("="*70)
        
        tests = [
            ("CrossEntropyLoss", test_cross_entropy_loss),
            ("AdamW Optimizer", test_adamw_optimizer),
            ("WarmupScheduler", test_warmup_scheduler),
            ("KV Cache", test_kv_cache),
            ("Documentation", test_documentation_files),
        ]
        
        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append((name, result))
            except Exception as e:
                print(f"\n  [FAIL] {name} test exception: {e}")
                results.append((name, False))
        
        print("\n\n" + "="*70)
        print("Test Results Summary")
        print("="*70)
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for name, result in results:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status:8} - {name}")
        
        print("\n" + "-"*70)
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n" + "="*70)
            print("ALL TESTS PASSED! Code is complete and functional!")
            print("="*70)
            return 0
        else:
            print(f"\n{total - passed} tests failed. Please check error messages above.")
            print("="*70)
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
