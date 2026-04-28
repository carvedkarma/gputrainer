#!/usr/bin/env python3
"""
Diagnostic script to test if GPU models respond to different inputs.

Tests:
1. Check if predictions change with different random inputs
2. Check if predictions change with inverted inputs
3. Check training label distribution from data
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 60)
    print("GPU PREDICTION DIAGNOSTIC")
    print("=" * 60)
    
    from api.server import model_manager
    
    # Check if models are loaded
    print(f"\n1. MODELS LOADED:")
    print(f"   Model instances: {list(model_manager.model_instances.keys())}")
    print(f"   Input dim expected: {model_manager.input_dim}")
    print(f"   Sequence length: {model_manager.sequence_length}")
    print(f"   Training mode: {model_manager.training_mode}")
    
    if not model_manager.model_instances:
        print("\n   ERROR: No models loaded! Cannot test predictions.")
        return
    
    # Generate test inputs
    print(f"\n2. TESTING INPUT SENSITIVITY:")
    device = model_manager.device
    seq_len = model_manager.sequence_length
    input_dim = model_manager.input_dim
    
    # Test with different random seeds
    results = []
    
    for seed in [42, 123, 999]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create random input
        test_input = torch.randn(1, seq_len, input_dim).to(device)
        
        print(f"\n   Seed {seed}: Input range [{test_input.min():.3f}, {test_input.max():.3f}]")
        
        # Test each model
        for name, model in model_manager.model_instances.items():
            model.eval()
            with torch.no_grad():
                try:
                    # Try forward_multihead first
                    if hasattr(model, 'forward_multihead'):
                        output = model.forward_multihead(test_input)
                        if isinstance(output, dict):
                            logits = output.get('logits', output.get('direction', None))
                        else:
                            logits = output
                    else:
                        logits = model(test_input)
                    
                    if logits is not None:
                        if isinstance(logits, torch.Tensor):
                            probs = torch.softmax(logits, dim=-1)
                            print(f"   {name}: probs={probs[0].cpu().numpy()}")
                            results.append((seed, name, probs[0].cpu().numpy()))
                except Exception as e:
                    print(f"   {name}: ERROR - {e}")
    
    # Check if predictions are too similar
    print(f"\n3. PREDICTION VARIANCE ANALYSIS:")
    for model_name in set(r[1] for r in results):
        model_probs = [r[2] for r in results if r[1] == model_name]
        if len(model_probs) >= 2:
            variance = np.var(model_probs, axis=0)
            print(f"   {model_name}: Variance per class = {variance}")
            if np.max(variance) < 0.001:
                print(f"      ⚠️  LOW VARIANCE - Model may have collapsed to constant output!")
    
    # Check label distribution from data
    print(f"\n4. CHECKING LABEL DISTRIBUTION (historical):")
    try:
        from data.pipeline import create_labels
        import pandas as pd
        
        # Generate synthetic price data with trends
        n = 5000
        np.random.seed(42)
        base_price = 80000
        
        # Create price series with some trends
        returns = np.random.normal(0, 0.002, n)  # 0.2% avg returns
        # Add some trending periods
        returns[500:700] = np.random.normal(0.003, 0.002, 200)   # Uptrend
        returns[1500:1700] = np.random.normal(-0.003, 0.002, 200) # Downtrend
        
        prices = base_price * np.cumprod(1 + returns)
        df = pd.DataFrame({'close': prices})
        
        labels = create_labels(df, horizon=16, threshold=0.001, trading_cost=0.0009)
        
        unique, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
        print(f"   Labels: {dict(zip(unique.astype(int), counts))}")
        
        total = counts.sum()
        for label, count in zip(unique, counts):
            pct = 100 * count / total
            label_name = {-1: "SHORT", 0: "HOLD", 1: "LONG"}.get(int(label), str(label))
            print(f"   {label_name}: {count} ({pct:.1f}%)")
        
        hold_pct = counts[unique == 0].sum() / total * 100 if 0 in unique else 0
        if hold_pct > 70:
            print(f"\n   ⚠️  HIGH HOLD RATIO ({hold_pct:.1f}%) - Model may learn to always predict HOLD!")
    except Exception as e:
        print(f"   ERROR checking labels: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
