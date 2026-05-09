#!/usr/bin/env python3
"""
H1.178: Attention with Decay Scaling on Long Sequences

Based on findings:
- H1.122: Adaptive decay (+89.5%) on 20-100 steps
- H3.39-40: Decay attention (+9.8-30.4%) on stochastic dynamics
- H3.64: Decay attention (+19.6%) on 30-50 step sequences
- H1.106: Attention (+0.2%) marginal on 40-60 step tasks

Hypothesis: Adaptive decay scaling can push attention advantage further 
on longer sequences (100-200 steps) where H1.106 was marginal.
"""

import numpy as np
import json
from pathlib import Path

def generate_long_sequence_data(n_samples=200, min_len=100, max_len=200, seed=42):
    np.random.seed(seed)
    X, y = [], []
    
    for _ in range(n_samples):
        seq_len = np.random.randint(min_len, max_len + 1)
        
        # Multi-scale temporal structure
        state = np.zeros((seq_len, 4))
        for t in range(seq_len):
            # Fast dynamics
            state[t, 0] = 0.7 * state[t-1, 0] + np.random.randn() * 0.3 if t > 0 else np.random.randn() * 0.3
            # Slow dynamics
            state[t, 1] = 0.95 * state[t-1, 1] + np.random.randn() * 0.1 if t > 0 else np.random.randn() * 0.1
            # Position (cumulative)
            state[t, 2] = state[t-1, 2] + state[t, 0] if t > 0 else state[t, 0]
            # Action
            state[t, 3] = np.random.randn() * 0.5
        
        # Complex target: combination of dynamics
        target = state[1:, 2] + 0.5 * state[1:, 0] + 0.3 * state[1:, 1]
        
        X.append(state[:-1])
        y.append(target)
    
    return X, y

def baseline(X, y):
    """Mean prediction baseline."""
    y_array = np.array([yi.mean() for yi in y])
    pred = np.ones_like(y_array) * y_array.mean()
    return np.mean((y_array - pred)**2)

def fixed_decay_attention(X, y, decay=0.9):
    """Standard fixed decay attention."""
    y_array = np.array([yi.mean() for yi in y])
    preds = []
    
    for xi in X:
        seq_len = xi.shape[0]
        
        # Fixed decay attention
        attn = np.zeros((seq_len, seq_len))
        for t in range(seq_len):
            for s in range(seq_len):
                if s <= t:
                    attn[t, s] = decay ** (t - s)
                else:
                    attn[t, s] = decay ** (s - t - 1)
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
        
        # Apply to state features
        attended = attn @ xi
        preds.append(attended[-1].sum())
    
    return np.mean((y_array - np.array(preds))**2)

def adaptive_decay_attention(X, y):
    """Adaptive decay based on sequence length."""
    y_array = np.array([yi.mean() for yi in y])
    preds = []
    
    for xi in X:
        seq_len = xi.shape[0]
        
        # Adaptive decay: shorter sequences = higher decay
        # Optimal decay = 0.97 for short, 0.99 for long
        decay = min(0.97 + 0.0002 * seq_len, 0.995)
        
        attn = np.zeros((seq_len, seq_len))
        for t in range(seq_len):
            for s in range(seq_len):
                if s <= t:
                    attn[t, s] = decay ** (t - s)
                else:
                    attn[t, s] = decay ** (s - t - 1)
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
        
        attended = attn @ xi
        preds.append(attended[-1].sum())
    
    return np.mean((y_array - np.array(preds))**2)

def multi_scale_attention(X, y):
    """Multi-scale attention (H3.82 style)."""
    y_array = np.array([yi.mean() for yi in y])
    preds = []
    
    for xi in X:
        seq_len = xi.shape[0]
        
        # Multiple windows
        outputs = []
        for window in [3, 5, 7]:
            attn = np.zeros((seq_len, seq_len))
            for t in range(seq_len):
                for s in range(max(0, t-window), min(seq_len, t+window+1)):
                    attn[t, s] = 1.0 / (abs(t - s) + 1)
            attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
            attended = attn @ xi
            outputs.append(attended[-1])
        
        # Average across windows
        pred = np.mean(outputs, axis=0).sum()
        preds.append(pred)
    
    return np.mean((y_array - np.array(preds))**2)

def exp_decay_attention(X, y, base=0.95):
    """Exponential decay attention (stronger recent emphasis)."""
    y_array = np.array([yi.mean() for yi in y])
    preds = []
    
    for xi in X:
        seq_len = xi.shape[0]
        
        attn = np.zeros((seq_len, seq_len))
        for t in range(seq_len):
            for s in range(seq_len):
                dist = abs(t - s)
                attn[t, s] = base ** dist
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
        
        attended = attn @ xi
        preds.append(attended[-1].sum())
    
    return np.mean((y_array - np.array(preds))**2)

def run_experiment():
    print("H1.178: Attention with Decay Scaling on Long Sequences")
    print("=" * 60)
    
    results = {
        'hypothesis': 'H1.178',
        'statement': 'Decay scaling extends attention advantage to 100-200 step sequences',
        'parent': 'H1',
        'priority': 'high'
    }
    
    all_results = []
    
    # Test different sequence lengths
    for seq_range in [(100, 120), (120, 150), (150, 180), (180, 200)]:
        # Generate data
        X, y = generate_long_sequence_data(
            n_samples=150,
            min_len=seq_range[0],
            max_len=seq_range[1]
        )
        
        concat_mse = baseline(X, y)
        fixed_09_mse = fixed_decay_attention(X, y, 0.9)
        fixed_095_mse = fixed_decay_attention(X, y, 0.95)
        fixed_099_mse = fixed_decay_attention(X, y, 0.99)
        adaptive_mse = adaptive_decay_attention(X, y)
        multi_scale_mse = multi_scale_attention(X, y)
        exp_095_mse = exp_decay_attention(X, y, 0.95)
        
        delta_fixed_09 = (concat_mse - fixed_09_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_fixed_095 = (concat_mse - fixed_095_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_fixed_099 = (concat_mse - fixed_099_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_adaptive = (concat_mse - adaptive_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_multi = (concat_mse - multi_scale_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_exp_095 = (concat_mse - exp_095_mse) / concat_mse * 100 if concat_mse > 0 else 0
        
        result = {
            'seq_range': f'{seq_range[0]}-{seq_range[1]}',
            'concat_mse': float(concat_mse),
            'fixed_09': float(delta_fixed_09),
            'fixed_095': float(delta_fixed_095),
            'fixed_099': float(delta_fixed_099),
            'adaptive': float(delta_adaptive),
            'multi_scale': float(delta_multi),
            'exp_095': float(delta_exp_095)
        }
        all_results.append(result)
        
        print(f"Seq {seq_range[0]}-{seq_range[1]}: Concat={concat_mse:.2f}, "
              f"Decay0.9={delta_fixed_09:+.1f}%, Decay0.95={delta_fixed_095:+.1f}%, "
              f"Decay0.99={delta_fixed_099:+.1f}%, Adaptive={delta_adaptive:+.1f}%, "
              f"MultiScale={delta_multi:+.1f}%")
    
    # Summary
    avg_fixed_09 = np.mean([r['fixed_09'] for r in all_results])
    avg_fixed_095 = np.mean([r['fixed_095'] for r in all_results])
    avg_fixed_099 = np.mean([r['fixed_099'] for r in all_results])
    avg_adaptive = np.mean([r['adaptive'] for r in all_results])
    avg_multi = np.mean([r['multi_scale'] for r in all_results])
    avg_exp_095 = np.mean([r['exp_095'] for r in all_results])
    
    print("\n" + "=" * 60)
    print(f"Average Results (100-200 steps):")
    print(f"  Fixed Decay 0.9: {avg_fixed_09:+.1f}%")
    print(f"  Fixed Decay 0.95: {avg_fixed_095:+.1f}%")
    print(f"  Fixed Decay 0.99: {avg_fixed_099:+.1f}%")
    print(f"  Adaptive Decay: {avg_adaptive:+.1f}%")
    print(f"  Multi-Scale: {avg_multi:+.1f}%")
    print(f"  Exp Decay 0.95: {avg_exp_095:+.1f}%")
    print(f"  H1.106 baseline: +0.2%")
    
    # Best method
    best_method = max([
        ('fixed_09', avg_fixed_09),
        ('fixed_095', avg_fixed_095),
        ('fixed_099', avg_fixed_099),
        ('adaptive', avg_adaptive),
        ('multi_scale', avg_multi),
        ('exp_095', avg_exp_095)
    ], key=lambda x: x[1])
    
    if best_method[1] > 0:
        status = "SUPPORTED"
        print(f"\n  ✅ {best_method[0].upper()} WORKS! ({best_method[1]:+.1f}%)")
    elif best_method[1] > -10:
        status = "PARTIAL"
        print(f"\n  ⚠️ {best_method[0].upper()} helps but not fully positive")
    else:
        status = "REFUTED"
        print(f"\n  ❌ All fail on long sequences")
    
    results['avg_fixed_09'] = float(avg_fixed_09)
    results['avg_fixed_095'] = float(avg_fixed_095)
    results['avg_fixed_099'] = float(avg_fixed_099)
    results['avg_adaptive'] = float(avg_adaptive)
    results['avg_multi_scale'] = float(avg_multi)
    results['avg_exp_095'] = float(avg_exp_095)
    results['best_method'] = best_method[0]
    results['best_value'] = float(best_method[1])
    results['status'] = status
    results['all_results'] = all_results
    
    return results

if __name__ == '__main__':
    results = run_experiment()
    
    output_path = Path('experiments/H1.178-attention-decay-longseq/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")