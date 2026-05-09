#!/usr/bin/env python3
"""
H1.176: Hierarchical Multi-Object Attention for Complex Interactions

Based on findings:
- H1.80: Hierarchical planning (+86.6%)
- H1.104: Hierarchical compositional (+34.9%)
- H1.114: Hierarchical attention ALOHA (+94.3%)
- H3.83: Attention (-47.0%) fails on multi-object with interactions
- H3.84: Attention (+25.2%) wins on simple multi-object

Hypothesis: Hierarchical attention (within-object then across-object) 
can handle complex interactions better than flat attention.
"""

import numpy as np
import json
from pathlib import Path

def generate_complex_multiobject(n_samples=200, n_timesteps=20, n_objects=4, seed=42):
    np.random.seed(seed)
    X, y = [], []
    
    for _ in range(n_samples):
        seq_len = n_timesteps + np.random.randint(-3, 4)
        
        # Object states with complex interactions
        obj_states = []
        for i in range(n_objects):
            pos = np.cumsum(np.random.randn(seq_len) * 0.3)
            vel = np.cumsum(np.random.randn(seq_len) * 0.1)
            obj_states.append(np.stack([pos, vel], axis=1))
        
        state = np.stack(obj_states, axis=2)  # (seq, 2, n_obj)
        
        # Complex interactions: each object affects others
        interaction = np.zeros((seq_len, n_objects))
        for t in range(seq_len):
            for i in range(n_objects):
                for j in range(n_objects):
                    if i != j:
                        # Complex: distance-based + velocity-dependent
                        dist = np.abs(state[t, 0, i] - state[t, 0, j])
                        vel_diff = np.abs(state[t, 1, i] - state[t, 1, j])
                        interaction[t, i] += 0.3 * np.exp(-dist) * (1 + vel_diff)
        
        # Target: predict next state
        target = state[1:, 0, :].sum(axis=1)  # Next positions
        
        X.append({
            'state': state,  # (seq, 2, n_obj)
            'interaction': interaction
        })
        y.append(target)
    
    return X, y

def baseline(X, y):
    """Mean prediction baseline."""
    y_array = np.array([yi.sum() for yi in y])
    pred = np.ones_like(y_array) * y_array.mean()
    return np.mean((y_array - pred)**2)

def flat_attention(X, y):
    """Flat cross-object attention."""
    y_array = np.array([yi.sum() for yi in y])
    preds = []
    
    for xi in X:
        state = xi['state']  # (seq, 2, n_obj)
        seq_len, _, n_obj = state.shape
        
        # Flatten to (seq, 2*n_obj)
        flat = state.reshape(seq_len, -1)
        
        # Temporal attention
        attn = np.zeros((seq_len, seq_len))
        for t in range(seq_len):
            for s in range(seq_len):
                attn[t, s] = np.exp(-0.05 * abs(t - s))
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
        
        # Apply
        attended = attn @ flat
        preds.append(attended[-1].sum())
    
    return np.mean((y_array - np.array(preds))**2)

def hierarchical_attention(X, y):
    """Hierarchical: within-object then across-object."""
    y_array = np.array([yi.sum() for yi in y])
    preds = []
    
    for xi in X:
        state = xi['state']  # (seq, 2, n_obj)
        seq_len, feat_dim, n_obj = state.shape
        
        # Level 1: Within-object attention (temporal)
        within = np.zeros_like(state)
        for obj in range(n_obj):
            obj_seq = state[:, :, obj]  # (seq, 2)
            attn = np.zeros((seq_len, seq_len))
            for t in range(seq_len):
                for s in range(seq_len):
                    attn[t, s] = np.exp(-0.05 * abs(t - s))
            attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
            within[:, :, obj] = attn @ obj_seq
        
        # Level 2: Across-object attention
        across = np.zeros((seq_len, feat_dim, n_obj))
        for t in range(seq_len):
            obj_emb = within[t].T  # (n_obj, feat_dim)
            attn = np.exp(-0.3 * np.abs(np.arange(n_obj)[:, None] - np.arange(n_obj)[None, :]))
            attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
            across_obj = attn @ obj_emb  # (n_obj, feat_dim)
            across[t] = across_obj.T
        
        # Final: temporal aggregation
        final = across[-1].sum()
        preds.append(final)
    
    return np.mean((y_array - np.array(preds))**2)

def hierarchical_with_decay(X, y):
    """Hierarchical with recency decay."""
    y_array = np.array([yi.sum() for yi in y])
    preds = []
    
    for xi in X:
        state = xi['state']  # (seq, 2, n_obj)
        seq_len, feat_dim, n_obj = state.shape
        
        # Level 1: Within-object with decay
        within = np.zeros_like(state)
        decay = 0.97
        for obj in range(n_obj):
            obj_seq = state[:, :, obj]  # (seq, 2)
            for t in range(seq_len):
                weighted = np.zeros(feat_dim)
                weight_sum = 0
                for s in range(seq_len):
                    w = decay ** (t - s) if s <= t else decay ** (s - t - 1)
                    weighted += w * obj_seq[s]
                    weight_sum += w
                within[t, :, obj] = weighted / (weight_sum + 1e-8)
        
        # Level 2: Object interactions
        across = np.zeros((seq_len, feat_dim, n_obj))
        for t in range(seq_len):
            obj_emb = within[t].T  # (n_obj, feat_dim)
            attn = np.exp(-0.2 * np.abs(np.arange(n_obj)[:, None] - np.arange(n_obj)[None, :]))
            attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
            across_obj = attn @ obj_emb
            across[t] = across_obj.T
        
        # Recency-weighted final
        recency_weight = decay ** np.arange(seq_len)[::-1]
        recency_weight = recency_weight / (recency_weight.sum() + 1e-8)
        
        weighted_final = 0
        for t in range(seq_len):
            weighted_final += recency_weight[t] * across[t].sum()
        
        preds.append(weighted_final)
    
    return np.mean((y_array - np.array(preds))**2)

def run_experiment():
    print("H1.176: Hierarchical Multi-Object Attention for Complex Interactions")
    print("=" * 60)
    
    results = {
        'hypothesis': 'H1.176',
        'statement': 'Hierarchical attention handles complex multi-object interactions',
        'parent': 'H1',
        'priority': 'high'
    }
    
    all_results = []
    
    for n_objects in [2, 3, 4, 5]:
        # Generate data
        X, y = generate_complex_multiobject(
            n_samples=150,
            n_timesteps=20,
            n_objects=n_objects
        )
        
        concat_mse = baseline(X, y)
        flat_mse = flat_attention(X, y)
        hier_mse = hierarchical_attention(X, y)
        hier_decay_mse = hierarchical_with_decay(X, y)
        
        delta_flat = (concat_mse - flat_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_hier = (concat_mse - hier_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_hier_decay = (concat_mse - hier_decay_mse) / concat_mse * 100 if concat_mse > 0 else 0
        
        result = {
            'n_objects': n_objects,
            'concat_mse': float(concat_mse),
            'flat_mse': float(flat_mse),
            'hier_mse': float(hier_mse),
            'hier_decay_mse': float(hier_decay_mse),
            'flat_vs_concat': float(delta_flat),
            'hier_vs_concat': float(delta_hier),
            'hier_decay_vs_concat': float(delta_hier_decay)
        }
        all_results.append(result)
        
        print(f"Objects={n_objects}: Concat={concat_mse:.2f}, Flat={delta_flat:+.1f}%, "
              f"Hier={delta_hier:+.1f}%, Hier+Decay={delta_hier_decay:+.1f}%")
    
    # Summary
    avg_flat = np.mean([r['flat_vs_concat'] for r in all_results])
    avg_hier = np.mean([r['hier_vs_concat'] for r in all_results])
    avg_hier_decay = np.mean([r['hier_decay_vs_concat'] for r in all_results])
    
    print("\n" + "=" * 60)
    print(f"Average Results:")
    print(f"  Flat Attention: {avg_flat:+.1f}%")
    print(f"  Hierarchical: {avg_hier:+.1f}%")
    print(f"  Hierarchical + Decay: {avg_hier_decay:+.1f}%")
    print(f"  H3.83 baseline: -47.0%")
    
    # Best method
    best_method = 'hierarchical_decay' if avg_hier_decay > avg_hier and avg_hier_decay > avg_flat else \
                  ('hierarchical' if avg_hier > avg_flat else 'flat')
    best_val = max(avg_flat, avg_hier, avg_hier_decay)
    
    if best_val > 0:
        status = "SUPPORTED"
        print(f"\n  ✅ {best_method.upper()} WORKS! ({best_val:+.1f}%)")
    elif best_val > -30:
        status = "PARTIAL"
        print(f"\n  ⚠️ {best_method.upper()} helps but not fully positive")
    else:
        status = "REFUTED"
        print(f"\n  ❌ All fail on complex interactions")
    
    results['avg_flat'] = float(avg_flat)
    results['avg_hier'] = float(avg_hier)
    results['avg_hier_decay'] = float(avg_hier_decay)
    results['best_method'] = best_method
    results['status'] = status
    results['all_results'] = all_results
    
    return results

if __name__ == '__main__':
    results = run_experiment()
    
    output_path = Path('experiments/H1.176-hierarchical-multiobject/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")