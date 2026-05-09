#!/usr/bin/env python3
"""
H3.84: Graph + Attention Hybrid for Multi-Object Tasks

Based on findings:
- H3.83: Attention (-47.0%) fails on multi-object with interactions, concat wins
- H2.9: Graph (+50.4%) excels at multi-object compositional temporal reasoning
- H3.82: Multi-Scale (+74.1%) best for generalization

Hypothesis: Graph + Attention hybrid should combine:
- Graph: Object-level structure (from H2.9)
- Attention: Temporal dynamics (from H3.82)
"""

import numpy as np
import json
from pathlib import Path

def generate_multiobject_data(n_samples=200, n_timesteps=20, n_objects=3, seed=42):
    np.random.seed(seed)
    X, y = [], []
    
    for _ in range(n_samples):
        seq_len = n_timesteps + np.random.randint(-3, 4)
        n_obj = n_objects
        
        # State: [x, y] position per object
        state = np.random.randn(seq_len, 2 * n_obj)
        
        # Add temporal correlation (physics)
        for t in range(1, seq_len):
            state[t, :n_obj] += 0.7 * state[t-1, :n_obj]
            state[t, n_obj:] += 0.7 * state[t-1, n_obj:]
        
        # Object interactions (force field)
        interaction = np.zeros((seq_len, n_obj))
        for i in range(n_obj):
            for j in range(n_obj):
                if i != j:
                    interaction[:, i] += 0.1 * np.sum(state[:, n_obj*j:n_obj*(j+1)], axis=1)
        
        # Target: predict next interaction
        target = interaction[1:].sum(axis=1)
        
        X.append({
            'state': state[:-1],
            'n_obj': n_obj
        })
        y.append(target)
    
    return X, y

def simple_predict(X, y):
    """Simple mean prediction baseline."""
    y_array = np.array([yi.sum() for yi in y])
    baseline_pred = np.ones_like(y_array) * y_array.mean()
    return np.mean((y_array - baseline_pred)**2)

def graph_model(X, y):
    """Graph model with message passing."""
    y_array = np.array([yi.sum() for yi in y])
    preds = []
    
    for xi in X:
        n_obj = xi['n_obj']
        state = xi['state']
        seq_len = state.shape[0]
        
        # Object embeddings
        obj_emb = state[:, :2*n_obj].reshape(seq_len, n_obj, 2).mean(axis=2)  # (seq, n_obj)
        
        # Simple message passing (3 passes)
        adj = np.ones((n_obj, n_obj)) / n_obj  # Fully connected
        emb = obj_emb[-1]  # Start from last timestep
        for _ in range(3):
            new_emb = adj @ emb
            emb = 0.5 * emb + 0.5 * new_emb
        
        preds.append(emb.sum())
    
    return np.mean((y_array - np.array(preds))**2)

def attention_model(X, y):
    """Temporal attention model."""
    y_array = np.array([yi.sum() for yi in y])
    preds = []
    
    for xi in X:
        state = xi['state']
        seq_len = state.shape[0]
        n_obj = xi['n_obj']
        
        # Object features at each timestep
        obj_feat = state[:, :2*n_obj].reshape(seq_len, n_obj, 2).mean(axis=2)  # (seq, n_obj)
        
        # Attention weights (decay)
        attn = np.zeros((seq_len, seq_len))
        for t in range(seq_len):
            for s in range(seq_len):
                attn[t, s] = np.exp(-0.1 * abs(t - s))
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
        
        # Apply attention
        attended = attn @ obj_feat  # (seq, n_obj)
        
        # Prediction from last timestep
        preds.append(attended[-1].sum())
    
    return np.mean((y_array - np.array(preds))**2)

def hybrid_model(X, y):
    """Graph + Attention Hybrid."""
    y_array = np.array([yi.sum() for yi in y])
    preds = []
    
    for xi in X:
        n_obj = xi['n_obj']
        state = xi['state']
        seq_len = state.shape[0]
        
        # Object embeddings
        obj_feat = state[:, :2*n_obj].reshape(seq_len, n_obj, 2).mean(axis=2)
        
        # Graph component
        adj = np.ones((n_obj, n_obj)) / n_obj
        emb = obj_feat[-1].copy()
        for _ in range(3):
            new_emb = adj @ emb
            emb = 0.5 * emb + 0.5 * new_emb
        graph_out = emb.sum()
        
        # Attention component
        attn = np.zeros((seq_len, seq_len))
        for t in range(seq_len):
            for s in range(seq_len):
                attn[t, s] = np.exp(-0.1 * abs(t - s))
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)
        attended = (attn @ obj_feat)[-1].sum()
        
        # Combine (weighted average)
        hybrid_pred = 0.5 * graph_out + 0.5 * attended
        preds.append(hybrid_pred)
    
    return np.mean((y_array - np.array(preds))**2)

def run_experiment():
    print("H3.84: Graph + Attention Hybrid for Multi-Object Tasks")
    print("=" * 60)
    
    results = {
        'hypothesis': 'H3.84',
        'statement': 'Graph + Attention hybrid outperforms both individually on multi-object tasks',
        'parent': 'H3',
        'priority': 'high'
    }
    
    all_results = []
    
    for n_objects in [2, 3, 4, 5]:
        # Generate data
        X, y = generate_multiobject_data(
            n_samples=150,
            n_timesteps=20,
            n_objects=n_objects
        )
        
        # Compute predictions
        concat_mse = simple_predict(X, y)
        graph_mse = graph_model(X, y)
        attn_mse = attention_model(X, y)
        hybrid_mse = hybrid_model(X, y)
        
        delta_graph = (concat_mse - graph_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_attn = (concat_mse - attn_mse) / concat_mse * 100 if concat_mse > 0 else 0
        delta_hybrid = (concat_mse - hybrid_mse) / concat_mse * 100 if concat_mse > 0 else 0
        
        result = {
            'n_objects': n_objects,
            'concat_mse': float(concat_mse),
            'graph_mse': float(graph_mse),
            'attn_mse': float(attn_mse),
            'hybrid_mse': float(hybrid_mse),
            'graph_vs_concat': float(delta_graph),
            'attn_vs_concat': float(delta_attn),
            'hybrid_vs_concat': float(delta_hybrid)
        }
        all_results.append(result)
        
        print(f"Objects={n_objects}: Concat={concat_mse:.4f}, Graph={delta_graph:+.1f}%, "
              f"Attn={delta_attn:+.1f}%, Hybrid={delta_hybrid:+.1f}%")
    
    # Summary
    avg_graph = np.mean([r['graph_vs_concat'] for r in all_results])
    avg_attn = np.mean([r['attn_vs_concat'] for r in all_results])
    avg_hybrid = np.mean([r['hybrid_vs_concat'] for r in all_results])
    
    print("\n" + "=" * 60)
    print(f"Average Results:")
    print(f"  Graph Only: {avg_graph:+.1f}%")
    print(f"  Attention Only: {avg_attn:+.1f}%")
    print(f"  Graph + Attention Hybrid: {avg_hybrid:+.1f}%")
    print(f"  H3.83 baseline (attention alone): -47.0%")
    
    if avg_hybrid > avg_graph and avg_hybrid > avg_attn and avg_hybrid > 0:
        status = "SUPPORTED"
        print(f"\n  ✅ GRAPH + ATTENTION HYBRID WORKS!")
    elif avg_hybrid > avg_graph and avg_hybrid > avg_attn:
        status = "PARTIAL"
        print(f"\n  ⚠️ Hybrid best but still negative")
    elif avg_graph > avg_hybrid and avg_graph > avg_attn:
        status = "REFUTED (graph alone wins)"
        print(f"\n  ❌ Hybrid doesn't add over graph alone")
    elif avg_attn > 0:
        status = "SUPPORTED (attention wins)"
        print(f"\n  ⚠️ Attention helps but hybrid doesn't add")
    else:
        status = "REFUTED"
        print(f"\n  ❌ Neither helps")
    
    results['avg_graph'] = float(avg_graph)
    results['avg_attn'] = float(avg_attn)
    results['avg_hybrid'] = float(avg_hybrid)
    results['status'] = status
    results['all_results'] = all_results
    
    return results

if __name__ == '__main__':
    results = run_experiment()
    
    # Save results
    output_path = Path('experiments/H3.84-graph-attn-hybrid-multiobject/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")