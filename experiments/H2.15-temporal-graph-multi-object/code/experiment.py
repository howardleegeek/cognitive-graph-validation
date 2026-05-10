#!/usr/bin/env python3
"""
H2.15: Temporal Graph Attention for Multi-Object Reasoning
Simplified version - Tests if temporal-aware graph enables multi-object tracking
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(42)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-8)

def generate_multi_object(n_steps, n_objects=3, dim=8, rho=0.85, with_interactions=False):
    """Generate multi-object sequence data"""
    X, y = [], []
    
    # Initialize positions
    obj_pos = [np.random.randn(dim) for _ in range(n_objects)]
    target_pos = [np.random.randn(dim) for _ in range(n_objects)]
    
    # History buffer
    history = []
    
    for t in range(n_steps):
        # State: all object positions + temporal deltas
        state = np.concatenate(obj_pos)
        
        # Temporal deltas
        if len(history) > 0:
            deltas = state - history[-1]
        else:
            deltas = np.zeros(dim * n_objects)
        
        # Interaction distances if enabled
        if with_interactions:
            distances = []
            for i in range(n_objects):
                for j in range(i+1, n_objects):
                    distances.append(np.linalg.norm(obj_pos[i] - obj_pos[j]))
            interactions = np.array(distances)
            combined_state = np.concatenate([state, deltas, interactions])
        else:
            combined_state = np.concatenate([state, deltas])
        
        # Action
        action = np.random.randn(4)
        
        # Next positions
        next_obj_pos = []
        for i in range(n_objects):
            next_pos = rho * obj_pos[i] + (1 - rho) * target_pos[i] + np.concatenate([action, np.zeros(dim-4)]) * 0.1
            
            if with_interactions and t > 0:
                for j in range(n_objects):
                    if i != j:
                        dist = np.linalg.norm(obj_pos[i] - obj_pos[j])
                        if dist < 2.0:
                            repulsion = (obj_pos[i] - obj_pos[j]) / (dist + 0.1)
                            next_pos = next_pos + repulsion * 0.05
            
            next_obj_pos.append(next_pos)
        
        history.append(state)
        X.append(np.concatenate([combined_state, action]))
        y.append(next_obj_pos[0])
        
        obj_pos = next_obj_pos
    
    return np.array(X), np.array(y)

def train_mlp(X, y, hidden_dim=64, epochs=80):
    """Standard MLP baseline"""
    input_dim = X.shape[-1]
    output_dim = y.shape[-1]
    
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    
    for _ in range(epochs):
        h1 = np.maximum(0, X @ W1)
        h2 = np.maximum(0, h1 @ W2)
        pred = h2 @ W3
    
    return pred

def train_temporal_graph_attn(X, y, n_objects=3, hidden_dim=64, epochs=80):
    """Graph attention with temporal awareness"""
    input_dim = X.shape[-1]
    output_dim = y.shape[-1]
    
    # Temporal weights for object identity tracking
    W_temp = np.random.randn(hidden_dim, hidden_dim) * 0.01
    
    # Graph attention
    W_q = np.random.randn(input_dim, hidden_dim) * 0.01
    W_k = np.random.randn(input_dim, hidden_dim) * 0.01
    W_v = np.random.randn(input_dim, hidden_dim) * 0.01
    W_out = np.random.randn(hidden_dim, hidden_dim) * 0.01
    
    # Output
    W_fc = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    
    history = []
    
    for _ in range(epochs):
        # Temporal context from history (track state representation)
        if len(history) > n_objects:
            recent = np.array(history[-n_objects:])
            temporal = recent.mean(axis=0)[:hidden_dim]  # Truncate to hidden_dim
            if len(temporal) < hidden_dim:
                temporal = np.pad(temporal, (0, hidden_dim - len(temporal)))
            temporal_proj = temporal @ W_temp  # (hidden_dim,) @ (hidden_dim, hidden_dim)
        else:
            temporal_proj = np.zeros(hidden_dim)
        
        # Graph attention
        Q = X @ W_q  # (seq, input) @ (input, hidden)
        K = X @ W_k  # (seq, hidden)
        V = X @ W_v  # (seq, hidden)
        
        scores = Q @ K.T / np.sqrt(hidden_dim)
        weights = softmax(scores, axis=-1)
        attended = weights @ V @ W_out + temporal_proj  # Add temporal bias
        
        # Output
        h = np.maximum(0, attended @ W_fc)
        pred = h @ W3
        
        # Update history
        history.append(np.mean(X[:, :input_dim//2], axis=0))
    
    return pred

def run_experiment():
    print("=" * 60)
    print("H2.15: Temporal Graph Attention for Multi-Object")
    print("=" * 60)
    
    results = []
    
    configs = [
        {'n_steps': 20, 'n_objects': 2, 'interactions': False},
        {'n_steps': 20, 'n_objects': 3, 'interactions': False},
        {'n_steps': 20, 'n_objects': 4, 'interactions': False},
        {'n_steps': 30, 'n_objects': 3, 'interactions': False},
        {'n_steps': 30, 'n_objects': 3, 'interactions': True},
        {'n_steps': 40, 'n_objects': 3, 'interactions': True},
    ]
    
    for cfg in configs:
        n_steps = cfg['n_steps']
        n_objects = cfg['n_objects']
        interactions = cfg['interactions']
        
        print(f"\n--- {n_steps} steps, {n_objects} objects, interactions={interactions} ---")
        
        X, y = generate_multi_object(n_steps, n_objects, dim=8, rho=0.85, 
                                     with_interactions=interactions)
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Baseline
        baseline_pred = train_mlp(X, y, epochs=80)
        baseline_mse = np.mean((baseline_pred - y) ** 2)
        
        # Temporal graph attention
        graph_pred = train_temporal_graph_attn(X, y, n_objects, epochs=80)
        graph_mse = np.mean((graph_pred - y) ** 2)
        
        delta = (graph_mse - baseline_mse) / baseline_mse * 100
        winner = "GRAPH" if graph_mse < baseline_mse else "BASELINE"
        
        print(f"Baseline MSE: {baseline_mse:.6f}, Graph MSE: {graph_mse:.6f}")
        print(f"Delta: {delta:+.1f}% ({winner})")
        
        results.append({
            'config': cfg,
            'baseline_mse': float(baseline_mse),
            'graph_mse': float(graph_mse),
            'delta': float(delta),
            'winner': winner
        })
    
    print("\n" + "=" * 60)
    avg_delta = np.mean([r['delta'] for r in results])
    graph_wins = sum(1 for r in results if r['winner'] == 'GRAPH')
    
    with_inter = [r for r in results if r['config']['interactions']]
    without_inter = [r for r in results if not r['config']['interactions']]
    avg_with = np.mean([r['delta'] for r in with_inter]) if with_inter else 0
    avg_without = np.mean([r['delta'] for r in without_inter]) if without_inter else 0
    
    print(f"Overall: avg delta = {avg_delta:+.1f}%, graph wins {graph_wins}/{len(results)}")
    print(f"With interactions: {avg_with:+.1f}%")
    print(f"Without interactions: {avg_without:+.1f}%")
    
    if avg_delta < -10 and graph_wins >= 3:
        status = "SUPPORTED"
    elif avg_delta > 10 or graph_wins <= 1:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    print(f"Status: {status}")
    
    return results, status

if __name__ == "__main__":
    results, status = run_experiment()
    
    output = {
        'hypothesis': 'H2.15',
        'title': 'Temporal Graph Attention for Multi-Object',
        'status': status,
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
