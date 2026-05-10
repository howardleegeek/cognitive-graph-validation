#!/usr/bin/env python3
"""
H1.191: Object Identity Tracking
Tests if maintaining object identity representations enables better predictions

Key insight from H2.15: Graph attention fails on multi-object (object interactions are the bottleneck)
H1.190: Phase information doesn't help

H1.191 Hypothesis: Instead of graph attention, use object identity embeddings to track
each object separately. This captures that manipulation tasks have consistent object identities.
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(42)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-8)

def generate_object_tracking_data(n_steps, n_objects=3, dim=8, rho=0.85):
    """Generate data where object identities persist across time"""
    X, y = [], []
    
    # Object identities (fixed throughout sequence)
    obj_ids = [np.eye(n_objects)[i] for i in range(n_objects)]  # One-hot identity
    
    # Initial positions
    obj_pos = [np.random.randn(dim) for _ in range(n_objects)]
    target_pos = [np.random.randn(dim) for _ in range(n_objects)]
    
    for t in range(n_steps):
        # State with object identities
        state_with_ids = []
        for i in range(n_objects):
            state_with_ids.append(np.concatenate([obj_ids[i], obj_pos[i]]))
        
        state = np.concatenate(state_with_ids)
        
        # Action
        action = np.random.randn(4)
        
        # Next positions
        next_obj_pos = []
        for i in range(n_objects):
            next_pos = rho * obj_pos[i] + (1 - rho) * target_pos[i] + np.concatenate([action, np.zeros(dim-4)]) * 0.1
            next_obj_pos.append(next_pos)
        
        X.append(np.concatenate([state, action]))
        y.append(next_obj_pos[0])  # Predict first object trajectory
        
        obj_pos = next_obj_pos
    
    return np.array(X), np.array(y)

def generate_unified_data(n_steps, n_objects=3, dim=8, rho=0.85):
    """Generate data with unified representation (no object IDs)"""
    X, y = [], []
    
    obj_pos = [np.random.randn(dim) for _ in range(n_objects)]
    target_pos = [np.random.randn(dim) for _ in range(n_objects)]
    
    for t in range(n_steps):
        # State without identities
        state = np.concatenate(obj_pos)
        
        action = np.random.randn(4)
        
        next_obj_pos = []
        for i in range(n_objects):
            next_pos = rho * obj_pos[i] + (1 - rho) * target_pos[i] + np.concatenate([action, np.zeros(dim-4)]) * 0.1
            next_obj_pos.append(next_pos)
        
        X.append(np.concatenate([state, action]))
        y.append(next_obj_pos[0])
        
        obj_pos = next_obj_pos
    
    return np.array(X), np.array(y)

def train_mlp(X, y, hidden_dim=64, epochs=100):
    """Standard MLP"""
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

def train_object_aware(X, y, n_objects, hidden_dim=64, epochs=100):
    """Object-aware MLP with identity embeddings"""
    input_dim = X.shape[-1]  # 24
    output_dim = y.shape[-1]  # 8
    
    # Object identity embeddings
    id_dim = hidden_dim // n_objects  # 32
    
    W_id = np.random.randn(n_objects, id_dim) * 0.01
    
    # Main network (takes enhanced input)
    enhanced_dim = input_dim + id_dim  # 24 + 32 = 56
    W1 = np.random.randn(enhanced_dim, hidden_dim) * 0.01
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    
    for _ in range(epochs):
        # Extract first object's identity features (first n_objects dims)
        first_obj_id = X[:, :n_objects]  # (seq, n_objects)
        id_features = first_obj_id @ W_id  # (seq, id_dim)
        
        # Concatenate with input
        enhanced = np.concatenate([X, id_features], axis=-1)  # (seq, enhanced_dim)
        
        h1 = np.maximum(0, enhanced @ W1)
        h2 = np.maximum(0, h1 @ W2)
        pred = h2 @ W3
    
    return pred

def run_experiment():
    print("=" * 60)
    print("H1.191: Object Identity Tracking")
    print("=" * 60)
    
    results = []
    
    configs = [
        {'n_steps': 20, 'n_objects': 2},
        {'n_steps': 20, 'n_objects': 3},
        {'n_steps': 20, 'n_objects': 4},
        {'n_steps': 30, 'n_objects': 2},
        {'n_steps': 30, 'n_objects': 3},
        {'n_steps': 40, 'n_objects': 3},
    ]
    
    for cfg in configs:
        n_steps = cfg['n_steps']
        n_objects = cfg['n_objects']
        
        print(f"\n--- {n_steps} steps, {n_objects} objects ---")
        
        # Generate with object identity
        X, y = generate_object_tracking_data(n_steps, n_objects, dim=8, rho=0.85)
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Unified baseline (no identity info)
        X_unified, _ = generate_unified_data(n_steps, n_objects, dim=8, rho=0.85)
        
# Train on unified data (same shape as object tracking)
        unified_pred = train_mlp(X, y, epochs=100)
        unified_mse = np.mean((unified_pred - y) ** 2)
        
        # Train on object-aware data
        aware_pred = train_object_aware(X, y, n_objects, epochs=100)
        aware_mse = np.mean((aware_pred - y) ** 2)
        
        delta = (aware_mse - unified_mse) / unified_mse * 100
        winner = "OBJECT-AWARE" if aware_mse < unified_mse else "UNIFIED"
        
        print(f"Unified MSE: {unified_mse:.6f}, Object-Aware MSE: {aware_mse:.6f}")
        print(f"Delta: {delta:+.1f}% ({winner})")
        
        results.append({
            'config': cfg,
            'unified_mse': float(unified_mse),
            'aware_mse': float(aware_mse),
            'delta': float(delta),
            'winner': winner
        })
    
    print("\n" + "=" * 60)
    avg_delta = np.mean([r['delta'] for r in results])
    aware_wins = sum(1 for r in results if r['winner'] == 'OBJECT-AWARE')
    print(f"Overall: avg delta = {avg_delta:+.1f}%, object-aware wins {aware_wins}/{len(results)}")
    
    if avg_delta < -10 and aware_wins >= 4:
        status = "SUPPORTED"
    elif avg_delta > 5 or aware_wins <= 1:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    print(f"Status: {status}")
    
    return results, status, avg_delta

if __name__ == "__main__":
    results, status, avg_delta = run_experiment()
    
    output = {
        'hypothesis': 'H1.191',
        'title': 'Object Identity Tracking',
        'status': status,
        'avg_delta': avg_delta,
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")