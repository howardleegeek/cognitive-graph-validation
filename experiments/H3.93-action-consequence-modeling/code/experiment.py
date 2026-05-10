#!/usr/bin/env python3
"""
H3.93: Action Consequence Modeling
Tests if explicit action-consequence modeling (causal structure) outperforms sequential prediction

Key insight from recent experiments: Task structure matters more than mechanism.
Hypothesis: Real robot data has causal structure (action → consequence) that enables learning.
If we model this explicitly, synthetic data should work better.
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(42)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-8)

def generate_causal_data(n_steps, dim=16, rho=0.85):
    """Generate data with explicit action-consequence structure"""
    X, causal_features, y = [], [], []
    
    state = np.random.randn(dim)
    
    for t in range(n_steps):
        # Random action
        action = np.random.randn(8)
        
        # Causal features: action influence on state
        # State change = f(action) + noise
        action_effect = np.zeros(dim)
        action_effect[:8] = action[:8] * 0.5
        
        # Next state with causal structure
        next_state = rho * state + (1 - rho) * np.random.randn(dim) + action_effect + np.random.randn(dim) * 0.05
        
        # Input: [state, action]
        # Causal features: [action_effect, next_state - state]  
        # Target: next_state
        
        X.append(np.concatenate([state, action]))
        causal_features.append(np.concatenate([action_effect, next_state - state]))
        y.append(next_state)
        
        state = next_state
    
    return np.array(X), np.array(causal_features), np.array(y)

def generate_sequential_data(n_steps, dim=16, rho=0.85):
    """Generate standard sequential data without explicit causal structure"""
    X, y = [], []
    
    state = np.random.randn(dim)
    
    for t in range(n_steps):
        action = np.random.randn(8)
        next_state = rho * state + (1 - rho) * np.random.randn(dim) + np.concatenate([action, np.zeros(dim-8)]) * 0.1
        
        X.append(np.concatenate([state, action]))
        y.append(next_state)
        
        state = next_state
    
    return np.array(X), np.array(y)

def train_sequential(X, y, hidden_dim=64, epochs=100):
    """Standard sequential prediction"""
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

def train_causal(X, causal, y, hidden_dim=64, epochs=100):
    """Causal prediction with action-consequence modeling"""
    input_dim = X.shape[-1]
    causal_dim = causal.shape[-1]
    output_dim = y.shape[-1]
    
    # Action-to-consequence predictor
    W_action = np.random.randn(8, hidden_dim) * 0.01
    W_state = np.random.randn(input_dim - 8, hidden_dim) * 0.01
    W_causal = np.random.randn(causal_dim, hidden_dim) * 0.01
    
    # Fusion
    W_fusion = np.random.randn(hidden_dim * 3, hidden_dim) * 0.01
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    
    for _ in range(epochs):
        # Extract components
        action = X[:, -8:]
        state = X[:, :-8]
        
        # Causal features from action
        action_effect = action @ W_action
        
        # State encoding
        state_enc = state @ W_state
        
        # Causal encoding
        causal_enc = causal @ W_causal
        
        # Fusion
        combined = np.concatenate([action_effect, state_enc, causal_enc], axis=-1)
        h = np.maximum(0, combined @ W_fusion)
        h = np.maximum(0, h @ W2)
        pred = h @ W3
    
    return pred

def run_experiment():
    print("=" * 60)
    print("H3.93: Action Consequence Modeling")
    print("=" * 60)
    
    results = []
    
    for n_steps in [20, 30, 40, 50]:
        print(f"\n--- {n_steps} steps ---")
        
        # Generate causal data
        X, causal, y = generate_causal_data(n_steps, dim=16, rho=0.85)
        print(f"Data shape: X={X.shape}, causal={causal.shape}, y={y.shape}")
        
        # Generate sequential data (control)
        X_seq, y_seq = generate_sequential_data(n_steps, dim=16, rho=0.85)
        
        # Sequential baseline
        seq_pred = train_sequential(X_seq, y_seq, epochs=100)
        seq_mse = np.mean((seq_pred - y_seq) ** 2)
        
        # Causal prediction
        causal_pred = train_causal(X, causal, y, epochs=100)
        causal_mse = np.mean((causal_pred - y) ** 2)
        
        delta = (causal_mse - seq_mse) / seq_mse * 100
        winner = "CAUSAL" if causal_mse < seq_mse else "SEQUENTIAL"
        
        print(f"Sequential MSE: {seq_mse:.6f}, Causal MSE: {causal_mse:.6f}")
        print(f"Delta: {delta:+.1f}% ({winner})")
        
        results.append({
            'n_steps': n_steps,
            'sequential_mse': float(seq_mse),
            'causal_mse': float(causal_mse),
            'delta': float(delta),
            'winner': winner
        })
    
    print("\n" + "=" * 60)
    avg_delta = np.mean([r['delta'] for r in results])
    causal_wins = sum(1 for r in results if r['winner'] == 'CAUSAL')
    print(f"Overall: avg delta = {avg_delta:+.1f}%, causal wins {causal_wins}/{len(results)}")
    
    if avg_delta < -10 and causal_wins >= 3:
        status = "SUPPORTED"
    elif avg_delta > 5 or causal_wins <= 1:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    print(f"Status: {status}")
    
    return results, status, avg_delta

if __name__ == "__main__":
    results, status, avg_delta = run_experiment()
    
    output = {
        'hypothesis': 'H3.93',
        'title': 'Action Consequence Modeling',
        'status': status,
        'avg_delta': avg_delta,
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")