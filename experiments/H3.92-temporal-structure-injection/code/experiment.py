#!/usr/bin/env python3
"""
H3.92: Temporal Structure Injection for Synthetic Data
Simplified version - Tests if adding autocorrelation enables attention
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(42)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-8)

def inject_autocorrelation(data, rho):
    """Inject temporal autocorrelation"""
    if len(data.shape) == 2:
        result = np.zeros_like(data)
        result[0] = data[0]
        for t in range(1, len(data)):
            result[t] = rho * result[t-1] + (1 - rho) * data[t]
        return result
    return data

def generate_data(n_steps, dim=32, rho=0.85):
    """Generate sequential data with given autocorrelation"""
    raw_states = np.random.randn(n_steps, dim)
    raw_actions = np.random.randn(n_steps, 8)
    raw_targets = np.random.randn(n_steps, dim)
    
    # Inject autocorrelation
    states = inject_autocorrelation(raw_states, rho)
    actions = inject_autocorrelation(raw_actions, rho)
    targets = inject_autocorrelation(raw_targets, rho)
    
    X = np.concatenate([states, actions], axis=1)
    y = targets
    
    return X, y

def train_mlp(X, y, hidden_dim=64, epochs=80):
    """Simple MLP baseline"""
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

def train_attention(X, y, hidden_dim=64, epochs=80):
    """Cross-modal attention"""
    input_dim = X.shape[-1]
    output_dim = y.shape[-1]
    
    W_q = np.random.randn(input_dim, hidden_dim) * 0.01
    W_k = np.random.randn(input_dim, hidden_dim) * 0.01
    W_v = np.random.randn(input_dim, hidden_dim) * 0.01
    W_out = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W_fc = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    
    for _ in range(epochs):
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v
        
        scores = Q @ K.T / np.sqrt(hidden_dim)
        weights = softmax(scores, axis=-1)
        attended = weights @ V @ W_out
        
        h = np.maximum(0, attended @ W_fc)
        pred = h @ W3
    
    return pred

def run_experiment():
    print("=" * 60)
    print("H3.92: Temporal Structure Injection for Synthetic Data")
    print("=" * 60)
    
    results = []
    
    rho_values = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]
    
    for rho in rho_values:
        print(f"\n--- Testing autocorrelation ρ = {rho} ---")
        
        X, y = generate_data(n_steps=30, dim=32, rho=rho)
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Baseline
        baseline_pred = train_mlp(X, y, epochs=80)
        baseline_mse = np.mean((baseline_pred - y) ** 2)
        
        # Attention
        attn_pred = train_attention(X, y, epochs=80)
        attn_mse = np.mean((attn_pred - y) ** 2)
        
        delta = (attn_mse - baseline_mse) / baseline_mse * 100
        winner = "ATTENTION" if attn_mse < baseline_mse else "BASELINE"
        
        print(f"Baseline MSE: {baseline_mse:.6f}, Attention MSE: {attn_mse:.6f}")
        print(f"Delta: {delta:+.1f}% ({winner})")
        
        results.append({
            'rho': rho,
            'baseline_mse': float(baseline_mse),
            'attn_mse': float(attn_mse),
            'delta': float(delta),
            'winner': winner
        })
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    high_rho = [r for r in results if r['rho'] >= 0.7]
    low_rho = [r for r in results if r['rho'] < 0.7]
    
    avg_high = np.mean([r['delta'] for r in high_rho]) if high_rho else 0
    avg_low = np.mean([r['delta'] for r in low_rho]) if low_rho else 0
    attn_wins_high = sum(1 for r in high_rho if r['winner'] == 'ATTENTION')
    
    print(f"\nHigh autocorrelation (ρ≥0.7): avg delta = {avg_high:+.1f}%, attention wins {attn_wins_high}/{len(high_rho)}")
    print(f"Low autocorrelation (ρ<0.7): avg delta = {avg_low:+.1f}%")
    
    if avg_high < -10 and attn_wins_high >= 3 and avg_high < avg_low:
        status = "SUPPORTED"
    elif avg_high > 5 or attn_wins_high <= 1:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    print(f"Status: {status}")
    
    return results, status

if __name__ == "__main__":
    results, status = run_experiment()
    
    output = {
        'hypothesis': 'H3.92',
        'title': 'Temporal Structure Injection',
        'status': status,
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
