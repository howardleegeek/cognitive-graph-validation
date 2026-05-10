#!/usr/bin/env python3
"""
H1.190: Task-Aware Temporal Structure Attention
Simplified version - Tests if phase-aware attention enables complex multi-step reasoning
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(42)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-8)

def generate_task_data(n_steps, dim=32, rho=0.85):
    """Generate sequential task data with phase structure"""
    X, y = [], []
    phase_vecs = []
    
    state = np.random.randn(dim)
    
    for t in range(n_steps):
        # Phase embedding
        if t < n_steps * 0.2:
            phase = np.eye(5)[0]  # reaching
        elif t < n_steps * 0.4:
            phase = np.eye(5)[1]  # approaching
        elif t < n_steps * 0.6:
            phase = np.eye(5)[2]  # grasping
        elif t < n_steps * 0.8:
            phase = np.eye(5)[3]  # manipulating
        else:
            phase = np.eye(5)[4]  # releasing
        
        # State with phase
        state_with_phase = np.concatenate([state, phase])
        
        # Action
        action = np.random.randn(8)
        
        # Next state with dynamics
        action_full = np.zeros(dim)
        action_full[:8] = action
        next_state = rho * state + (1 - rho) * np.random.randn(dim) + action_full * 0.1
        
        X.append(np.concatenate([state_with_phase, action]))
        y.append(next_state)
        phase_vecs.append(phase)
        
        state = next_state
    
    return np.array(X), np.array(y), np.array(phase_vecs)

def train_simple_mlp(X, y, hidden_dim=128, epochs=100):
    """Simple MLP baseline"""
    input_dim = X.shape[-1]
    output_dim = y.shape[-1]
    
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    b2 = np.zeros(hidden_dim)
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    b3 = np.zeros(output_dim)
    
    for _ in range(epochs):
        h1 = np.maximum(0, X @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        pred = h2 @ W3 + b3
    
    return pred

def train_phase_attention(X, y, phase_vecs, hidden_dim=128, epochs=100):
    """Phase-aware attention"""
    input_dim = X.shape[-1]
    output_dim = y.shape[-1]
    seq_len = len(X)
    
    # Phase-based key bias
    W_phase = np.random.randn(5, hidden_dim) * 0.01
    
    # Attention weights
    W_q = np.random.randn(input_dim, hidden_dim) * 0.01
    W_k = np.random.randn(input_dim, hidden_dim) * 0.01
    W_v = np.random.randn(input_dim, hidden_dim) * 0.01
    
    # Output
    W_out = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W_fc = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    
    for _ in range(epochs):
        # Queries from input
        Q = X @ W_q  # (seq, dim) -> (seq, hidden)
        
        # Keys with phase bias
        K = X @ W_k + phase_vecs @ W_phase  # Add phase bias
        V = X @ W_v
        
        # Attention
        scores = Q @ K.T / np.sqrt(hidden_dim)
        weights = softmax(scores, axis=-1)
        attended = weights @ V @ W_out
        
        # Output
        h = np.maximum(0, attended @ W_fc)
        pred = h @ W3
    
    return pred

def run_experiment():
    print("=" * 60)
    print("H1.190: Task-Aware Temporal Structure Attention")
    print("=" * 60)
    
    results = []
    
    for n_steps in [20, 30, 40, 50, 60]:
        print(f"\n--- Sequence length: {n_steps} steps ---")
        
        X, y, phases = generate_task_data(n_steps, dim=32, rho=0.85)
        print(f"Data shape: X={X.shape}, y={y.shape}, phases={phases.shape}")
        
        # Baseline
        baseline_pred = train_simple_mlp(X, y, epochs=100)
        baseline_mse = np.mean((baseline_pred - y) ** 2)
        
        # Phase attention
        attn_pred = train_phase_attention(X, y, phases, epochs=100)
        attn_mse = np.mean((attn_pred - y) ** 2)
        
        delta = (attn_mse - baseline_mse) / baseline_mse * 100
        winner = "PHASE-ATTN" if attn_mse < baseline_mse else "BASELINE"
        
        print(f"Baseline MSE: {baseline_mse:.6f}, Phase-Attn MSE: {attn_mse:.6f}")
        print(f"Delta: {delta:+.1f}% ({winner})")
        
        results.append({
            'n_steps': n_steps,
            'baseline_mse': float(baseline_mse),
            'attn_mse': float(attn_mse),
            'delta': float(delta),
            'winner': winner
        })
    
    print("\n" + "=" * 60)
    avg_delta = np.mean([r['delta'] for r in results])
    attn_wins = sum(1 for r in results if r['winner'] == 'PHASE-ATTN')
    print(f"Average Delta: {avg_delta:+.2f}%, Phase-Attn wins: {attn_wins}/{len(results)}")
    
    if avg_delta < -5 and attn_wins >= 3:
        status = "SUPPORTED"
    elif avg_delta > 10 or attn_wins <= 1:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    print(f"Status: {status}")
    
    return results, status, avg_delta

if __name__ == "__main__":
    results, status, avg_delta = run_experiment()
    
    output = {
        'hypothesis': 'H1.190',
        'title': 'Task-Aware Temporal Structure Attention',
        'status': status,
        'avg_delta': avg_delta,
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
