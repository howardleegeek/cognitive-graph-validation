#!/usr/bin/env python3
"""
H3.68: Attention Crossover at Intermediate Sequences (15-25 timesteps)

Tests attention vs concatenation at intermediate complexity where crossover is expected.
Tests: 15, 18, 20, 22, 25 step sequences.
"""

import numpy as np
import json
import time

np.random.seed(42)

def run_experiment():
    results = {}
    
    for steps in [15, 18, 20, 22, 25]:
        n_samples = 1000
        input_dim = 64
        
        x = np.random.randn(n_samples, input_dim).astype(np.float32)
        
        W = np.random.randn(input_dim, input_dim).astype(np.float32) * 0.1
        b = np.random.randn(input_dim).astype(np.float32) * 0.01
        
        x_centered = x - x.mean(axis=0, keepdims=True)
        x_scaled = x_centered / (np.std(x_centered, axis=0, keepdims=True) + 1e-8)
        
        for _ in range(steps):
            x_scaled = x_scaled @ W + b + np.random.randn(*x_scaled.shape).astype(np.float32) * 0.05
        
        y_true = x_scaled @ np.random.randn(input_dim, 1).astype(np.float32) * 0.5 + np.random.randn(n_samples, 1).astype(np.float32) * 0.1
        
        W_concat = np.random.randn(input_dim * 2, input_dim).astype(np.float32) * 0.01
        b_concat = np.random.randn(input_dim).astype(np.float32) * 0.01
        
        x_aug = np.concatenate([x_scaled, x_scaled], axis=1)
        
        pred_concat = x_aug @ W_concat[:input_dim*2] @ W_concat[:input_dim] + b_concat
        
        W_attn = np.random.randn(input_dim, input_dim).astype(np.float32) * 0.1
        attn_scores = x_scaled @ W_attn @ x_scaled.T
        attn_weights = np.exp(attn_scores / np.sqrt(input_dim))
        attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-8)
        context = attn_weights @ x_scaled
        
        W_out = np.random.randn(input_dim, input_dim).astype(np.float32) * 0.1
        b_out = np.random.randn(input_dim).astype(np.float32) * 0.01
        pred_attn = context @ W_out + b_out
        
        mse_concat = np.mean((pred_concat - y_true) ** 2)
        mse_attn = np.mean((pred_attn - y_true) ** 2)
        
        delta = (mse_concat - mse_attn) / (mse_concat + 1e-8) * 100
        
        results[steps] = {
            "concat_mse": float(mse_concat),
            "attn_mse": float(mse_attn),
            "delta": round(delta, 1)
        }
        
        print(f"Steps={steps}: Concat={mse_concat:.4f}, Attn={mse_attn:.4f}, Delta={delta:+.1f}%")
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("H3.68: Attention Crossover at Intermediate Sequences (15-25)")
    print("=" * 60)
    
    start = time.time()
    results = run_experiment()
    elapsed = time.time() - start
    
    avg_delta = np.mean([r["delta"] for r in results.values()])
    winner = "ATTENTION" if avg_delta > 0 else "CONCATENATION"
    
    print("=" * 60)
    print(f"Average: {avg_delta:+.1f}% → {winner} wins")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 60)
    
    with open("results.json", "w") as f:
        json.dump({
            "experiment": "H3.68",
            "results": results,
            "average_delta": round(avg_delta, 1),
            "winner": winner,
            "elapsed": round(elapsed, 1)
        }, f, indent=2)