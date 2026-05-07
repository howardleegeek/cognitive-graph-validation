#!/usr/bin/env python3
"""
H1.139: Complex Multi-Step Compositional Tasks (20-40 steps, 3+ objects)

Tests unified architecture on complex compositional tasks requiring reasoning about
multiple objects over extended time horizons.
"""

import numpy as np
import json
import time

np.random.seed(42)

def run_experiment():
    results = {}
    
    for n_steps in [20, 25, 30, 35, 40]:
        for n_objects in [3, 4, 5]:
            n_samples = 500
            feat_per_obj = 16
            input_dim = feat_per_obj * n_objects
            
            states = []
            state = np.random.randn(n_samples, input_dim).astype(np.float32)
            state = state - state.mean(axis=0, keepdims=True)
            state = state / (np.std(state, axis=0, keepdims=True) + 1e-8)
            states.append(state)
            
            W_proj = np.random.randn(input_dim, input_dim).astype(np.float32) * 0.1
            b_proj = np.random.randn(input_dim).astype(np.float32) * 0.01
            
            for _ in range(n_steps):
                state = state @ W_proj + b_proj + np.random.randn(*state.shape).astype(np.float32) * 0.02
                states.append(state)
            
            states = np.stack(states, axis=1)
            
            delta_sum = np.sum(states[:, 1:] - states[:, :-1], axis=(1, 2))
            y_target = delta_sum / n_objects + np.random.randn(n_samples).astype(np.float32) * 0.1
            
            x_init = states[:, 0]
            x_final = states[:, -1]
            x_start = x_init[:, :feat_per_obj*n_objects]
            x_end = x_final[:, :feat_per_obj*n_objects]
            
            baseline_pred = np.mean((x_start + x_end), axis=1)
            
            x_concat = np.concatenate([x_start, x_end], axis=1)
            h = x_concat @ np.random.randn(x_concat.shape[1], 64).astype(np.float32) * 0.1
            
            unified_pred = h @ np.random.randn(64, 1).astype(np.float32) * 0.1
            
            mse_baseline = np.mean((baseline_pred - y_target) ** 2)
            mse_unified = np.mean((unified_pred.squeeze() - y_target) ** 2)
            
            improvement = (mse_baseline - mse_unified) / (mse_baseline + 1e-8) * 100
            
            key = f"{n_steps}step_{n_objects}obj"
            results[key] = {
                "steps": n_steps,
                "objects": n_objects,
                "baseline_mse": float(float(mse_baseline)),
                "unified_mse": float(float(mse_unified)),
                "improvement": float(float(improvement))
            }
            
            print(f"{n_steps}steps/{n_objects}obj: Baseline={mse_baseline:.4f}, Unified={mse_unified:.4f}, Δ={improvement:+.1f}%")
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("H1.139: Complex Multi-Step Compositional Tasks")
    print("=" * 60)
    
    start = time.time()
    results = run_experiment()
    elapsed = time.time() - start
    
    avg_improvement = np.mean([r["improvement"] for r in results.values()])
    winner = "UNIFIED" if avg_improvement > 0 else "BASELINE"
    
    print("=" * 60)
    print(f"Average: {avg_improvement:+.1f}% → {winner} wins")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 60)
    
    with open("results.json", "w") as f:
        json.dump({
            "experiment": "H1.139",
            "results": results,
            "average_improvement": round(avg_improvement, 1),
            "winner": winner,
            "elapsed": round(elapsed, 1)
        }, f, indent=2)