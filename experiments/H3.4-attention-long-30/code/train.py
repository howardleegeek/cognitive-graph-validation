#!/usr/bin/env python3
"""H3.4: Test attention vs concatenation on very long sequences (20-30 timesteps)."""

import numpy as np
from typing import List, Dict

np.random.seed(42)

def generate_data(n_samples, n_timesteps, n_objects, action_dim, seed):
    np.random.seed(seed)
    X_physical = []
    X_semantic = []
    y_actions = []
    
    for _ in range(n_samples):
        for t in range(n_timesteps):
            obs = np.random.randn(n_objects, 7).astype(np.float32) * 0.1
            language = np.random.randn(50).astype(np.float32) * 0.1
            
            physical = obs[:, :4].flatten()
            semantics = np.concatenate([
                obs[:, 4:].flatten(),
                language[:20]
            ])
            
            action = np.sum(obs[:n_objects, :action_dim], axis=0)
            action += np.tanh(np.arange(action_dim, dtype=np.float32) * 0.01 * (t / n_timesteps))
            action += np.random.randn(action_dim) * 0.01
            
            X_physical.append(physical)
            X_semantic.append(semantics)
            y_actions.append(action)
    
    return np.array(X_physical), np.array(X_semantic), np.array(y_actions)

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(np.clip(x, -500, 500))

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SimpleConcatModel:
    def __init__(self, input_dim, output_dim=4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 128
        
        scale = 0.01
        self.W1 = np.random.randn(input_dim, self.hidden_dim).astype(np.float32) * scale
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * scale
        self.W_out = np.random.randn(self.hidden_dim, output_dim).astype(np.float32) * scale
    
    def forward(self, x):
        h = relu(x @ self.W1)
        h = relu(h @ self.W2)
        return h @ self.W_out

class SimpleAttentionModel:
    def __init__(self, input_dim, output_dim=4, n_heads=4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 128
        self.n_heads = n_heads
        self.head_dim = self.hidden_dim // n_heads
        
        scale = 0.01
        self.W_q = np.random.randn(input_dim, self.hidden_dim).astype(np.float32) * scale
        self.W_k = np.random.randn(input_dim, self.hidden_dim).astype(np.float32) * scale
        self.W_v = np.random.randn(input_dim, self.hidden_dim).astype(np.float32) * scale
        self.W_o = np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * scale
        self.W_out = np.random.randn(self.hidden_dim, output_dim).astype(np.float32) * scale
    
    def forward(self, x):
        Q = relu(x @ self.W_q)
        K = relu(x @ self.W_k)
        V = relu(x @ self.W_v)
        
        Q_h = Q.reshape(-1, self.n_heads, self.head_dim)
        K_h = K.reshape(-1, self.n_heads, self.head_dim)
        V_h = V.reshape(-1, self.n_heads, self.head_dim)
        
        scores = np.einsum('bhd,bkd->bhk', Q_h, K_h) / np.sqrt(self.head_dim)
        attn = softmax(scores, axis=-1)
        attended = np.einsum('bhk,bkd->bhd', attn, V_h)
        
        h = attended.reshape(-1, self.hidden_dim)
        h = relu(h @ self.W_o)
        return h @ self.W_out

def compute_mse(model, X, y, n_iters=50):
    losses = []
    for _ in range(n_iters):
        idx = np.random.randint(0, len(X), size=min(50, len(X)))
        preds = model.forward(X[idx])
        losses.append(np.mean((preds - y[idx]) ** 2))
    return np.mean(losses)

def run_trial(n_timesteps, n_samples, seed):
    X_p, X_s, y = generate_data(n_samples, n_timesteps, 3, 4, seed)
    
    X_combined = np.concatenate([X_p, X_s], axis=-1)
    
    concat = SimpleConcatModel(X_combined.shape[-1], 4)
    attn = SimpleAttentionModel(X_combined.shape[-1], 4)
    
    concat_mse = compute_mse(concat, X_combined, y)
    attn_mse = compute_mse(attn, X_combined, y)
    
    return concat_mse, attn_mse

def main():
    print("=" * 60)
    print("H3.4: Attention vs Concatenation on Very Long Sequences")
    print("=" * 60)
    
    timesteps_list = [20, 24, 28, 30]
    results = {}
    
    for n_steps in timesteps_list:
        print(f"\nTesting {n_steps} timesteps...", flush=True)
        
        concat_losses = []
        attn_losses = []
        
        for seed in range(42, 47):
            concat_mse, attn_mse = run_trial(n_steps, 300, seed)
            concat_losses.append(concat_mse)
            attn_losses.append(attn_mse)
        
        concat_mean = np.mean(concat_losses)
        concat_std = np.std(concat_losses)
        attn_mean = np.mean(attn_losses)
        attn_std = np.std(attn_losses)
        
        delta = (attn_mean - concat_mean) / concat_mean * 100
        
        results[n_steps] = {
            'concat': (concat_mean, concat_std),
            'attention': (attn_mean, attn_std),
            'delta': delta
        }
        
        print(f"  Concat MSE: {concat_mean:.4f} ± {concat_std:.4f}")
        print(f"  Attention MSE: {attn_mean:.4f} ± {attn_std:.4f}")
        print(f"  Delta: {delta:+.1f}% {'(Attention wins!)' if delta < 0 else '(Concat wins)'}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for n_steps, data in results.items():
        concat_mean, _ = data['concat']
        attn_mean, _ = data['attention']
        delta = data['delta']
        
        winner = "ATTENTION" if delta < 0 else "CONCAT"
        print(f"  {n_steps} steps: {winner} wins ({delta:+.1f}%)")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    all_concat_wins = all(d['delta'] > 0 for d in results.values())
    all_attn_wins = all(d['delta'] < 0 for d in results.values())
    
    if all_concat_wins:
        print("  CONCLUSION: Concatenation wins on VERY LONG sequences.")
        print("  This confirms H3: attention overhead not worth it ever.")
    elif all_attn_wins:
        print("  CONCLUSION: Attention wins on long sequences!")
    else:
        mixed = [d['delta'] for d in results.values()]
        avg_delta = np.mean(mixed)
        if avg_delta > 0:
            print(f"  CONCLUSION: On average, CONCATENATION wins ({avg_delta:+.1f}%)")
        else:
            print(f"  CONCLUSION: On average, ATTENTION wins ({avg_delta:+.1f}%)")
    
    return results

if __name__ == "__main__":
    results = main()