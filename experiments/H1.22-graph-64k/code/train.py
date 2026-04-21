#!/usr/bin/env python3
"""H1.22: Graph + unified combined architecture."""

import numpy as np

np.random.seed(42)

def generate_data(n_samples, n_timesteps, n_objects, action_dim, seed):
    np.random.seed(seed)
    X_combined = []
    y_actions = []
    
    for _ in range(n_samples):
        for t in range(n_timesteps):
            obs = np.random.randn(n_objects, 7).astype(np.float32) * 0.1
            language = np.random.randn(50).astype(np.float32) * 0.1
            
            physical = obs[:, :4].flatten()  # 12 dims
            semantics = np.concatenate([obs[:, 4:].flatten(), language[:20]])  # 9 + 20 = 29 dims
            x = np.concatenate([physical, semantics])  # 41 dims total
            
            action = np.sum(obs[:n_objects, :action_dim], axis=0) + np.random.randn(action_dim) * 0.01
            
            X_combined.append(x)
            y_actions.append(action)
    
    return np.array(X_combined), np.array(y_actions)

def relu(x):
    return np.maximum(0, x)

class BaselineModel:
    def __init__(self, input_dim=41, hidden=128):
        scale = 0.01
        self.W1 = np.random.randn(input_dim, hidden).astype(np.float32) * scale
        self.W_out = np.random.randn(hidden, 4).astype(np.float32) * scale
    
    def forward(self, x):
        h = relu(x @ self.W1)
        return h @ self.W_out

class UnifiedModel:
    def __init__(self, total_dim, use_graph=False):
        scale = 0.01
        self.hidden = min(total_dim, 2048)
        
        # Input: 12 physical + 29 semantic = 41
        self.W_phys = np.random.randn(12, self.hidden).astype(np.float32) * scale
        self.W_sem = np.random.randn(29, self.hidden).astype(np.float32) * scale
        self.W_fuse = np.random.randn(self.hidden * 2, self.hidden).astype(np.float32) * scale
        self.W_out = np.random.randn(self.hidden, 4).astype(np.float32) * scale
        
        self.use_graph = use_graph
        if use_graph:
            self.W_graph = np.random.randn(self.hidden, self.hidden).astype(np.float32) * 0.001
    
    def forward(self, x):
        p = x[:, :12]
        s = x[:, 12:41]
        
        p_enc = relu(p @ self.W_phys)
        s_enc = relu(s @ self.W_sem)
        combined = np.concatenate([p_enc, s_enc], axis=-1)
        fused = relu(combined @ self.W_fuse)
        
        if self.use_graph:
            graph_out = relu(fused @ self.W_graph + fused)
            fused = fused + 0.1 * graph_out
        
        return fused @ self.W_out

def compute_mse(model, X, y, n_iters=30):
    losses = []
    for _ in range(n_iters):
        idx = np.random.randint(0, len(X), size=min(50, len(X)))
        preds = model.forward(X[idx])
        losses.append(np.mean((preds - y[idx]) ** 2))
    return np.mean(losses)

def main():
    print("=" * 60)
    print("H1.22: Graph + Unified Combined")
    print("=" * 60)
    
    results = {'baseline': [], 'unified': [], 'graph': []}
    
    for seed in range(42, 47):
        X, y = generate_data(300, 8, 3, 4, seed)
        
        baseline = BaselineModel()
        unified = UnifiedModel(32768, use_graph=False)
        graph = UnifiedModel(32768, use_graph=True)
        
        baseline_mse = compute_mse(baseline, X, y)
        unified_mse = compute_mse(unified, X, y)
        graph_mse = compute_mse(graph, X, y)
        
        results['baseline'].append(baseline_mse)
        results['unified'].append(unified_mse)
        results['graph'].append(graph_mse)
        
        print(f"Seed {seed}: Baseline={baseline_mse:.4f}, Unified={unified_mse:.4f}, Graph={graph_mse:.4f}")
    
    avg_b = np.mean(results['baseline'])
    avg_u = np.mean(results['unified'])
    avg_g = np.mean(results['graph'])
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline: {avg_b:.4f}")
    print(f"  Unified: {avg_u:.4f}")
    print(f"  Graph+Unified: {avg_g:.4f}")
    
    delta_g = (avg_g - avg_b) / avg_b * 100
    delta_u = (avg_u - avg_b) / avg_b * 100
    
    print(f"\n  Graph+Unified vs Baseline: {delta_g:+.1f}%")
    print(f"  Unified vs Baseline: {delta_u:+.1f}%")
    
    if avg_g < avg_u:
        print("\n  CONCLUSION: Graph + Unified COMBINED outperforms Unified alone!")
    else:
        print("\n  CONCLUSION: Unified alone is sufficient.")

if __name__ == "__main__":
    main()