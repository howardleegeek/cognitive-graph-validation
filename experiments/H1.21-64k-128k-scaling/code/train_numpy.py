"""H1.21: Test 64k-128k dimension scaling with optimal alpha - NumPy version."""

import numpy as np
import random
from typing import List, Tuple, Dict

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

class NumpyUnifiedModel:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 total_dim: int, physical_pct: float = 0.22, alpha: float = 0.3):
        self.total_dim = total_dim
        self.physical_dim = int(total_dim * physical_pct)
        self.semantic_dim = total_dim - self.physical_dim
        self.alpha = alpha
        self.input_dim = input_dim
        
        scale = np.sqrt(2.0 / input_dim)
        self.W_physical = np.random.randn(input_dim, total_dim).astype(np.float32) * scale
        self.b_physical = np.zeros(total_dim, dtype=np.float32)
        
        scale = np.sqrt(2.0 / input_dim)
        self.W_semantic = np.random.randn(input_dim, total_dim).astype(np.float32) * scale
        self.b_semantic = np.zeros(total_dim, dtype=np.float32)
        
        scale = np.sqrt(2.0 / (total_dim * 2))
        self.W_fusion = np.random.randn(total_dim * 2, total_dim).astype(np.float32) * scale
        self.b_fusion = np.zeros(total_dim, dtype=np.float32)
        
        scale = np.sqrt(2.0 / total_dim)
        self.W_decoder = np.random.randn(total_dim, output_dim).astype(np.float32) * scale
        self.b_decoder = np.zeros(output_dim, dtype=np.float32)
        
        self.masks = {}
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def dropout(self, x, alpha):
        mask = (np.random.rand(*x.shape) > alpha).astype(np.float32)
        return x * mask

    def forward(self, X):
        X_phys = X[:, :self.input_dim // 2] if X.shape[1] > self.input_dim // 2 else X
        X_sem = X[:, self.input_dim // 2:] if X.shape[1] > self.input_dim // 2 else X[:, :min(self.input_dim // 2, X.shape[1])]
        
        h_phys = self.relu(X_phys @ self.W_physical + self.b_physical)
        h_sem = self.relu(X_sem @ self.W_semantic + self.b_semantic)
        
        combined = np.concatenate([h_phys, h_sem], axis=-1)
        fused = self.relu(combined @ self.W_fusion + self.b_fusion)
        fused = self.dropout(fused, self.alpha)
        
        output = fused @ self.W_decoder + self.b_decoder
        return output

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.01):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            pred = self.forward(X)
            error = pred - y
            mse = np.mean(error ** 2)
            
            if epoch % 50 == 0:
                pass

def generate_data(n_samples: int, n_timesteps: int, n_objects: int, 
                  action_dim: int, input_dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    
    X_list = []
    y_list = []
    
    for _ in range(n_samples):
        for _ in range(n_timesteps):
            features = np.random.randn(input_dim).astype(np.float32)
            action = np.sum(features[:n_objects * action_dim]) + np.random.randn(action_dim) * 0.01
            
            X_list.append(features)
            y_list.append(action)
    
    return np.array(X_list), np.array(y_list)

def test_config(dim: int, alpha: float, input_dim: int = 64, n_runs: int = 3) -> float:
    results = []
    
    for seed in range(42, 42 + n_runs):
        set_seed(seed)
        
        X, y = generate_data(200, 8, 3, 4, input_dim, seed)
        
        model = NumpyUnifiedModel(input_dim, dim, 4, dim, alpha=alpha)
        model.fit(X, y, epochs=150, lr=0.01)
        
        pred = model.forward(X)
        mse = np.mean((pred - y) ** 2)
        results.append(mse)
    
    return np.mean(results)

def main():
    print("=" * 60)
    print("H1.21: 64k-128k Dimension Scaling")
    print("=" * 60)
    
    configs = [
        (4096, 0.1),
        (16384, 0.1),
        (32768, 0.3),
        (65536, 0.3),
        (131072, 0.3),
    ]
    
    results = {}
    input_dim = 64
    
    for dim, alpha in configs:
        print(f"\nTesting {dim} dims with α={alpha}...", flush=True)
        mse = test_config(dim, alpha, input_dim=input_dim, n_runs=3)
        results[(dim, alpha)] = mse
        print(f"  MSE: {mse:.4f}")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for (dim, alpha), loss in sorted_results:
        print(f"  {dim:>6} dims, α={alpha}: MSE={loss:.4f}")
    
    best = sorted_results[0]
    print(f"\nBest: {best[0][0]} dims, α={best[0][1]} → MSE={best[1]:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()