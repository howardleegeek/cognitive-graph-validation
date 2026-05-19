#!/usr/bin/env python3
"""
H1.446 - Reproduce H1.444 to verify +2.6% improvement is reproducible
or was a statistical anomaly.

H1.444 config:
- Combined modifications: edge_aware + high_dim + residual
- Task: action_prediction
- n_trials: 5 (more than original 2 to reduce noise)
- epochs: 50
- batch_size: 64
- noise: 0.05
- n_samples: 500

This runs the exact same config with more trials to get better statistics.
"""

import sys
import os
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")
from data_loader import LIBERODataset


class GraphCG(nn.Module):
    """Graph Cognitive Graph - with combined modifications from H1.444"""
    
    def __init__(self, input_dim=9, hidden_dim=256, output_dim=2, n_heads=4, use_edge_aware=True, use_residual=True):
        super().__init__()
        self.use_edge_aware = use_edge_aware
        self.use_residual = use_residual
        self.input_dim = input_dim
        
        # High-dim projection (from H1.444)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Edge-aware attention (from H1.444)
        if use_edge_aware:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Residual connection (from H1.444)
        if use_residual:
            self.res_proj = nn.Linear(input_dim, hidden_dim)
        
        self.output_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape
        
        # High-dim projection
        h = self.input_proj(x)
        
        # Residual
        if self.use_residual:
            res = self.res_proj(x)
        
        # Self-attention
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out)
        
        # FFN
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)
        
        # Add residual if enabled
        if self.use_residual:
            h = h + res
        
        # Output
        return self.output_head(h)


class MLPBaseline(nn.Module):
    """Simple MLP baseline"""
    
    def __init__(self, input_dim=9, hidden_dim=256, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_task_data(n_samples=500, seq_len=10, noise=0.05):
    """Generate action prediction task data matching H1.444"""
    # Simulate object-centric manipulation data
    n_objects = 3  # Same as H1.444
    
    X, y = [], []
    for _ in range(n_samples):
        # Generate sequence of object states
        seq = []
        for t in range(seq_len):
            # Object positions (n_objects * 2 for x, y)
            obj_pos = np.random.randn(n_objects * 2) * 0.5
            # Gripper state
            gripper = np.random.rand(1) * 0.5
            # Action target (next step)
            action = np.random.randn(2) * 0.3
            
            state = np.concatenate([obj_pos, gripper, action])
            seq.append(state)
        
        seq = np.array(seq)
        # Add noise
        seq += np.random.randn(*seq.shape) * noise
        
        X.append(seq[:-1])  # Input: all but last
        y.append(seq[1:, -2:])  # Output: action (last 2 dims) for all but first
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(model, X, y, epochs=50, batch_size=64, lr=1e-3):
    """Train model on task"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = torch.tensor(X[batch_idx], dtype=torch.float32)
            y_batch = torch.tensor(y[batch_idx], dtype=torch.float32)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def evaluate_model(model, X, y):
    """Evaluate model MSE"""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        pred = model(X_t)
        mse = F.mse_loss(pred, y_t).item()
    return mse


def run_experiment(n_trials=5, n_samples=500, epochs=50, noise=0.05):
    """Run reproduction experiment with more trials for statistical significance"""
    
    results = {
        "mlp_mse": [],
        "graphcg_mse": [],
        "improvement_pct": []
    }
    
    for trial in range(n_trials):
        set_seed(42 + trial)  # Different seed per trial
        
        # Generate data
        X, y = generate_task_data(n_samples=n_samples, noise=noise)
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train MLP
        mlp = MLPBaseline(input_dim=9, hidden_dim=256, output_dim=2)
        train_model(mlp, X_train, y_train, epochs=epochs)
        mlp_mse = evaluate_model(mlp, X_test, y_test)
        
        # Train GraphCG with combined modifications (from H1.444)
        graphcg = GraphCG(
            input_dim=9, 
            hidden_dim=256, 
            output_dim=2, 
            n_heads=4,
            use_edge_aware=True,   # H1.444 modification
            use_residual=True      # H1.444 modification
        )
        train_model(graphcg, X_train, y_train, epochs=epochs)
        graphcg_mse = evaluate_model(graphcg, X_test, y_test)
        
        improvement = ((mlp_mse - graphcg_mse) / mlp_mse) * 100
        
        results["mlp_mse"].append(mlp_mse)
        results["graphcg_mse"].append(graphcg_mse)
        results["improvement_pct"].append(improvement)
        
        print(f"Trial {trial+1}/{n_trials}: MLP={mlp_mse:.4f}, GraphCG={graphcg_mse:.4f}, Improvement={improvement:+.2f}%")
    
    # Summary
    avg_mlp = np.mean(results["mlp_mse"])
    avg_graphcg = np.mean(results["graphcg_mse"])
    avg_improvement = np.mean(results["improvement_pct"])
    std_improvement = np.std(results["improvement_pct"])
    
    print(f"\n=== H1.446 REPRODUCTION RESULTS ===")
    print(f"Average MLP MSE: {avg_mlp:.4f} ± {np.std(results['mlp_mse']):.4f}")
    print(f"Average GraphCG MSE: {avg_graphcg:.4f} ± {np.std(results['graphcg_mse']):.4f}")
    print(f"Average Improvement: {avg_improvement:+.2f}% ± {std_improvement:.2f}%")
    print(f"Win Rate: {sum(1 for i in results['improvement_pct'] if i > 0)}/{n_trials}")
    
    # Determine conclusion
    if avg_improvement > 2.0 and std_improvement < 3.0:
        conclusion = "REPRODUCED - H1.444's +2.6% is reproducible"
    elif avg_improvement > 0:
        conclusion = "PARTIAL - Small positive effect but noisy"
    else:
        conclusion = "REFUTED - H1.444 result was statistical anomaly"
    
    print(f"Conclusion: {conclusion}")
    
    return {
        "results": results,
        "summary": {
            "avg_mlp_mse": avg_mlp,
            "avg_graphcg_mse": avg_graphcg,
            "avg_improvement_pct": avg_improvement,
            "std_improvement_pct": std_improvement,
            "win_rate": f"{sum(1 for i in results['improvement_pct'] if i > 0)}/{n_trials}",
            "conclusion": conclusion
        }
    }


if __name__ == "__main__":
    print("=" * 60)
    print("H1.446 - Reproducing H1.444 with more trials (5 vs 2)")
    print("=" * 60)
    
    results = run_experiment(n_trials=5, n_samples=500, epochs=50, noise=0.05)
    
    # Save results
    output_dir = Path("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/446-reproduce-h1444")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
