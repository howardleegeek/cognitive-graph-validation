#!/usr/bin/env python3
"""
H1.396: Architecture Tuning Investigation
Test different architecture configurations to improve CG performance.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[H1.396] Using device: {device}")


class BaselineModel(nn.Module):
    """Baseline concatenation model."""
    
    def __init__(self, input_dim=64, hidden_dim=512, output_dim=32):
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


class CognitiveGraphModel(nn.Module):
    """Cognitive Graph model with configurable architecture."""
    
    def __init__(self, input_dim=64, hidden_dim=512, output_dim=32, n_heads=4, n_nodes=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.node_dim = hidden_dim // n_nodes
        
        # Node embeddings
        self.node_embed = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention for graph reasoning
        if n_heads > 0:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.node_dim,
                num_heads=min(n_heads, self.node_dim // 16) if self.node_dim >= 16 else 1,
                batch_first=True
            )
        else:
            self.attention = None
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Project to node space
        nodes = self.node_embed(x)  # [B, hidden_dim]
        nodes = nodes.view(-1, self.n_nodes, self.node_dim)  # [B, n_nodes, node_dim]
        
        # Apply attention if available
        if self.attention is not None:
            attn_out, _ = self.attention(nodes, nodes, nodes)
            nodes = nodes + attn_out  # Residual
        
        # Aggregate and project
        nodes = nodes.view(-1, self.hidden_dim)
        return self.output_proj(nodes)


def generate_synthetic_data(n_samples=200, complexity=100, seed=42):
    """Generate synthetic data with specified complexity."""
    np.random.seed(seed)
    
    # Input: physical + semantic features
    X = np.random.randn(n_samples, 64).astype(np.float32)
    
    # Target: complex function of inputs
    # Higher complexity = more non-linear interactions
    Y = np.zeros((n_samples, 32), dtype=np.float32)
    
    for i in range(32):
        # Base linear term
        Y[:, i] = X[:, i] * 0.5
        
        # Add non-linear interactions based on complexity
        for j in range(min(complexity // 10, 32)):
            Y[:, i] += np.sin(X[:, j] * X[:, (i + j) % 32]) * 0.1
        
        # Add cross-terms
        if complexity > 50:
            Y[:, i] += X[:, (i + 16) % 32] * X[:, (i + 24) % 32] * 0.05
        
        if complexity > 100:
            Y[:, i] += np.tanh(X[:, (i + 8) % 32] + X[:, (i + 12) % 32]) * 0.1
    
    # Add noise
    Y += np.random.randn(n_samples, 32).astype(np.float32) * 0.1
    
    return X, Y


def train_model(model, X_train, Y_train, X_test, Y_test, epochs=20, lr=1e-3):
    """Train model and return best test MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    Y_test_t = torch.FloatTensor(Y_test).to(device)
    
    best_mse = float('inf')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_mse = criterion(test_pred, Y_test_t).item()
            best_mse = min(best_mse, test_mse)
    
    return best_mse


def run_config(config_name, hidden_dim, n_heads, epochs, lr, complexity_levels):
    """Run experiment with specific configuration."""
    results = {}
    
    for complexity in complexity_levels:
        # Generate data
        X, Y = generate_synthetic_data(n_samples=200, complexity=complexity, seed=SEED)
        
        # Split
        n_train = 160
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        
        # Train baseline
        baseline = BaselineModel(hidden_dim=hidden_dim).to(device)
        baseline_mse = train_model(baseline, X_train, Y_train, X_test, Y_test, epochs=epochs, lr=lr)
        
        # Train CG model
        cg = CognitiveGraphModel(hidden_dim=hidden_dim, n_heads=n_heads).to(device)
        cg_mse = train_model(cg, X_train, Y_train, X_test, Y_test, epochs=epochs, lr=lr)
        
        # Calculate improvement
        improvement = (baseline_mse - cg_mse) / baseline_mse * 100
        
        results[complexity] = {
            'baseline_mse': baseline_mse,
            'cg_mse': cg_mse,
            'improvement': improvement,
            'cg_wins': cg_mse < baseline_mse
        }
        
        print(f"  Complexity {complexity}: Baseline={baseline_mse:.4f}, CG={cg_mse:.4f}, Improvement={improvement:+.1f}%")
    
    return results


def main():
    print(f"\n{'='*60}")
    print("H1.396: Architecture Tuning Investigation")
    print(f"{'='*60}\n")
    
    # Configurations to test
    configs = [
        {'name': 'A', 'hidden_dim': 256, 'n_heads': 2, 'epochs': 20, 'lr': 1e-3, 'desc': 'Smaller model'},
        {'name': 'B', 'hidden_dim': 512, 'n_heads': 1, 'epochs': 20, 'lr': 1e-3, 'desc': 'Fewer attention heads'},
        {'name': 'C', 'hidden_dim': 512, 'n_heads': 4, 'epochs': 40, 'lr': 1e-3, 'desc': 'More training'},
        {'name': 'D', 'hidden_dim': 512, 'n_heads': 4, 'epochs': 20, 'lr': 1e-4, 'desc': 'Lower learning rate'},
        {'name': 'E', 'hidden_dim': 128, 'n_heads': 1, 'epochs': 20, 'lr': 1e-3, 'desc': 'Minimal model'},
    ]
    
    complexity_levels = [100, 300]  # Focus on key complexity levels
    
    all_results = {}
    
    for config in configs:
        print(f"\nConfig {config['name']}: {config['desc']}")
        print(f"  hidden_dim={config['hidden_dim']}, n_heads={config['n_heads']}, epochs={config['epochs']}, lr={config['lr']}")
        
        results = run_config(
            config['name'],
            config['hidden_dim'],
            config['n_heads'],
            config['epochs'],
            config['lr'],
            complexity_levels
        )
        
        all_results[config['name']] = {
            'config': config,
            'results': results
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("\nComplexity=100:")
    for name, data in all_results.items():
        r = data['results'][100]
        print(f"  Config {name}: {r['improvement']:+.1f}% ({'CG wins' if r['cg_wins'] else 'Baseline wins'})")
    
    print("\nComplexity=300:")
    for name, data in all_results.items():
        r = data['results'][300]
        print(f"  Config {name}: {r['improvement']:+.1f}% ({'CG wins' if r['cg_wins'] else 'Baseline wins'})")
    
    # Find best config
    best_config = None
    best_avg_improvement = -float('inf')
    
    for name, data in all_results.items():
        avg_imp = np.mean([r['improvement'] for r in data['results'].values()])
        if avg_imp > best_avg_improvement:
            best_avg_improvement = avg_imp
            best_config = name
    
    print(f"\nBest configuration: Config {best_config} (avg improvement: {best_avg_improvement:+.1f}%)")
    
    # Save results
    output = {
        'experiment': 'H1.396',
        'timestamp': datetime.now().isoformat(),
        'seed': SEED,
        'device': str(device),
        'results': all_results,
        'best_config': best_config,
        'best_avg_improvement': best_avg_improvement
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\nResults saved to results.json")
    
    return output


if __name__ == "__main__":
    main()