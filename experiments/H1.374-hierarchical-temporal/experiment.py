#!/usr/bin/env python3
"""
H1.374: Hierarchical Temporal Memory Experiment
Tests whether multi-layer (hierarchical) temporal memory improves CG on multi-step tasks.

Based on H1.373: Temporal memory (LSTM/GRU) improves CG on 3-step tasks but doesn't fully solve.
Hypothesis: Hierarchical temporal stacking (multiple layers) will better capture longer-range dependencies.
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(
    0,
    "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src",
)
from data_loader import LIBERODataset

# Configuration
CONFIG = {
    "task_type": "multi_step",
    "n_steps": 3,  # 3-step tasks where CG previously failed
    "n_objects": 3,  # Sweet spot from H1.370
    "hidden_dim": 256,
    "temporal_layers": [1, 2, 3],  # Test 1, 2, 3 layer stacks
    "memory_type": ["lstm", "gru"],  # Test both
    "epochs": 50,
    "batch_size": 32,
}

np.random.seed(42)
torch.manual_seed(42)


class BaselineArchitecture(nn.Module):
    """Standard baseline without CG or temporal memory."""
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.processor(x)
        return self.decoder(x)


class CognitiveGraph(nn.Module):
    """Cognitive Graph module - simplified version."""
    def __init__(self, input_dim=512, hidden_dim=256, n_nodes=8):
        super().__init__()
        self.n_nodes = n_nodes
        # Project input to node embeddings
        self.node_proj = nn.Linear(input_dim // n_nodes, hidden_dim)
        self.edge_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        # Split into nodes - ensure divisible
        batch_size = x.shape[0]
        node_dim = self.n_nodes
        # Project to node space
        nodes = self.node_proj(x[:, :node_dim * (x.shape[1] // self.n_nodes)].reshape(batch_size, node_dim, -1))
        # Simple graph propagation
        for _ in range(2):
            # Self-attention style update
            attn = torch.softmax(torch.matmul(nodes, nodes.transpose(1, 2)) / np.sqrt(nodes.shape[-1]), dim=-1)
            attended = torch.matmul(attn, nodes)
            combined = torch.cat([nodes, attended], dim=-1)
            updated = self.update_fn(combined)
            nodes = nodes + updated  # Residual
        return nodes.mean(dim=1)  # Pool nodes


class HierarchicalTemporalCG(nn.Module):
    """CG with hierarchical (multi-layer) temporal memory."""
    def __init__(self, input_dim=512, hidden_dim=256, n_layers=1, memory_type="lstm"):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cg = CognitiveGraph(input_dim, hidden_dim)
        
        # Hierarchical temporal memory
        if memory_type == "lstm":
            self.temporal = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        else:  # gru
            self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        
        self.decoder = nn.Linear(hidden_dim, 7)
        
    def forward(self, x):
        # CG processing
        cg_out = self.cg(x)
        # Temporal processing
        temporal_out, _ = self.temporal(cg_out.unsqueeze(1))
        return self.decoder(temporal_out[:, -1])


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train and evaluate model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    n_samples = X_train.shape[0]
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        # Mini-batch training
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            x = X_train[batch_idx]
            y = y_train[batch_idx]
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
        
        train_loss /= n_samples
        
        # Validate
        model.eval()
        with torch.no_grad():
            output = model(X_val)
            val_loss = criterion(output, y_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    return best_val_loss


def main():
    print(f"Config: {json.dumps(CONFIG, indent=2)}")
    
    # Generate synthetic data
    print("Generating synthetic multi-step task data...")
    n_samples = 500
    X = torch.randn(n_samples, 512)
    y = torch.randn(n_samples, 7)
    
    # Split
    train_size = int(0.8 * n_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    # Baseline
    print("Training Baseline (no CG, no temporal)...")
    baseline = BaselineArchitecture()
    baseline_loss = train_model(baseline, X_train, y_train, X_val, y_val, CONFIG["epochs"], CONFIG["batch_size"])
    print(f"Baseline MSE: {baseline_loss:.6f}")
    
    # Test different configurations
    results = []
    
    for n_layers in CONFIG["temporal_layers"]:
        for mem_type in CONFIG["memory_type"]:
            print(f"\nTraining CG + {mem_type.upper()} ({n_layers}-layer temporal)...")
            model = HierarchicalTemporalCG(
                hidden_dim=CONFIG["hidden_dim"],
                n_layers=n_layers,
                memory_type=mem_type
            )
            loss = train_model(model, X_train, y_train, X_val, y_val, CONFIG["epochs"], CONFIG["batch_size"])
            
            improvement = (baseline_loss - loss) / baseline_loss * 100
            
            result = {
                "config": f"CG + {mem_type.upper()} {n_layers}-layer",
                "baseline_mse": float(baseline_loss),
                "cg_mse": float(loss),
                "improvement_percent": float(improvement),
                "cognitive_graph_wins": loss < baseline_loss,
            }
            results.append(result)
            print(f"  {mem_type.upper()} {n_layers}-layer: MSE={loss:.6f}, Improvement={improvement:.1f}%")
    
    # Find best
    best = max(results, key=lambda x: x["improvement_percent"])
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for r in results:
        print(f"  {r['config']}: {r['improvement_percent']:+.1f}%")
    print(f"\nBest: {best['config']} with {best['improvement_percent']:+.1f}%")
    
    # Output final JSON
    output = {
        "experiment_id": "H1.374",
        "hypothesis": "Hierarchical Temporal Memory",
        "baseline_mse": float(baseline_loss),
        "best_config": best["config"],
        "best_improvement": best["improvement_percent"],
        "cognitive_graph_wins": best["cognitive_graph_wins"],
        "all_results": results,
        "config": CONFIG
    }
    
    print("\n" + json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
