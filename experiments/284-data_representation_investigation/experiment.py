#!/usr/bin/env python3
"""
H1.470.1.1.45: Investigate Fundamental Data Representation Issue

Context: H1.470.1.1.44 showed 100% underfitting across ALL configurations.
- Larger hidden dimensions WORSEN performance (128 > 256 > 512)
- SiLU slightly better than GELU/ReLU
- Problem is NOT model capacity - it's fundamental to data/representation

Hypothesis: The synthetic data generation or representation is causing systematic
underfitting. Testing:
1. Different data distributions (uniform, normal, multimodal)
2. Different sequence correlation structures
3. Input vs output representation issues
4. Learning objective modifications
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class SimpleGRU(nn.Module):
    """Baseline GRU model."""
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.decoder(out[:, -1, :])


class CognitiveGraph(nn.Module):
    """Cognitive Graph with unified representation."""
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, num_layers=2):
        super().__init__()
        # Physical encoder (144 dims)
        self.physical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 144)
        )
        # Semantic encoder (368 dims)
        self.semantic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 368)
        )
        # Unified processor
        self.graph_processor = nn.GRU(512, hidden_dim, num_layers=num_layers, batch_first=True)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        unified = torch.cat([physical, semantic], dim=-1)
        if len(unified.shape) == 2:
            unified = unified.unsqueeze(1)
        out, _ = self.graph_processor(unified)
        return self.decoder(out[:, -1, :])


def generate_data_distribution(dist_type, n_samples, seq_len, input_dim, output_dim, seed=42):
    """Generate data with different distribution types."""
    np.random.seed(seed)
    
    if dist_type == "uniform":
        # Uniform distribution - bounded values
        X = np.random.uniform(-1, 1, (n_samples, seq_len, input_dim)).astype(np.float32)
        Y = np.random.uniform(-1, 1, (n_samples, output_dim)).astype(np.float32)
        
    elif dist_type == "normal":
        # Normal distribution - unbounded
        X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
        Y = np.random.randn(n_samples, output_dim).astype(np.float32)
        
    elif dist_type == "multimodal":
        # Multimodal - mixture of gaussians
        X = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
        Y = np.zeros((n_samples, output_dim), dtype=np.float32)
        for i in range(n_samples):
            mode = np.random.randint(0, 3)
            X[i] = np.random.randn(seq_len, input_dim) + mode * 2
            Y[i] = np.random.randn(output_dim) + mode * 2
            
    elif dist_type == "correlated":
        # Temporally correlated sequences
        X = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
        for i in range(n_samples):
            X[i, 0] = np.random.randn(input_dim)
            for t in range(1, seq_len):
                X[i, t] = 0.9 * X[i, t-1] + 0.1 * np.random.randn(input_dim)
        Y = X[:, -1, :output_dim].copy()
        
    elif dist_type == "deterministic":
        # Deterministic relationship (should be learnable)
        X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
        # Create a deterministic relationship
        W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
        Y = np.tanh(X[:, -1, :] @ W)  # Deterministic output from last input
        
    elif dist_type == "identity":
        # Identity mapping (easiest possible task)
        X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
        Y = X[:, -1, :output_dim].copy()  # Output = last input (truncated)
        
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    return X, Y


def generate_libero_style_data(n_samples, seq_len, input_dim=512, output_dim=7, seed=42):
    """Generate synthetic LIBERO-style data (current approach)."""
    np.random.seed(seed)
    
    # This mimics the current synthetic data generation
    X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    
    # Add some structure (simulating physical + semantic)
    physical = X[:, :, :144] * 0.5  # Physical features
    semantic = X[:, :, 144:] * 0.3  # Semantic features
    
    X = np.concatenate([physical, semantic], axis=-1)
    
    # Generate actions with some noise
    Y = np.random.randn(n_samples, output_dim).astype(np.float32) * 0.1
    
    return X, Y


def train_and_evaluate(model, X_train, Y_train, X_val, Y_val, epochs=50, lr=1e-3, verbose=False):
    """Train model and return metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t)
            val_losses.append(val_loss.item())
    
    # Calculate underfit percentage
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    
    # Underfit = val loss significantly higher than train loss
    if final_train > 0:
        underfit_pct = ((final_val - final_train) / final_train) * 100
    else:
        underfit_pct = 0
    
    return {
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "underfit_pct": underfit_pct,
        "train_losses": train_losses,
        "val_losses": val_losses
    }


def run_experiment():
    """Run the data representation investigation."""
    results = {
        "experiment_id": "H1.470.1.1.45",
        "description": "Investigate fundamental data representation issue",
        "timestamp": datetime.now().isoformat(),
        "configurations": [],
        "key_findings": {}
    }
    
    # Parameters
    n_samples = 1000
    seq_len = 10
    input_dim = 512
    output_dim = 7
    epochs = 50
    n_runs = 3
    
    # Test different data distributions
    distributions = [
        "uniform",
        "normal", 
        "multimodal",
        "correlated",
        "deterministic",
        "identity",
        "libero_style"
    ]
    
    # Test different model types
    model_types = ["simple_gru", "cognitive_graph"]
    
    all_results = {}
    
    for dist_type in distributions:
        print(f"\n=== Testing distribution: {dist_type} ===")
        dist_results = {}
        
        for model_type in model_types:
            print(f"  Model: {model_type}")
            run_results = []
            
            for run in range(n_runs):
                seed = 42 + run
                
                # Generate data
                if dist_type == "libero_style":
                    X, Y = generate_libero_style_data(n_samples, seq_len, input_dim, output_dim, seed)
                else:
                    X, Y = generate_data_distribution(dist_type, n_samples, seq_len, input_dim, output_dim, seed)
                
                # Split
                n_train = int(0.8 * n_samples)
                X_train, Y_train = X[:n_train], Y[:n_train]
                X_val, Y_val = X[n_train:], Y[n_train:]
                
                # Create model
                if model_type == "simple_gru":
                    model = SimpleGRU(input_dim, 64, output_dim, 2)
                else:
                    model = CognitiveGraph(input_dim, 64, output_dim, 2)
                
                # Train
                metrics = train_and_evaluate(model, X_train, Y_train, X_val, Y_val, epochs=epochs)
                run_results.append(metrics)
                
                print(f"    Run {run+1}: train={metrics['final_train_loss']:.4f}, "
                      f"val={metrics['final_val_loss']:.4f}, underfit={metrics['underfit_pct']:.1f}%")
            
            # Aggregate
            avg_train = np.mean([r["final_train_loss"] for r in run_results])
            avg_val = np.mean([r["final_val_loss"] for r in run_results])
            avg_underfit = np.mean([r["underfit_pct"] for r in run_results])
            
            dist_results[model_type] = {
                "avg_train_loss": avg_train,
                "avg_val_loss": avg_val,
                "avg_underfit_pct": avg_underfit,
                "runs": run_results
            }
            
            results["configurations"].append({
                "distribution": dist_type,
                "model": model_type,
                "avg_train_loss": avg_train,
                "avg_val_loss": avg_val,
                "avg_underfit_pct": avg_underfit
            })
        
        all_results[dist_type] = dist_results
    
    # Analyze results
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Find best distribution for each model
    best_by_model = {}
    for model_type in model_types:
        best_dist = None
        best_underfit = float('inf')
        for dist_type in distributions:
            underfit = all_results[dist_type][model_type]["avg_underfit_pct"]
            if underfit < best_underfit:
                best_underfit = underfit
                best_dist = dist_type
        best_by_model[model_type] = {"distribution": best_dist, "underfit_pct": best_underfit}
    
    # Check if identity/deterministic tasks are learnable
    identity_learnable = all_results["identity"]["simple_gru"]["avg_underfit_pct"] < 50
    deterministic_learnable = all_results["deterministic"]["simple_gru"]["avg_underfit_pct"] < 50
    
    # Check if libero_style is the problem
    libero_underfit = all_results["libero_style"]["simple_gru"]["avg_underfit_pct"]
    normal_underfit = all_results["normal"]["simple_gru"]["avg_underfit_pct"]
    
    results["key_findings"] = {
        "best_by_model": best_by_model,
        "identity_task_learnable": identity_learnable,
        "deterministic_task_learnable": deterministic_learnable,
        "libero_style_underfit_pct": libero_underfit,
        "normal_underfit_pct": normal_underfit,
        "underfit_by_distribution": {
            dist: {
                "simple_gru": all_results[dist]["simple_gru"]["avg_underfit_pct"],
                "cognitive_graph": all_results[dist]["cognitive_graph"]["avg_underfit_pct"]
            }
            for dist in distributions
        }
    }
    
    # Determine conclusion
    if identity_learnable and deterministic_learnable:
        # Model CAN learn - problem is with data
        if libero_underfit > normal_underfit * 1.5:
            conclusion = "SUPPORTED - libero_style data generation is problematic"
        else:
            conclusion = "INCONCLUSIVE - data distribution not the main issue"
    else:
        conclusion = "REFUTED - model cannot learn even simple tasks (architecture issue)"
    
    results["conclusion"] = conclusion
    
    print(f"\nConclusion: {conclusion}")
    print(f"Identity task learnable: {identity_learnable}")
    print(f"Deterministic task learnable: {deterministic_learnable}")
    print(f"LIBERO-style underfit: {libero_underfit:.1f}%")
    print(f"Normal distribution underfit: {normal_underfit:.1f}%")
    
    # Save results
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    results = run_experiment()