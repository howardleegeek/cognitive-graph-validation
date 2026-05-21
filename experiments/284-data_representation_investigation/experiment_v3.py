#!/usr/bin/env python3
"""
H1.470.1.1.45-v3: Overfitting Root Cause Analysis

Critical finding from v2:
- GRU: Train loss 0.0005, Val loss 1.46 (288682% "underfit" - actually overfitting!)
- Linear model: Train loss 0.86, Val loss 1.97 (127.8% underfit - much better generalization)

The GRU is MEMORIZING training data instead of learning the underlying pattern.
This is the opposite of what we thought - it's not underfitting, it's SEVERE OVERFITTING.

Hypothesis: The GRU architecture with current settings is prone to memorization.
Test:
1. Add regularization (dropout, weight decay)
2. Reduce model capacity
3. Use early stopping
4. Compare with simpler architectures
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class SimpleGRU(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, num_layers=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.decoder(out[:, -1, :])


class SimpleGRUWithDropout(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.decoder(out)


class TinyGRU(nn.Module):
    """Very small GRU to prevent memorization."""
    def __init__(self, input_dim=512, hidden_dim=16, output_dim=7):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.decoder(out[:, -1, :])


class LinearBaseline(nn.Module):
    """Simple linear model - just uses last timestep."""
    def __init__(self, input_dim=512, output_dim=7):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, -1, :]  # Use last timestep
        return self.linear(x)


class MLPBaseline(nn.Module):
    """Simple MLP - uses last timestep with one hidden layer."""
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, -1, :]
        return self.net(x)


def generate_identity_data(n_samples, seq_len, input_dim, output_dim, seed=42):
    """Generate identity mapping data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    Y = X[:, -1, :output_dim].copy()
    return X, Y


def train_with_early_stopping(model, X_train, Y_train, X_val, Y_val, epochs=200, lr=1e-3, 
                               weight_decay=0.0, patience=10, verbose=False):
    """Train with early stopping based on validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
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
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"   Early stopping at epoch {epoch}")
                    break
    
    final_train = train_losses[best_epoch]
    final_val = val_losses[best_epoch]
    underfit_pct = ((final_val - final_train) / max(final_train, 1e-8)) * 100
    
    return {
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "underfit_pct": underfit_pct,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "train_losses": train_losses,
        "val_losses": val_losses
    }


def run_experiment():
    """Run the overfitting root cause analysis."""
    results = {
        "experiment_id": "H1.470.1.1.45-v3",
        "description": "Overfitting root cause analysis",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    n_samples = 1000
    seq_len = 10
    input_dim = 512
    output_dim = 7
    n_train = 800
    epochs = 200
    
    # Generate identity data
    X_all, Y_all = generate_identity_data(n_samples, seq_len, input_dim, output_dim, seed=42)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]
    
    print("="*70)
    print("TEST: Identity Task - Different Architectures and Regularization")
    print("="*70)
    
    configs = [
        ("Linear (baseline)", LinearBaseline(input_dim, output_dim), {}),
        ("MLP h64", MLPBaseline(input_dim, 64, output_dim), {}),
        ("MLP h64 + dropout 0.3", MLPBaseline(input_dim, 64, output_dim, dropout=0.3), {}),
        ("MLP h16", MLPBaseline(input_dim, 16, output_dim), {}),
        ("TinyGRU h16", TinyGRU(input_dim, 16, output_dim), {}),
        ("GRU h16", SimpleGRU(input_dim, 16, output_dim, num_layers=1), {}),
        ("GRU h32", SimpleGRU(input_dim, 32, output_dim, num_layers=1), {}),
        ("GRU h64", SimpleGRU(input_dim, 64, output_dim, num_layers=2), {}),
        ("GRU h64 + dropout 0.3", SimpleGRUWithDropout(input_dim, 64, output_dim, num_layers=2, dropout=0.3), {}),
        ("GRU h64 + wd 1e-4", SimpleGRU(input_dim, 64, output_dim, num_layers=2), {"weight_decay": 1e-4}),
        ("GRU h64 + wd 1e-3", SimpleGRU(input_dim, 64, output_dim, num_layers=2), {"weight_decay": 1e-3}),
        ("GRU h64 + wd 1e-2", SimpleGRU(input_dim, 64, output_dim, num_layers=2), {"weight_decay": 1e-2}),
    ]
    
    for name, model, extra_kwargs in configs:
        print(f"\n{name}:")
        kwargs = {"epochs": epochs, "patience": 20}
        kwargs.update(extra_kwargs)
        metrics = train_with_early_stopping(model, X_train, Y_train, X_val, Y_val, **kwargs)
        
        print(f"   Train loss: {metrics['final_train_loss']:.6f}")
        print(f"   Val loss: {metrics['final_val_loss']:.6f}")
        print(f"   Underfit: {metrics['underfit_pct']:.1f}%")
        print(f"   Best epoch: {metrics['best_epoch']}")
        
        results["tests"].append({
            "name": name,
            "train_loss": metrics["final_train_loss"],
            "val_loss": metrics["final_val_loss"],
            "underfit_pct": metrics["underfit_pct"],
            "best_epoch": metrics["best_epoch"],
            "total_epochs": metrics["total_epochs"]
        })
    
    # Find best configuration
    best_config = min(results["tests"], key=lambda x: x["underfit_pct"])
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBest configuration: {best_config['name']}")
    print(f"  Train loss: {best_config['train_loss']:.6f}")
    print(f"  Val loss: {best_config['val_loss']:.6f}")
    print(f"  Underfit: {best_config['underfit_pct']:.1f}%")
    
    # Analyze patterns
    linear_underfit = next(t for t in results["tests"] if t["name"] == "Linear (baseline)")["underfit_pct"]
    gru_h64_underfit = next(t for t in results["tests"] if t["name"] == "GRU h64")["underfit_pct"]
    gru_h64_wd_underfit = next(t for t in results["tests"] if t["name"] == "GRU h64 + wd 1e-2")["underfit_pct"]
    
    if gru_h64_wd_underfit < gru_h64_underfit * 0.5:
        conclusion = "SUPPORTED - Weight decay significantly reduces overfitting"
        key_insight = f"Weight decay 1e-2 reduces underfit from {gru_h64_underfit:.1f}% to {gru_h64_wd_underfit:.1f}%"
    elif linear_underfit < gru_h64_underfit * 0.1:
        conclusion = "SUPPORTED - Simpler architectures generalize better"
        key_insight = f"Linear model ({linear_underfit:.1f}%) vastly outperforms GRU ({gru_h64_underfit:.1f}%)"
    else:
        conclusion = "INCONCLUSIVE"
        key_insight = "No clear pattern found"
    
    results["conclusion"] = conclusion
    results["key_insight"] = key_insight
    results["best_config"] = best_config
    
    print(f"\nCONCLUSION: {conclusion}")
    print(f"KEY INSIGHT: {key_insight}")
    
    # Save results
    output_path = Path(__file__).parent / "results_v3.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment()