#!/usr/bin/env python3
"""
H1.470.1.1.45-v2: Train/Val Split Distribution Shift Investigation

Critical finding from v1:
- Multimodal distribution: 6-20% underfit (GOOD)
- Identity task: 3000-5000% underfit (TERRIBLE - should be trivial!)

Hypothesis: The random train/val split is creating distribution shift.
For identity/deterministic tasks, the validation set has completely different
patterns than training, causing massive generalization gap.

Test: Use stratified splitting or same-seed generation to ensure
train and val come from the same distribution.
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
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, num_layers=2):
        super().__init__()
        self.physical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 144)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 368)
        )
        self.graph_processor = nn.GRU(512, hidden_dim, num_layers=num_layers, batch_first=True)
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


def generate_identity_data(n_samples, seq_len, input_dim, output_dim, seed=42):
    """Generate identity mapping data - output = last input (truncated)."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    Y = X[:, -1, :output_dim].copy()
    return X, Y


def generate_deterministic_data(n_samples, seq_len, input_dim, output_dim, seed=42):
    """Generate deterministic relationship data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
    Y = np.tanh(X[:, -1, :] @ W)
    return X, Y, W


def train_and_evaluate(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3):
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
    
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    underfit_pct = ((final_val - final_train) / max(final_train, 1e-8)) * 100
    
    return {
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "underfit_pct": underfit_pct,
        "train_losses": train_losses,
        "val_losses": val_losses
    }


def run_experiment():
    """Run the train/val split investigation."""
    results = {
        "experiment_id": "H1.470.1.1.45-v2",
        "description": "Train/val split distribution shift investigation",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    n_samples = 1000
    seq_len = 10
    input_dim = 512
    output_dim = 7
    epochs = 100
    n_train = 800
    
    print("="*70)
    print("TEST 1: Identity Task with Different Split Strategies")
    print("="*70)
    
    # Generate all data with same seed
    X_all, Y_all = generate_identity_data(n_samples, seq_len, input_dim, output_dim, seed=42)
    
    # Test 1a: Random split (original approach - problematic)
    print("\n1a. Random split (original):")
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]
    
    model = SimpleGRU(input_dim, 64, output_dim, 2)
    metrics_random = train_and_evaluate(model, X_train, Y_train, X_val, Y_val, epochs=epochs)
    print(f"   Train loss: {metrics_random['final_train_loss']:.6f}")
    print(f"   Val loss: {metrics_random['final_val_loss']:.6f}")
    print(f"   Underfit: {metrics_random['underfit_pct']:.1f}%")
    
    # Test 1b: Sequential split (first 80% train, last 20% val)
    print("\n1b. Sequential split:")
    X_train_seq, Y_train_seq = X_all[:n_train], Y_all[:n_train]
    X_val_seq, Y_val_seq = X_all[n_train:], Y_all[n_train:]
    
    model = SimpleGRU(input_dim, 64, output_dim, 2)
    metrics_seq = train_and_evaluate(model, X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, epochs=epochs)
    print(f"   Train loss: {metrics_seq['final_train_loss']:.6f}")
    print(f"   Val loss: {metrics_seq['final_val_loss']:.6f}")
    print(f"   Underfit: {metrics_seq['underfit_pct']:.1f}%")
    
    # Test 1c: Generate train and val from SAME distribution with different seeds
    print("\n1c. Same distribution, different seeds:")
    X_train_sd, Y_train_sd = generate_identity_data(n_train, seq_len, input_dim, output_dim, seed=42)
    X_val_sd, Y_val_sd = generate_identity_data(n_samples - n_train, seq_len, input_dim, output_dim, seed=43)
    
    model = SimpleGRU(input_dim, 64, output_dim, 2)
    metrics_sd = train_and_evaluate(model, X_train_sd, Y_train_sd, X_val_sd, Y_val_sd, epochs=epochs)
    print(f"   Train loss: {metrics_sd['final_train_loss']:.6f}")
    print(f"   Val loss: {metrics_sd['final_val_loss']:.6f}")
    print(f"   Underfit: {metrics_sd['underfit_pct']:.1f}%")
    
    print("\n" + "="*70)
    print("TEST 2: Deterministic Task with Different Split Strategies")
    print("="*70)
    
    # Generate deterministic data
    X_all_det, Y_all_det, W = generate_deterministic_data(n_samples, seq_len, input_dim, output_dim, seed=42)
    
    # Test 2a: Random split
    print("\n2a. Random split:")
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train, Y_train = X_all_det[train_idx], Y_all_det[train_idx]
    X_val, Y_val = X_all_det[val_idx], Y_all_det[val_idx]
    
    model = SimpleGRU(input_dim, 64, output_dim, 2)
    metrics_det_random = train_and_evaluate(model, X_train, Y_train, X_val, Y_val, epochs=epochs)
    print(f"   Train loss: {metrics_det_random['final_train_loss']:.6f}")
    print(f"   Val loss: {metrics_det_random['final_val_loss']:.6f}")
    print(f"   Underfit: {metrics_det_random['underfit_pct']:.1f}%")
    
    # Test 2b: Sequential split
    print("\n2b. Sequential split:")
    X_train_seq, Y_train_seq = X_all_det[:n_train], Y_all_det[:n_train]
    X_val_seq, Y_val_seq = X_all_det[n_train:], Y_all_det[n_train:]
    
    model = SimpleGRU(input_dim, 64, output_dim, 2)
    metrics_det_seq = train_and_evaluate(model, X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, epochs=epochs)
    print(f"   Train loss: {metrics_det_seq['final_train_loss']:.6f}")
    print(f"   Val loss: {metrics_det_seq['final_val_loss']:.6f}")
    print(f"   Underfit: {metrics_det_seq['underfit_pct']:.1f}%")
    
    # Test 2c: Same distribution, different seeds
    print("\n2c. Same distribution, different seeds:")
    X_train_sd, Y_train_sd, _ = generate_deterministic_data(n_train, seq_len, input_dim, output_dim, seed=42)
    X_val_sd, Y_val_sd, _ = generate_deterministic_data(n_samples - n_train, seq_len, input_dim, output_dim, seed=43)
    
    model = SimpleGRU(input_dim, 64, output_dim, 2)
    metrics_det_sd = train_and_evaluate(model, X_train_sd, Y_train_sd, X_val_sd, Y_val_sd, epochs=epochs)
    print(f"   Train loss: {metrics_det_sd['final_train_loss']:.6f}")
    print(f"   Val loss: {metrics_det_sd['final_val_loss']:.6f}")
    print(f"   Underfit: {metrics_det_sd['underfit_pct']:.1f}%")
    
    print("\n" + "="*70)
    print("TEST 3: Direct Weight Learning (Can model learn the transformation?)")
    print("="*70)
    
    # Test if model can learn a simple linear transformation
    print("\n3. Learning a known linear transformation:")
    np.random.seed(42)
    W_true = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
    
    # Generate data where Y = X @ W_true
    X_lin = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    Y_lin = X_lin[:, -1, :] @ W_true  # Use last timestep
    
    # Split
    X_train_lin, Y_train_lin = X_lin[:n_train], Y_lin[:n_train]
    X_val_lin, Y_val_lin = X_lin[n_train:], Y_lin[n_train:]
    
    model = SimpleGRU(input_dim, 64, output_dim, 2)
    metrics_lin = train_and_evaluate(model, X_train_lin, Y_train_lin, X_val_lin, Y_val_lin, epochs=epochs)
    print(f"   Train loss: {metrics_lin['final_train_loss']:.6f}")
    print(f"   Val loss: {metrics_lin['final_val_loss']:.6f}")
    print(f"   Underfit: {metrics_lin['underfit_pct']:.1f}%")
    
    # Also test with a simple linear model
    print("\n3b. Simple linear model (baseline):")
    linear_model = nn.Linear(input_dim, output_dim)
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_lin[:n_train, -1, :])  # Just use last timestep
    Y_train_t = torch.FloatTensor(Y_lin[:n_train])
    X_val_t = torch.FloatTensor(X_lin[n_train:, -1, :])
    Y_val_t = torch.FloatTensor(Y_lin[n_train:])
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = linear_model(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        val_pred = linear_model(X_val_t)
        val_loss = criterion(val_pred, Y_val_t).item()
    
    print(f"   Train loss: {loss.item():.6f}")
    print(f"   Val loss: {val_loss:.6f}")
    print(f"   Underfit: {((val_loss - loss.item()) / loss.item() * 100):.1f}%")
    
    # Compile results
    results["tests"] = [
        {
            "name": "identity_random_split",
            "train_loss": metrics_random["final_train_loss"],
            "val_loss": metrics_random["final_val_loss"],
            "underfit_pct": metrics_random["underfit_pct"]
        },
        {
            "name": "identity_sequential_split",
            "train_loss": metrics_seq["final_train_loss"],
            "val_loss": metrics_seq["final_val_loss"],
            "underfit_pct": metrics_seq["underfit_pct"]
        },
        {
            "name": "identity_same_distribution",
            "train_loss": metrics_sd["final_train_loss"],
            "val_loss": metrics_sd["final_val_loss"],
            "underfit_pct": metrics_sd["underfit_pct"]
        },
        {
            "name": "deterministic_random_split",
            "train_loss": metrics_det_random["final_train_loss"],
            "val_loss": metrics_det_random["final_val_loss"],
            "underfit_pct": metrics_det_random["underfit_pct"]
        },
        {
            "name": "deterministic_sequential_split",
            "train_loss": metrics_det_seq["final_train_loss"],
            "val_loss": metrics_det_seq["final_val_loss"],
            "underfit_pct": metrics_det_seq["underfit_pct"]
        },
        {
            "name": "deterministic_same_distribution",
            "train_loss": metrics_det_sd["final_train_loss"],
            "val_loss": metrics_det_sd["final_val_loss"],
            "underfit_pct": metrics_det_sd["underfit_pct"]
        },
        {
            "name": "linear_transformation_gru",
            "train_loss": metrics_lin["final_train_loss"],
            "val_loss": metrics_lin["final_val_loss"],
            "underfit_pct": metrics_lin["underfit_pct"]
        }
    ]
    
    # Determine conclusion
    # If same-distribution split reduces underfit significantly, the issue is train/val distribution shift
    same_dist_underfit = min(metrics_sd["underfit_pct"], metrics_det_sd["underfit_pct"])
    random_split_underfit = min(metrics_random["underfit_pct"], metrics_det_random["underfit_pct"])
    
    if same_dist_underfit < random_split_underfit * 0.5:
        conclusion = "SUPPORTED - Train/val distribution shift is a major factor"
        results["key_insight"] = f"Same-distribution split reduces underfit from {random_split_underfit:.1f}% to {same_dist_underfit:.1f}%"
    else:
        conclusion = "INCONCLUSIVE - Split strategy doesn't explain underfitting"
        results["key_insight"] = "Distribution shift not the primary cause"
    
    results["conclusion"] = conclusion
    print(f"\n\nCONCLUSION: {conclusion}")
    
    # Save results
    output_path = Path(__file__).parent / "results_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment()