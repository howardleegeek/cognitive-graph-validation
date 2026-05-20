#!/usr/bin/env python3
"""
Quick test of ensemble disagreement on multi-step tasks.
Simplified version that runs quickly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from datetime import datetime

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_simple_multi_step_data(n_samples=200, seq_length=20):
    """Generate simple multi-step data for quick testing."""
    X = np.zeros((n_samples, seq_length, 16), dtype=np.float32)
    y = np.zeros((n_samples, seq_length, 3), dtype=np.float32)
    noise = np.zeros((n_samples, seq_length), dtype=np.float32)
    
    for i in range(n_samples):
        # Simple correlated process
        base = np.zeros((seq_length, 8))
        base[0] = np.random.randn(8) * 0.1
        for t in range(1, seq_length):
            base[t] = 0.8 * base[t-1] + np.random.randn(8) * 0.05
        
        # Add some phase structure
        if seq_length > 10:
            mid = seq_length // 2
            base[:mid] += np.sin(np.linspace(0, np.pi, mid))[:, None] * 0.1
            base[mid:] += np.cos(np.linspace(0, np.pi, seq_length-mid))[:, None] * 0.1
        
        # Features
        features = np.random.randn(seq_length, 8).astype(np.float32) * 0.05
        X[i] = np.concatenate([base, features], axis=-1)
        
        # Targets (simple transformation)
        y[i] = np.tanh(base[:, :3] * 0.5 + features[:, :3] * 0.3)
        
        # Noise (phase-dependent)
        noise[i, :seq_length//2] = 0.02
        noise[i, seq_length//2:] = 0.04
    
    return X, y, noise

class SimpleModel(nn.Module):
    """Very simple model for quick testing."""
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=3, seq_length=20):
        super().__init__()
        self.seq_length = seq_length
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        encoded = self.encoder(x)
        gru_out, _ = self.gru(encoded)
        output = self.decoder(gru_out)
        return output

def train_simple(model, X_train, y_train, X_val, y_val, epochs=20, lr=0.01):
    """Simple training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = F.mse_loss(predictions, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = F.mse_loss(val_pred, y_val).item()
            val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train={loss.item():.6f}, val={val_loss:.6f}")
    
    return train_losses, val_losses

def main():
    print("Quick test of ensemble disagreement on multi-step tasks")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating data...")
    X, y, noise = generate_simple_multi_step_data(n_samples=200, seq_length=20)
    
    # Split
    n_train = 150
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    noise_train, noise_val = noise[:n_train], noise[n_train:]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    noise_train_t = torch.FloatTensor(noise_train)
    
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Train: {n_train} samples, Val: {50} samples")
    
    # Test 1: Baseline
    print("\n2. Training baseline model...")
    baseline_model = SimpleModel(seq_length=20)
    baseline_train_loss, baseline_val_loss = train_simple(
        baseline_model, X_train_t, y_train_t, X_val_t, y_val_t, epochs=20
    )
    baseline_final = baseline_val_loss[-1]
    print(f"   Baseline final loss: {baseline_final:.6f}")
    
    # Test 2: Train ensemble for disagreement
    print("\n3. Training ensemble (3 models)...")
    n_ensemble = 3
    ensemble_models = []
    
    for i in range(n_ensemble):
        print(f"   Training model {i+1}/{n_ensemble}...")
        model = SimpleModel(seq_length=20)
        train_loss, val_loss = train_simple(
            model, X_train_t, y_train_t, X_val_t, y_val_t, epochs=20
        )
        ensemble_models.append(model)
    
    # Compute ensemble disagreement
    print("\n4. Computing ensemble disagreement...")
    all_preds = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            pred = model(X_train_t).numpy()
            all_preds.append(pred)
    
    all_preds = np.stack(all_preds, axis=0)  # [n_models, n_samples, seq_len, output_dim]
    disagreement = np.var(all_preds, axis=0).mean(axis=-1)  # [n_samples, seq_len]
    
    # Normalize disagreement
    disagreement = (disagreement - disagreement.min()) / (disagreement.max() - disagreement.min() + 1e-8)
    disagreement = disagreement * 0.2 + 0.05  # Scale to [0.05, 0.25]
    disagreement_t = torch.FloatTensor(disagreement)
    
    print(f"   Disagreement stats: min={disagreement.min():.4f}, max={disagreement.max():.4f}, mean={disagreement.mean():.4f}")
    
    # Test 3: Model trained with disagreement-based weighting
    print("\n5. Training with disagreement-weighted loss...")
    
    # Create weighted loss function
    def weighted_loss(pred, target, weights):
        return (weights.unsqueeze(-1) * (pred - target) ** 2).mean()
    
    ed_model = SimpleModel(seq_length=20)
    optimizer = torch.optim.Adam(ed_model.parameters(), lr=0.01)
    
    ed_train_losses = []
    ed_val_losses = []
    
    for epoch in range(20):
        # Train
        ed_model.train()
        optimizer.zero_grad()
        predictions = ed_model(X_train_t)
        loss = weighted_loss(predictions, y_train_t, 1.0/(disagreement_t + 0.1))
        loss.backward()
        optimizer.step()
        ed_train_losses.append(loss.item())
        
        # Validate
        ed_model.eval()
        with torch.no_grad():
            val_pred = ed_model(X_val_t)
            val_loss = F.mse_loss(val_pred, y_val_t).item()
            ed_val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch}: train={loss.item():.6f}, val={val_loss:.6f}")
    
    ed_final = ed_val_losses[-1]
    improvement = (baseline_final - ed_final) / baseline_final * 100
    
    print(f"\n6. Results:")
    print(f"   Baseline loss: {baseline_final:.6f}")
    print(f"   Ensemble disagreement loss: {ed_final:.6f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    # Compare with oracle (true noise)
    print("\n7. Comparing with oracle (true noise)...")
    
    # Model trained with true noise weighting
    oracle_model = SimpleModel(seq_length=20)
    optimizer = torch.optim.Adam(oracle_model.parameters(), lr=0.01)
    
    oracle_train_losses = []
    oracle_val_losses = []
    
    for epoch in range(20):
        # Train
        oracle_model.train()
        optimizer.zero_grad()
        predictions = oracle_model(X_train_t)
        loss = weighted_loss(predictions, y_train_t, 1.0/(noise_train_t + 0.1))
        loss.backward()
        optimizer.step()
        oracle_train_losses.append(loss.item())
        
        # Validate
        oracle_model.eval()
        with torch.no_grad():
            val_pred = oracle_model(X_val_t)
            val_loss = F.mse_loss(val_pred, y_val_t).item()
            oracle_val_losses.append(val_loss)
    
    oracle_final = oracle_val_losses[-1]
    oracle_improvement = (baseline_final - oracle_final) / baseline_final * 100
    
    print(f"   Oracle (true noise) loss: {oracle_final:.6f}")
    print(f"   Oracle improvement: {oracle_improvement:.2f}%")
    
    # Calculate oracle ratio
    if oracle_improvement > 0:
        oracle_ratio = improvement / oracle_improvement * 100
    else:
        oracle_ratio = float('inf') if improvement > 0 else 0
    
    print(f"   Ensemble disagreement achieves {oracle_ratio:.1f}% of oracle performance")
    
    # Save results
    results = {
        'experiment_id': 'H1.470.1.1.25-quick',
        'description': 'Quick test of ensemble disagreement on multi-step tasks',
        'seq_length': 20,
        'n_samples': 200,
        'key_metrics': {
            'baseline_test_loss': float(baseline_final),
            'oracle_test_loss': float(oracle_final),
            'oracle_improvement': float(oracle_improvement),
            'ensemble_disagreement_loss': float(ed_final),
            'ensemble_disagreement_improvement': float(improvement),
            'ensemble_disagreement_oracle_ratio': float(oracle_ratio),
            'ensemble_outperforms_oracle': ed_final < oracle_final,
            'ensemble_outperformance_margin': float(oracle_final - ed_final) / oracle_final * 100 if oracle_final > 0 else 0
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_dir = Path("experiments/H1.470.1.1.25-ensemble-disagreement-multi-step-real-robot")
    with open(output_dir / "quick_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'quick_results.json'}")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if ed_final < oracle_final:
        print("✓ SUPPORTED: Ensemble disagreement outperforms oracle noise")
        print(f"  on multi-step tasks ({improvement:.2f}% vs {oracle_improvement:.2f}% improvement)")
    elif ed_final < baseline_final:
        print("✓ PARTIALLY SUPPORTED: Ensemble disagreement helps but doesn't beat oracle")
        print(f"  ({improvement:.2f}% improvement, {oracle_ratio:.1f}% of oracle)")
    else:
        print("✗ REFUTED: Ensemble disagreement doesn't help on multi-step tasks")
    
    return results

if __name__ == "__main__":
    main()