#!/usr/bin/env python3
"""
H1.470.1.1.27: Test Ensemble Disagreement on Complex Hierarchical Tasks (4-5 Phases)
=====================================================================================

Context: H1.470.1.1.26 showed ensemble disagreement performed -4.05% worse than baseline 
on 3-phase hierarchical tasks. The hypothesis is that simple hierarchical structures don't 
provide enough signal for ensemble disagreement to capture useful uncertainty.

Hypothesis: Ensemble disagreement will show improved performance on more complex 
hierarchical tasks (4-5 phases) because:
1. More phase transitions create more uncertainty points
2. Longer sequences allow ensemble to better calibrate disagreement
3. Complex task structure provides richer signal for uncertainty estimation

Test Plan:
1. Generate hierarchical multi-step data with 4-5 phases:
   - Phase 1: Approach (t=0-8)
   - Phase 2: Grasp (t=8-16)
   - Phase 3: Transport (t=16-24)
   - Phase 4: Place (t=24-32)
   - Phase 5: Release (t=32-40) [optional for 5-phase]
2. Test 3 configurations:
   a) Baseline: Standard training
   b) Oracle noise: Ground truth noise levels
   c) Ensemble disagreement: 5-model ensemble variance
3. Evaluate on 40-timestep sequences
4. Compare to H1.470.1.1.26 results (3-phase tasks)

Expected: Ensemble disagreement improvement should increase with task complexity,
potentially turning positive on 4-5 phase tasks.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import random

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device
device = torch.device("cpu")


def generate_hierarchical_multi_step_data(n_samples=800, seq_length=40, n_phases=4):
    """
    Generate hierarchical multi-step robot data with multiple phases.
    
    Args:
        n_samples: Number of samples
        seq_length: Sequence length (40 for 4-phase, 50 for 5-phase)
        n_phases: Number of phases (4 or 5)
    
    Returns:
        X: Input sequences [n_samples, seq_length, 64]
        y: Target sequences [n_samples, seq_length, 7]
        noise_levels: Ground truth noise levels [n_samples, seq_length]
        phase_labels: Phase labels for each timestep [n_samples, seq_length]
    """
    X = np.zeros((n_samples, seq_length, 64), dtype=np.float32)
    y = np.zeros((n_samples, seq_length, 7), dtype=np.float32)
    noise_levels = np.zeros((n_samples, seq_length), dtype=np.float32)
    phase_labels = np.zeros((n_samples, seq_length), dtype=np.int32)
    
    phase_length = seq_length // n_phases
    
    for i in range(n_samples):
        # Generate base physics with high autocorrelation
        physics = np.zeros((seq_length, 32), dtype=np.float32)
        autocorr = 0.85
        physics[0] = np.random.randn(32) * 0.1
        
        for t in range(1, seq_length):
            physics[t] = autocorr * physics[t-1] + np.random.randn(32) * 0.05
        
        # Generate semantic features (language-like)
        semantics = np.zeros((seq_length, 32), dtype=np.float32)
        
        # Phase-specific dynamics
        for phase_idx in range(n_phases):
            start_t = phase_idx * phase_length
            end_t = min((phase_idx + 1) * phase_length, seq_length)
            
            phase_labels[i, start_t:end_t] = phase_idx
            
            # Each phase has different dynamics
            if phase_idx == 0:  # Approach
                # Smooth approach motion
                physics[start_t:end_t, :3] += np.linspace(0, 0.3, end_t - start_t)[:, None]
                semantics[start_t:end_t, :8] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                noise_levels[i, start_t:end_t] = 0.05 + 0.02 * np.random.rand()
                
            elif phase_idx == 1:  # Grasp
                # Gripper closing dynamics
                physics[start_t:end_t, 3:6] += np.linspace(0, 0.2, end_t - start_t)[:, None]
                semantics[start_t:end_t, :8] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
                noise_levels[i, start_t:end_t] = 0.08 + 0.03 * np.random.rand()  # Higher noise at grasp
                
            elif phase_idx == 2:  # Transport
                # Object transport
                physics[start_t:end_t, :3] += np.linspace(0.3, 0.6, end_t - start_t)[:, None]
                physics[start_t:end_t, 6:9] += np.linspace(0, 0.15, end_t - start_t)[:, None]
                semantics[start_t:end_t, :8] = np.array([0, 0, 1, 0, 0, 0, 0, 0])
                noise_levels[i, start_t:end_t] = 0.06 + 0.02 * np.random.rand()
                
            elif phase_idx == 3:  # Place
                # Placement motion
                physics[start_t:end_t, 3:6] += np.linspace(0.2, 0, end_t - start_t)[:, None]
                semantics[start_t:end_t, :8] = np.array([0, 0, 0, 1, 0, 0, 0, 0])
                noise_levels[i, start_t:end_t] = 0.07 + 0.03 * np.random.rand()  # Higher noise at place
                
            elif phase_idx == 4:  # Release (5-phase only)
                # Release motion
                physics[start_t:end_t, 6:9] += np.linspace(0.15, 0, end_t - start_t)[:, None]
                semantics[start_t:end_t, :8] = np.array([0, 0, 0, 0, 1, 0, 0, 0])
                noise_levels[i, start_t:end_t] = 0.05 + 0.02 * np.random.rand()
        
        # Add phase transition noise spikes
        for phase_idx in range(1, n_phases):
            transition_t = phase_idx * phase_length
            if transition_t < seq_length:
                # Spike in noise at phase transitions
                noise_levels[i, max(0, transition_t-2):min(seq_length, transition_t+3)] *= 1.5
        
        # Combine physics and semantics
        X[i, :, :32] = physics
        X[i, :, 32:64] = semantics
        
        # Generate targets (7-DOF robot action)
        y[i, :, :3] = physics[:, :3] * 0.5  # Position
        y[i, :, 3:6] = physics[:, 3:6] * 0.3  # Orientation
        y[i, :, 6] = np.sum(semantics[:, :4], axis=1) * 0.2  # Gripper state
        
        # Add noise to targets
        for t in range(seq_length):
            y[i, t] += np.random.randn(7) * noise_levels[i, t]
    
    return X, y, noise_levels, phase_labels


class SimplePredictor(nn.Module):
    """Simple predictor for multi-step robot tasks."""
    
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class EnsemblePredictor(nn.Module):
    """Ensemble of predictors for uncertainty estimation."""
    
    def __init__(self, n_models=5, input_dim=64, hidden_dim=128, output_dim=7):
        super().__init__()
        self.models = nn.ModuleList([
            SimplePredictor(input_dim, hidden_dim, output_dim)
            for _ in range(n_models)
        ])
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0)  # [n_models, batch, seq, output]
    
    def predict_with_uncertainty(self, x):
        outputs = self.forward(x)
        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)
        return mean, variance


def train_baseline(X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    """Train baseline model without noise-aware loss."""
    model = SimplePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = F.mse_loss(pred, y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = F.mse_loss(test_pred, y_test_t).item()
    
    return model, test_loss


def train_oracle_noise(X_train, y_train, noise_train, X_test, y_test, epochs=100, lr=0.001):
    """Train with oracle noise levels (upper bound)."""
    model = SimplePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    noise_train_t = torch.FloatTensor(noise_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        
        # Noise-weighted loss
        sample_loss = (pred - y_train_t) ** 2
        # Weight by inverse noise (lower noise = higher weight)
        weights = 1.0 / (noise_train_t.unsqueeze(-1) + 0.01)
        weighted_loss = (sample_loss * weights).mean()
        
        weighted_loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = F.mse_loss(test_pred, y_test_t).item()
    
    return model, test_loss


def train_ensemble_disagreement(X_train, y_train, X_test, y_test, epochs=100, lr=0.001, n_models=5):
    """Train ensemble and use disagreement for noise estimation."""
    ensemble = EnsemblePredictor(n_models=n_models).to(device)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    for epoch in range(epochs):
        ensemble.train()
        optimizer.zero_grad()
        
        # Get predictions from all models
        outputs = ensemble(X_train_t)  # [n_models, batch, seq, output]
        
        # Compute mean and disagreement
        mean_pred = outputs.mean(dim=0)
        disagreement = outputs.var(dim=0).mean(dim=-1)  # Variance across models
        
        # Weight samples by inverse disagreement
        weights = 1.0 / (disagreement + 0.01)
        weights = weights / weights.mean()  # Normalize
        
        # Weighted loss
        sample_loss = ((mean_pred - y_train_t) ** 2).mean(dim=-1)
        weighted_loss = (sample_loss * weights).mean()
        
        weighted_loss.backward()
        optimizer.step()
    
    ensemble.eval()
    with torch.no_grad():
        mean_pred, _ = ensemble.predict_with_uncertainty(X_test_t)
        test_loss = F.mse_loss(mean_pred, y_test_t).item()
    
    return ensemble, test_loss


def run_experiment(n_phases=4, n_samples=800, seq_length=40, epochs=100):
    """Run single experiment with specified number of phases."""
    print(f"\n{'='*60}")
    print(f"Testing {n_phases}-phase hierarchical tasks")
    print(f"{'='*60}")
    
    # Generate data
    X, y, noise_levels, phase_labels = generate_hierarchical_multi_step_data(
        n_samples=n_samples, 
        seq_length=seq_length, 
        n_phases=n_phases
    )
    
    # Split data
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    noise_train, noise_test = noise_levels[:n_train], noise_levels[n_train:]
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Train samples: {n_train}, Test samples: {n_samples - n_train}")
    
    # Train baseline
    print("\nTraining baseline...")
    baseline_model, baseline_loss = train_baseline(X_train, y_train, X_test, y_test, epochs=epochs)
    print(f"Baseline test loss: {baseline_loss:.6f}")
    
    # Train oracle noise
    print("\nTraining oracle noise...")
    oracle_model, oracle_loss = train_oracle_noise(X_train, y_train, noise_train, X_test, y_test, epochs=epochs)
    print(f"Oracle noise test loss: {oracle_loss:.6f}")
    
    # Train ensemble disagreement
    print("\nTraining ensemble disagreement...")
    ensemble_model, ensemble_loss = train_ensemble_disagreement(X_train, y_train, X_test, y_test, epochs=epochs)
    print(f"Ensemble disagreement test loss: {ensemble_loss:.6f}")
    
    # Compute improvements
    baseline_improvement = 0.0
    oracle_improvement = (baseline_loss - oracle_loss) / baseline_loss * 100
    ensemble_improvement = (baseline_loss - ensemble_loss) / baseline_loss * 100
    
    # Oracle ratio (ensemble improvement / oracle improvement)
    if abs(oracle_improvement) > 0.01:
        oracle_ratio = ensemble_improvement / oracle_improvement * 100
    else:
        oracle_ratio = float('inf') if ensemble_improvement > 0 else -float('inf')
    
    results = {
        "n_phases": n_phases,
        "n_samples": n_samples,
        "seq_length": seq_length,
        "baseline_test_loss": baseline_loss,
        "oracle_test_loss": oracle_loss,
        "ensemble_test_loss": ensemble_loss,
        "baseline_improvement_pct": baseline_improvement,
        "oracle_improvement_pct": oracle_improvement,
        "ensemble_improvement_pct": ensemble_improvement,
        "oracle_ratio_pct": oracle_ratio
    }
    
    print(f"\nResults for {n_phases}-phase tasks:")
    print(f"  Baseline improvement: {baseline_improvement:.2f}%")
    print(f"  Oracle improvement: {oracle_improvement:.2f}%")
    print(f"  Ensemble improvement: {ensemble_improvement:.2f}%")
    print(f"  Oracle ratio: {oracle_ratio:.1f}%")
    
    return results


def main():
    """Run experiments with 4 and 5 phases."""
    print("="*70)
    print("H1.470.1.1.27: Ensemble Disagreement on Complex Hierarchical Tasks")
    print("="*70)
    
    all_results = []
    
    # Test 4-phase tasks
    results_4phase = run_experiment(n_phases=4, n_samples=800, seq_length=40, epochs=100)
    all_results.append(results_4phase)
    
    # Test 5-phase tasks
    results_5phase = run_experiment(n_phases=5, n_samples=800, seq_length=50, epochs=100)
    all_results.append(results_5phase)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n| Phases | Seq Len | Baseline | Oracle | Ensemble | Oracle Imp | Ensemble Imp | Oracle Ratio |")
    print("|--------|---------|----------|--------|----------|------------|--------------|--------------|")
    
    for r in all_results:
        print(f"| {r['n_phases']:>6} | {r['seq_length']:>7} | {r['baseline_test_loss']:>8.4f} | {r['oracle_test_loss']:>6.4f} | {r['ensemble_test_loss']:>8.4f} | {r['oracle_improvement_pct']:>10.2f}% | {r['ensemble_improvement_pct']:>12.2f}% | {r['oracle_ratio_pct']:>12.1f}% |")
    
    # Compare to H1.470.1.1.26 (3-phase results)
    print("\nComparison to H1.470.1.1.26 (3-phase tasks):")
    print("  H1.470.1.1.26: Ensemble improvement = -4.05% (worse than baseline)")
    print(f"  H1.470.1.1.27 (4-phase): Ensemble improvement = {results_4phase['ensemble_improvement_pct']:.2f}%")
    print(f"  H1.470.1.1.27 (5-phase): Ensemble improvement = {results_5phase['ensemble_improvement_pct']:.2f}%")
    
    # Determine conclusion
    if results_4phase['ensemble_improvement_pct'] > 0 or results_5phase['ensemble_improvement_pct'] > 0:
        conclusion = "SUPPORTED - Ensemble disagreement improves on complex hierarchical tasks"
    else:
        conclusion = "REFUTED - Ensemble disagreement still fails on complex hierarchical tasks"
    
    print(f"\nConclusion: {conclusion}")
    
    # Save results
    output = {
        "experiment_id": "H1.470.1.1.27",
        "timestamp": datetime.now().isoformat(),
        "description": "Test ensemble disagreement on complex hierarchical tasks (4-5 phases)",
        "results": all_results,
        "comparison_to_H1_470_1_1_26": {
            "H1_470_1_1_26_ensemble_improvement": -4.05,
            "H1_470_1_1_27_4phase_improvement": results_4phase['ensemble_improvement_pct'],
            "H1_470_1_1_27_5phase_improvement": results_5phase['ensemble_improvement_pct']
        },
        "conclusion": conclusion
    }
    
    output_path = Path(__file__).parent.parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output


if __name__ == "__main__":
    main()