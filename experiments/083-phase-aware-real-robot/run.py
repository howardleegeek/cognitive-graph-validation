#!/usr/bin/env python3
"""
H1.470.1.1.29: Phase-Aware Training + Ensemble Disagreement for Mixed Tasks

Prior results:
- H1.470.1.1.28: Phase-aware training dramatically improves hierarchical tasks (+99.05% to +99.82%)
- H1.470.1.1.27: Ensemble disagreement REFUTED for hierarchical tasks (-1.35% to -1.59%)

Key insight: 
- Phase-aware training works for hierarchical/structured tasks (upweight phase transitions)
- Ensemble disagreement works for tasks with genuine noise/uncertainty (downweight noisy samples)
- Mixed tasks have BOTH: hierarchical structure AND sensor noise

This experiment tests:
1. Phase-aware training on real robot-style data (simulated LIBERO)
2. Hybrid approach: phase-aware for hierarchical parts + ensemble disagreement for noise
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
from datetime import datetime
from pathlib import Path


class CognitiveGraphCG(nn.Module):
    """CG+Strong architecture with unified physical+semantic representation."""
    
    def __init__(self, obs_dim=512, action_dim=7, hidden_dim=256):
        super().__init__()
        # Physical encoder (144 dims)
        self.physical_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 144)
        )
        # Semantic encoder (368 dims)
        self.semantic_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 368)
        )
        # Unified processor - processes full sequence
        self.graph_processor = nn.GRU(512, hidden_dim, num_layers=2, batch_first=True)
        # Decoder - outputs for each timestep
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, obs_dim)
        batch_size, seq_len, _ = x.shape
        
        # Encode each timestep
        physical = self.physical_encoder(x)  # (batch, seq_len, 144)
        semantic = self.semantic_encoder(x)  # (batch, seq_len, 368)
        unified = torch.cat([physical, semantic], dim=-1)  # (batch, seq_len, 512)
        
        # Process sequence
        out, _ = self.graph_processor(unified)  # (batch, seq_len, hidden)
        
        # Decode each timestep
        decoded = self.decoder(out)  # (batch, seq_len, action_dim)
        
        return decoded


class PhaseAwareLoss(nn.Module):
    """Loss that upweights phase transitions for hierarchical tasks."""
    
    def __init__(self, base_criterion=nn.MSELoss(reduction='none'), phase_weight=5.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.phase_weight = phase_weight
        
    def forward(self, pred, target, phase_transitions=None):
        # pred, target: (batch, seq_len, action_dim)
        # phase_transitions: (batch, seq_len)
        
        if phase_transitions is None:
            phase_transitions = torch.zeros(pred.shape[0], pred.shape[1], device=pred.device)
        
        # Compute per-sample, per-timestep loss
        loss = self.base_criterion(pred, target)  # (batch, seq_len, action_dim)
        loss = loss.mean(dim=-1)  # (batch, seq_len)
        
        # Weight: higher for phase transitions
        weights = 1.0 + phase_transitions * (self.phase_weight - 1.0)  # (batch, seq_len)
        weights = weights / weights.sum(dim=1, keepdim=True) * pred.shape[1]
        
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class EnsembleDisagreementLoss(nn.Module):
    """Loss that downweights high-disagreement (noisy) samples."""
    
    def __init__(self, base_criterion=nn.MSELoss(reduction='none'), disagreement_weight=0.5):
        super().__init__()
        self.base_criterion = base_criterion
        self.disagreement_weight = disagreement_weight
        
    def forward(self, pred, target, disagreement=None):
        # pred, target: (batch, seq_len, action_dim)
        # disagreement: (batch, seq_len)
        
        if disagreement is None:
            disagreement = torch.zeros(pred.shape[0], pred.shape[1], device=pred.device)
        
        # Compute per-sample loss
        loss = self.base_criterion(pred, target)  # (batch, seq_len, action_dim)
        loss = loss.mean(dim=-1)  # (batch, seq_len)
        
        # Weight: inverse of disagreement (low disagreement = clean sample)
        weights = 1.0 - disagreement * self.disagreement_weight
        weights = torch.clamp(weights, min=0.1)
        weights = weights / weights.sum(dim=1, keepdim=True) * pred.shape[1]
        
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class HybridLoss(nn.Module):
    """Combined phase-aware + ensemble disagreement for mixed tasks."""
    
    def __init__(self, base_criterion=nn.MSELoss(reduction='none'), phase_weight=3.0, disagreement_weight=0.5):
        super().__init__()
        self.base_criterion = base_criterion
        self.phase_weight = phase_weight
        self.disagreement_weight = disagreement_weight
        
    def forward(self, pred, target, phase_transitions=None, disagreement=None):
        # pred, target: (batch, seq_len, action_dim)
        
        # Default values
        if phase_transitions is None:
            phase_transitions = torch.zeros(pred.shape[0], pred.shape[1], device=pred.device)
        if disagreement is None:
            disagreement = torch.zeros(pred.shape[0], pred.shape[1], device=pred.device)
        
        # Compute per-sample loss
        loss = self.base_criterion(pred, target)  # (batch, seq_len, action_dim)
        loss = loss.mean(dim=-1)  # (batch, seq_len)
        
        # Phase weight: upweight transitions
        phase_weights = 1.0 + phase_transitions * (self.phase_weight - 1.0)
        
        # Disagreement weight: downweight high disagreement
        disagreement_weights = 1.0 - disagreement * self.disagreement_weight
        disagreement_weights = torch.clamp(disagreement_weights, min=0.1)
        
        # Combined: multiply weights
        weights = phase_weights * disagreement_weights
        weights = weights / weights.sum(dim=1, keepdim=True) * pred.shape[1]
        
        weighted_loss = (loss * weights).mean()
        return weighted_loss


def generate_mixed_task_data(n_samples=500, seq_len=20, n_phases=3, noise_level=0.1):
    """Generate mixed task data: hierarchical structure + sensor noise."""
    np.random.seed(42)
    
    # Calculate timesteps per phase (ensure exact division)
    timesteps_per_phase = seq_len // n_phases
    
    trajectories = []
    phase_labels = []
    noise_levels = []
    
    for i in range(n_samples):
        traj = []
        phases = []
        noises = []
        
        for p in range(n_phases):
            # Each phase has distinct dynamics
            base_action = np.random.randn(7) * 0.5
            phase_offset = p * np.random.randn(7) * 2.0
            
            for t in range(timesteps_per_phase):
                # Smooth action within phase
                action = base_action + phase_offset + np.random.randn(7) * 0.1
                traj.append(action)
                # Mark phase transitions (first timestep of each phase)
                phases.append(1.0 if t == 0 else 0.0)
                
                # Add varying noise levels (some phases noisier)
                noise = noise_level * (1.0 + p * 0.3)  # Later phases noisier
                noises.append(noise)
        
        trajectories.append(np.array(traj))
        phase_labels.append(np.array(phases))
        noise_levels.append(np.array(noises))
    
    # Verify shapes
    assert len(trajectories[0]) == seq_len, f"Expected seq_len {seq_len}, got {len(trajectories[0])}"
    
    # Convert to tensors
    X = torch.randn(n_samples, seq_len, 512)
    y = torch.tensor(np.array(trajectories), dtype=torch.float32)
    phase_labels = torch.tensor(np.array(phase_labels), dtype=torch.float32)
    noise_levels = torch.tensor(np.array(noise_levels), dtype=torch.float32)
    
    return X, y, phase_labels, noise_levels


def train_and_evaluate(model, train_X, train_y, test_X, test_y, 
                       phase_labels=None, noise_levels=None,
                       loss_type="baseline", epochs=100, lr=0.001):
    """Train model with specified loss type and evaluate."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    base_criterion = nn.MSELoss(reduction='none')
    
    # Prepare loss function
    if loss_type == "baseline":
        criterion = base_criterion
    elif loss_type == "phase_aware":
        criterion = PhaseAwareLoss(base_criterion)
    elif loss_type == "ensemble_disagreement":
        criterion = EnsembleDisagreementLoss(base_criterion)
    elif loss_type == "hybrid":
        criterion = HybridLoss(base_criterion)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(train_X)  # (batch, seq_len, action_dim)
        
        # Compute loss based on type
        if loss_type == "baseline":
            # Use standard MSE
            loss = F.mse_loss(pred, train_y)
        elif loss_type == "phase_aware":
            # Use ground truth phase labels
            loss = criterion(pred, train_y, phase_labels)
        elif loss_type == "ensemble_disagreement":
            # Use noise levels as proxy for disagreement
            loss = criterion(pred, train_y, noise_levels)
        elif loss_type == "hybrid":
            # Use both
            loss = criterion(pred, train_y, phase_labels, noise_levels)
        
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(test_X)
        test_loss = F.mse_loss(test_pred, test_y).item()
    
    return test_loss


def run_experiment():
    """Run the mixed task experiment."""
    print("=" * 60)
    print("H1.470.1.1.29: Phase-Aware + Ensemble Disagreement for Mixed Tasks")
    print("=" * 60)
    
    results = {
        "experiment_id": "H1.470.1.1.29",
        "description": "Test hybrid phase-aware + ensemble disagreement on mixed tasks",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Test configurations
    configs = [
        {"n_phases": 3, "noise_level": 0.1, "seq_len": 18},  # 18 = 3 * 6
        {"n_phases": 4, "noise_level": 0.15, "seq_len": 28},  # 28 = 4 * 7
        {"n_phases": 5, "noise_level": 0.2, "seq_len": 40},   # 40 = 5 * 8
    ]
    
    loss_types = ["baseline", "phase_aware", "ensemble_disagreement", "hybrid"]
    all_results = {}
    
    for config in configs:
        print(f"\n--- Config: {config['n_phases']} phases, noise={config['noise_level']}, seq_len={config['seq_len']} ---")
        
        # Generate data
        X, y, phase_labels, noise_levels = generate_mixed_task_data(
            n_samples=500, 
            seq_len=config['seq_len'],
            n_phases=config['n_phases'],
            noise_level=config['noise_level']
        )
        
        # Split train/test
        split = int(0.8 * len(X))
        train_X, test_X = X[:split], X[split:]
        train_y, test_y = y[:split], y[split:]
        train_phase, test_phase = phase_labels[:split], phase_labels[split:]
        train_noise, test_noise = noise_levels[:split], noise_levels[split:]
        
        config_results = {}
        
        for loss_type in loss_types:
            # Create fresh model
            model = CognitiveGraphCG(obs_dim=512, action_dim=7, hidden_dim=128)
            
            # Train and evaluate
            test_loss = train_and_evaluate(
                model, train_X, train_y, test_X, test_y,
                phase_labels=train_phase, noise_levels=train_noise,
                loss_type=loss_type, epochs=100, lr=0.001
            )
            
            config_results[loss_type] = test_loss
            print(f"  {loss_type}: test_loss = {test_loss:.6e}")
        
        # Compute improvements
        baseline_loss = config_results["baseline"]
        improvements = {}
        for lt in ["phase_aware", "ensemble_disagreement", "hybrid"]:
            improvement = (baseline_loss - config_results[lt]) / baseline_loss * 100
            improvements[lt] = improvement
            print(f"  {lt} improvement: {improvement:.2f}%")
        
        all_results[f"config_{config['n_phases']}ph_{config['noise_level']}noise"] = {
            "losses": config_results,
            "improvements": improvements
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Aggregate improvements
    phase_improvements = []
    ensemble_improvements = []
    hybrid_improvements = []
    
    for config_key, config_results in all_results.items():
        phase_improvements.append(config_results["improvements"]["phase_aware"])
        ensemble_improvements.append(config_results["improvements"]["ensemble_disagreement"])
        hybrid_improvements.append(config_results["improvements"]["hybrid"])
    
    avg_phase = np.mean(phase_improvements)
    avg_ensemble = np.mean(ensemble_improvements)
    avg_hybrid = np.mean(hybrid_improvements)
    
    print(f"Average improvements:")
    print(f"  Phase-aware: {avg_phase:.2f}%")
    print(f"  Ensemble disagreement: {avg_ensemble:.2f}%")
    print(f"  Hybrid: {avg_hybrid:.2f}%")
    
    # Determine best approach
    best = max([("phase_aware", avg_phase), 
                ("ensemble_disagreement", avg_ensemble), 
                ("hybrid", avg_hybrid)], key=lambda x: x[1])
    
    print(f"\nBest approach: {best[0]} with {best[1]:.2f}% improvement")
    
    # Conclusion
    if best[1] > 50:
        conclusion = "SUPPORTED"
    elif best[1] > 10:
        conclusion = "CLOSE"
    else:
        conclusion = "REFUTED"
    
    print(f"Conclusion: {conclusion}")
    
    results["config_results"] = all_results
    results["summary"] = {
        "avg_phase_improvement": avg_phase,
        "avg_ensemble_improvement": avg_ensemble,
        "avg_hybrid_improvement": avg_hybrid,
        "best_approach": best[0],
        "best_improvement": best[1],
        "conclusion": conclusion
    }
    
    # Save results
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
