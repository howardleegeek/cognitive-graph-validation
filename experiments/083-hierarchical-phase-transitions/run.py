#!/usr/bin/env python3
"""
H1.470.1.1.26: Ensemble Disagreement on Hierarchical Multi-Step Tasks with Phase Transitions

Prior results:
- H1.470.1.1.25: INCONCLUSIVE - 0.76% improvement on simple multi-step tasks
- H1.470.1.1.24: SUPPORTED - +15.24% improvement on real robot data

Hypothesis: Ensemble disagreement noise estimation will perform better on hierarchical 
multi-step tasks with phase transitions (e.g., pick -> carry -> place) because:
1. Phase transitions create natural decision boundaries
2. Different phases have different noise characteristics
3. Ensemble disagreement can capture uncertainty at phase boundaries
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
    
    def __init__(self, obs_dim=512, action_dim=7, hidden_dim=128):
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
        # Unified processor - simplified GRU
        self.graph_processor = nn.GRU(512, hidden_dim, num_layers=1, batch_first=True)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        # x: [obs_dim] or [batch, obs_dim]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        unified = torch.cat([physical, semantic], dim=-1)
        
        # Handle both single sample and batch
        if len(unified.shape) == 1:
            unified = unified.unsqueeze(0).unsqueeze(0)  # [1, 1, 512]
        elif len(unified.shape) == 2:
            unified = unified.unsqueeze(1)  # [batch, 1, 512]
        
        out, _ = self.graph_processor(unified)
        return self.decoder(out[:, -1, :])  # [batch, action_dim]


class EnsembleDisagreementEstimator:
    """Estimates noise level using ensemble disagreement (variance across models)."""
    
    def __init__(self, n_models=5, obs_dim=512, action_dim=7, hidden_dim=128):
        self.n_models = n_models
        self.models = nn.ModuleList([
            CognitiveGraphCG(obs_dim, action_dim, hidden_dim) 
            for _ in range(n_models)
        ])
        
    def predict_with_uncertainty(self, x):
        """Get predictions from all models and compute disagreement."""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_models, batch, action_dim]
        mean_pred = predictions.mean(dim=0)
        disagreement = predictions.var(dim=0).mean(dim=-1)  # [batch]
        
        return mean_pred, disagreement


def generate_hierarchical_multi_step_data(n_samples=200, seq_len=20, n_phases=3):
    """
    Generate hierarchical multi-step task data with phase transitions.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    data = []
    
    for i in range(n_samples):
        observations = []
        actions = []
        noise_levels = []
        
        phase_len = seq_len // n_phases
        
        for t in range(seq_len):
            # Determine current phase
            if t < phase_len:
                phase = "approach"
                phase_noise = 0.05
            elif t < 2 * phase_len:
                phase = "grasp"
                phase_noise = 0.15
            else:
                phase = "transport"
                phase_noise = 0.08
            
            obs = np.random.randn(512).astype(np.float32)
            obs[:128] += np.sin(t * 0.5) * 0.5
            
            action = np.random.randn(7).astype(np.float32) * 0.5
            action[0] = t / seq_len
            
            noise = np.random.randn(7).astype(np.float32) * phase_noise
            action_noisy = action + noise
            
            observations.append(obs)
            actions.append(action_noisy)
            noise_levels.append(phase_noise)
        
        data.append({
            'observations': np.array(observations),
            'actions': np.array(actions),
            'noise_levels': np.array(noise_levels),
        })
    
    return data


def train_baseline(train_data, test_data, epochs=20):
    """Train baseline model without noise awareness."""
    model = CognitiveGraphCG(512, 7, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_obs = torch.tensor(np.array([d['observations'] for d in train_data]), dtype=torch.float32)
    train_actions = torch.tensor(np.array([d['actions'] for d in train_data]), dtype=torch.float32)
    test_obs = torch.tensor(np.array([d['observations'] for d in test_data]), dtype=torch.float32)
    test_actions = torch.tensor(np.array([d['actions'] for d in test_data]), dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        for i in range(len(train_obs)):
            obs = train_obs[i]
            actions = train_actions[i]
            
            optimizer.zero_grad()
            
            # Process sequence and get final prediction
            obs_seq = obs  # [seq_len, obs_dim]
            hidden = None
            
            for t in range(len(obs)):
                pred = model(obs_seq[t])
            
            loss = criterion(pred, actions[-1])
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = []
        for obs in test_obs:
            for t in range(len(obs)):
                pred = model(obs[t])
            test_preds.append(pred)
        
        test_preds = torch.stack(test_preds)
        test_loss = criterion(test_preds, test_actions[:, -1, :]).item()
    
    return model, test_loss


def train_oracle_noise(train_data, test_data, epochs=20):
    """Train with oracle noise levels (upper bound)."""
    model = CognitiveGraphCG(512, 7, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')
    
    train_obs = torch.tensor(np.array([d['observations'] for d in train_data]), dtype=torch.float32)
    train_actions = torch.tensor(np.array([d['actions'] for d in train_data]), dtype=torch.float32)
    train_noise = torch.tensor(np.array([d['noise_levels'] for d in train_data]), dtype=torch.float32)
    test_obs = torch.tensor(np.array([d['observations'] for d in test_data]), dtype=torch.float32)
    test_actions = torch.tensor(np.array([d['actions'] for d in test_data]), dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        for i in range(len(train_obs)):
            obs = train_obs[i]
            actions = train_actions[i]
            noise = train_noise[i]
            
            optimizer.zero_grad()
            
            # Process sequence
            for t in range(len(obs)):
                pred = model(obs[t])
            
            # Weight by inverse noise
            weight = 1.0 / (1.0 + noise[-1].item() * 10)
            
            loss = criterion(pred, actions[-1]).mean() * weight
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = []
        for obs in test_obs:
            for t in range(len(obs)):
                pred = model(obs[t])
            test_preds.append(pred)
        
        test_preds = torch.stack(test_preds)
        criterion_mean = nn.MSELoss()
        test_loss = criterion_mean(test_preds, test_actions[:, -1, :]).item()
    
    return model, test_loss


def train_ensemble_disagreement(train_data, test_data, epochs=20):
    """Train with ensemble disagreement noise estimation."""
    ensemble = EnsembleDisagreementEstimator(n_models=5, obs_dim=512, action_dim=7, hidden_dim=128)
    optimizer = torch.optim.Adam(ensemble.models.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')
    
    train_obs = torch.tensor(np.array([d['observations'] for d in train_data]), dtype=torch.float32)
    train_actions = torch.tensor(np.array([d['actions'] for d in train_data]), dtype=torch.float32)
    test_obs = torch.tensor(np.array([d['observations'] for d in test_data]), dtype=torch.float32)
    test_actions = torch.tensor(np.array([d['actions'] for d in test_data]), dtype=torch.float32)
    
    for epoch in range(epochs):
        ensemble.models.train()
        for i in range(len(train_obs)):
            obs = train_obs[i]
            actions = train_actions[i]
            
            optimizer.zero_grad()
            
            # Get ensemble predictions for final timestep
            predictions = []
            for model in ensemble.models:
                model.train()
                for t in range(len(obs)):
                    pred = model(obs[t])
                predictions.append(pred)
            
            predictions = torch.stack(predictions)  # [n_models, action_dim]
            mean_pred = predictions.mean(dim=0)
            disagreement = predictions.var(dim=0).mean()  # scalar
            
            # Weight by inverse disagreement
            weight = 1.0 / (1.0 + disagreement.item() * 10)
            
            loss = criterion(mean_pred.unsqueeze(0), actions[-1].unsqueeze(0)).mean()
            weighted_loss = loss * weight
            weighted_loss.backward()
            optimizer.step()
    
    # Evaluate
    ensemble.models.eval()
    with torch.no_grad():
        test_preds = []
        for obs in test_obs:
            predictions = []
            for model in ensemble.models:
                for t in range(len(obs)):
                    pred = model(obs[t])
                predictions.append(pred)
            
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            test_preds.append(mean_pred)
        
        test_preds = torch.stack(test_preds)
        criterion_mean = nn.MSELoss()
        test_loss = criterion_mean(test_preds, test_actions[:, -1, :]).item()
    
    return ensemble, test_loss


def main():
    print("=" * 60)
    print("H1.470.1.1.26: Ensemble Disagreement on Hierarchical Multi-Step Tasks")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating hierarchical multi-step task data...")
    train_data = generate_hierarchical_multi_step_data(n_samples=200, seq_len=20, n_phases=3)
    test_data = generate_hierarchical_multi_step_data(n_samples=50, seq_len=20, n_phases=3)
    print(f"    Generated {len(train_data)} train, {len(test_data)} test trajectories")
    print(f"    Sequence length: 20, Phases: 3 (approach, grasp, transport)")
    
    # Train and evaluate baseline
    print("\n[2] Training baseline model...")
    baseline_model, baseline_loss = train_baseline(train_data, test_data, epochs=20)
    print(f"    Baseline test loss: {baseline_loss:.6f}")
    
    # Train and evaluate oracle noise
    print("\n[3] Training oracle noise model...")
    oracle_model, oracle_loss = train_oracle_noise(train_data, test_data, epochs=20)
    print(f"    Oracle noise test loss: {oracle_loss:.6f}")
    
    # Train and evaluate ensemble disagreement
    print("\n[4] Training ensemble disagreement model...")
    ensemble, ensemble_loss = train_ensemble_disagreement(train_data, test_data, epochs=20)
    print(f"    Ensemble disagreement test loss: {ensemble_loss:.6f}")
    
    # Calculate improvements
    baseline_to_oracle_improvement = (baseline_loss - oracle_loss) / baseline_loss * 100
    baseline_to_ensemble_improvement = (baseline_loss - ensemble_loss) / baseline_loss * 100
    
    # Oracle ratio
    if baseline_to_oracle_improvement > 0:
        oracle_ratio = baseline_to_ensemble_improvement / baseline_to_oracle_improvement * 100
    else:
        oracle_ratio = float('inf') if baseline_to_ensemble_improvement > 0 else 0
    
    # Determine conclusion
    if baseline_to_ensemble_improvement > 5:
        conclusion = "SUPPORTED"
    elif baseline_to_ensemble_improvement > 0:
        conclusion = "INCONCLUSIVE"
    else:
        conclusion = "REFUTED"
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n| Strategy | Test Loss | Improvement | Oracle Ratio |")
    print(f"|----------|-----------|-------------|--------------|")
    print(f"| Baseline | {baseline_loss:.6f} | +0.00% | N/A |")
    print(f"| Oracle noise | {oracle_loss:.6f} | +{baseline_to_oracle_improvement:.2f}% | 100% |")
    print(f"| Ensemble disagreement | {ensemble_loss:.6f} | +{baseline_to_ensemble_improvement:.2f}% | {oracle_ratio:.1f}% |")
    print(f"\nConclusion: {conclusion}")
    
    # Save results
    results = {
        "experiment_id": "H1.470.1.1.26",
        "description": "Test ensemble disagreement on hierarchical multi-step tasks with phase transitions",
        "result": {
            "conclusion": conclusion,
            "task": "hierarchical_multi_step_phase_transitions",
            "configurations_tested": 3,
            "architectures_tested": ["baseline", "oracle_noise", "ensemble_disagreement"],
            "sequence_lengths_tested": [20],
            "n_phases": 3,
            "key_metrics": {
                "baseline_test_loss": float(baseline_loss),
                "oracle_test_loss": float(oracle_loss),
                "oracle_improvement": float(baseline_to_oracle_improvement),
                "ensemble_disagreement_loss": float(ensemble_loss),
                "ensemble_disagreement_improvement": float(baseline_to_ensemble_improvement),
                "oracle_ratio": float(oracle_ratio),
            },
            "key_insights": [
                f"Phase transitions (approach/grasp/transport) create natural decision boundaries",
                f"Ensemble disagreement provides {baseline_to_ensemble_improvement:.2f}% improvement over baseline",
                f"Oracle ratio: {oracle_ratio:.1f}% (ensemble vs oracle performance)",
            ],
            "recommendations": [
                "R1: Test on tasks with more phases (4-5)",
                "R2: Test on tasks with noisy phase transitions",
                "R3: Consider adaptive phase detection"
            ]
        }
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
    return results


if __name__ == "__main__":
    main()
