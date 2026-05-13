#!/usr/bin/env python3
"""
H1.226: Unified Architecture on Complex Multi-Step Tasks WITH Autocorrelation
Based on H1 success (+25.6%) and H3's autocorrelation discovery (rho >= 0.93 enables attention)

Hypothesis: Unified architecture with autocorrelation injection will outperform on complex multi-step tasks
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

def generate_trajectory_data(n_samples, seq_len, n_steps, autocorrelation=0.95):
    """Generate trajectory data with specified autocorrelation."""
    # Generate base trajectory with autocorrelation
    t = np.linspace(0, 4 * np.pi, seq_len)
    base_freq = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2 * np.pi)
    
    trajectories = []
    for _ in range(n_samples):
        # Create smooth trajectory with autocorrelation
        x = np.sin(base_freq * t + phase)
        # Add autocorrelation
        x = autocorrelation * x + (1 - autocorrelation) * np.random.randn(seq_len) * 0.1
        
        # Multi-step: multiple waypoints
        waypoints = []
        for step in range(n_steps):
            wp_idx = (step + 1) * seq_len // (n_steps + 1) - 1
            waypoints.append([x[wp_idx], np.cos(wp_idx / seq_len * np.pi)])
        waypoints = np.array(waypoints).flatten()
        
        # Observation: current position + goal
        obs = np.concatenate([x[:5], waypoints[:4]])
        # Action: next position
        action = x[1]
        
        trajectories.append((obs, action))
    
    return trajectories

class UnifiedArchitecture(nn.Module):
    """Unified cognitive graph architecture."""
    def __init__(self, obs_dim=9, action_dim=1, total_dim=512):
        super().__init__()
        physical_dim = int(total_dim * 0.22)  # 22% optimal from H4
        semantic_dim = total_dim - physical_dim
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs, goal):
        z_phys = self.obs_encoder(obs)
        z_sem = self.goal_encoder(goal)
        z = torch.cat([z_phys, z_sem], dim=-1)
        return self.fusion(z)

class BaselineArchitecture(nn.Module):
    """Baseline: separate encoders with late fusion."""
    def __init__(self, obs_dim=9, action_dim=1, latent_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.goal_encoder = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, action_dim))
        
    def forward(self, obs, goal):
        return self.decoder(torch.cat([self.obs_encoder(obs), self.goal_encoder(goal)], dim=-1))

def train_and_evaluate(model, train_data, val_data, epochs=100):
    """Train and evaluate model."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    
    train_obs = torch.FloatTensor([d[0] for d in train_data])
    train_goal = torch.FloatTensor([d[0][-4:] for d in train_data])
    train_act = torch.FloatTensor([d[1] for d in train_data]).unsqueeze(-1)
    
    val_obs = torch.FloatTensor([d[0] for d in val_data])
    val_goal = torch.FloatTensor([d[0][-4:] for d in val_data])
    val_act = torch.FloatTensor([d[1] for d in val_data]).unsqueeze(-1)
    
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(train_obs))
        for i in range(0, len(idx), 32):
            batch_idx = idx[i:i+32]
            pred = model(train_obs[batch_idx], train_goal[batch_idx])
            loss = crit(pred, train_act[batch_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        val_pred = model(val_obs, val_goal)
        val_loss = crit(val_pred, val_act).item()
    
    return val_loss

def run_experiment():
    """Run H1.226 experiment."""
    results = {}
    
    # Test different complexities and autocorrelation levels
    complexities = [3, 5, 7, 10, 15]
    autocorr_levels = [0.85, 0.90, 0.93, 0.95, 0.98]
    
    for n_steps in complexities:
        for rho in autocorr_levels:
            print(f"\nTesting: {n_steps} steps, rho={rho}")
            
            # Generate data
            train_data = generate_trajectory_data(200, 20, n_steps, rho)
            val_data = generate_trajectory_data(50, 20, n_steps, rho)
            
            # Train unified
            unified = UnifiedArchitecture()
            unified_loss = train_and_evaluate(unified, train_data, val_data)
            
            # Train baseline
            baseline = BaselineArchitecture()
            baseline_loss = train_and_evaluate(baseline, train_data, val_data)
            
            improvement = (baseline_loss - unified_loss) / baseline_loss * 100
            
            key = f"steps_{n_steps}_rho_{rho}"
            results[key] = {
                "unified_loss": float(unified_loss),
                "baseline_loss": float(baseline_loss),
                "improvement": float(improvement),
                "unified_wins": unified_loss < baseline_loss
            }
            print(f"  Unified: {unified_loss:.6f}, Baseline: {baseline_loss:.6f}, Δ: {improvement:.1f}%")
    
    # Summary
    wins = sum(1 for r in results.values() if r["unified_wins"])
    avg_improvement = np.mean([r["improvement"] for r in results.values()])
    
    summary = {
        "experiment": "H1.226",
        "hypothesis": "Unified + autocorrelation on complex multi-step",
        "total_tests": len(results),
        "unified_wins": wins,
        "avg_improvement": float(avg_improvement),
        "status": "SUPPORTED" if avg_improvement > 0 else "REFUTED",
        "details": results
    }
    
    print(f"\n{'='*60}")
    print(f"H1.226 Results: {wins}/{len(results)} wins, avg {avg_improvement:.1f}%")
    print(f"Status: {summary['status']}")
    print(f"{'='*60}")
    
    return summary

if __name__ == "__main__":
    result = run_experiment()
    print("\n" + json.dumps(result, indent=2))