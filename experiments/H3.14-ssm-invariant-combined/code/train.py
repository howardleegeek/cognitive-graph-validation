#!/usr/bin/env python3
"""
H3.14: SSM + Invariant Learning Combined
Tests if SSM combined with invariant learning achieves both:
1. +93% on long sequences (from H3.8)
2. Solves cross-dynamics transfer (from H1.8)

This is the ultimate test - can one architecture solve both problems?
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class SSMInvariantModel(nn.Module):
    """SSM with invariant representation learning.
    
    Combines:
    - SSM (Mamba-style) for long-range temporal dependencies
    - Bisimulation loss for dynamics-invariant representations
    """
    def __init__(self, obs_dim=64, action_dim=8, hidden=256, z_dim=128, n_layers=4):
        super().__init__()
        self.z_dim = z_dim
        self.n_layers = n_layers
        
        # SSM-style state space layers
        self.ssm_layers = nn.ModuleList([
            nn.Linear(z_dim, z_dim * 2) for _ in range(n_layers)
        ])
        self.ssm_gates = nn.ModuleList([
            nn.Linear(z_dim, z_dim) for _ in range(n_layers)
        ])
        
        # Invariant encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        
        # Dynamics predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(z_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim)
        )
    
    def ssm_step(self, z, action_z):
        """SSM-style processing with gating."""
        for i in range(self.n_layers):
            # Combine state and action
            combined = z + action_z
            
            # Gated transformation
            gate = torch.sigmoid(self.ssm_gates[i](combined))
            transformed = self.ssm_layers[i](combined)
            
            # Split into candidate and gate
            candidate = transformed[:, :self.z_dim]
            z = gate * z + (1 - gate) * candidate
        
        return z
    
    def encode(self, obs):
        return self.encoder(obs)
    
    def forward(self, obs, action):
        z = self.encode(obs)
        action_z = self.action_encoder(action)
        z_next = self.ssm_step(z, action_z)
        obs_pred = self.decoder(z_next)
        return obs_pred
    
    def get_latent(self, obs, action):
        z = self.encode(obs)
        action_z = self.action_encoder(action)
        z_next = self.ssm_step(z, action_z)
        return z_next


class BaselineModel(nn.Module):
    """Baseline concatenation model."""
    def __init__(self, obs_dim=64, action_dim=8, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim)
        )
    
    def forward(self, obs, action):
        obs_enc = self.encoder(obs)
        action_enc = self.action_encoder(action)
        pred = self.predictor(torch.cat([obs_enc, action_enc], dim=-1))
        return pred


class SSMOnlyModel(nn.Module):
    """SSM without invariant learning (for comparison)."""
    def __init__(self, obs_dim=64, action_dim=8, hidden=256, z_dim=128, n_layers=4):
        super().__init__()
        self.z_dim = z_dim
        self.n_layers = n_layers
        
        self.ssm_layers = nn.ModuleList([
            nn.Linear(z_dim, z_dim * 2) for _ in range(n_layers)
        ])
        self.ssm_gates = nn.ModuleList([
            nn.Linear(z_dim, z_dim) for _ in range(n_layers)
        ])
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim)
        )
    
    def forward(self, obs, action):
        z = self.encoder(obs)
        action_z = self.action_encoder(action)
        
        for i in range(self.n_layers):
            combined = z + action_z
            gate = torch.sigmoid(self.ssm_gates[i](combined))
            transformed = self.ssm_layers[i](combined)
            candidate = transformed[:, :self.z_dim]
            z = gate * z + (1 - gate) * candidate
        
        return self.decoder(z)


def generate_trajectory_data(n_samples, seq_len, dynamics_params, seed=42):
    """Generate sequential data with specific dynamics."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    obs_dim = 64
    action_dim = 8
    friction, mass, damping = dynamics_params
    
    trajectories = []
    for _ in range(n_samples):
        obs = np.random.randn(obs_dim) * 0.5
        traj = [obs]
        
        for _ in range(seq_len):
            action = np.random.randn(action_dim) * 0.2
            next_obs = obs + friction * np.mean(action) * 0.1 + np.random.randn(obs_dim) * (mass * 0.1)
            next_obs = next_obs * (1 - damping * 0.01)
            traj.append(next_obs)
            obs = next_obs
        
        trajectories.append(traj)
    
    return trajectories


def bisimulation_loss(z1, z2, a1, a2, alpha=0.5):
    """Bisimulation loss for invariant representations."""
    diff_z = torch.abs(z1 - z2).mean()
    diff_a = torch.abs(a1 - a2).mean()
    return alpha * diff_z - (1 - alpha) * diff_a


def train_ssm_invariant(model, train_data, epochs=50, lr=1e-3):
    """Train SSM + Invariant with bisimulation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for traj in train_data[:50]:  # Use subset for speed
            for t in range(len(traj) - 1):
                obs = torch.FloatTensor(traj[t]).unsqueeze(0)
                action = torch.FloatTensor(np.random.randn(8) * 0.2).unsqueeze(0)
                target = torch.FloatTensor(traj[t + 1]).unsqueeze(0)
                
                optimizer.zero_grad()
                pred = model(obs, action)
                loss = criterion(pred, target)
                
                # Add bisimulation loss for invariance
                z = model.encode(obs)
                action_z = model.action_encoder(action)
                
                # Create negative sample from different dynamics
                neg_obs = torch.FloatTensor(np.random.randn(64) * 0.5).unsqueeze(0)
                z_neg = model.encode(neg_obs)
                
                bisim = bisimulation_loss(z, z_neg, action_z, action_z, alpha=0.3)
                total_loss = loss + 0.1 * bisim
                
                total_loss.backward()
                optimizer.step()
    
    return loss.item()


def train_ssm_only(model, train_data, epochs=50, lr=1e-3):
    """Train SSM without invariant loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for traj in train_data[:50]:
            for t in range(len(traj) - 1):
                obs = torch.FloatTensor(traj[t]).unsqueeze(0)
                action = torch.FloatTensor(np.random.randn(8) * 0.2).unsqueeze(0)
                target = torch.FloatTensor(traj[t + 1]).unsqueeze(0)
                
                optimizer.zero_grad()
                pred = model(obs, action)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
    
    return loss.item()


def train_baseline(model, train_data, epochs=50, lr=1e-3):
    """Train baseline."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for traj in train_data[:50]:
            for t in range(len(traj) - 1):
                obs = torch.FloatTensor(traj[t]).unsqueeze(0)
                action = torch.FloatTensor(np.random.randn(8) * 0.2).unsqueeze(0)
                target = torch.FloatTensor(traj[t + 1]).unsqueeze(0)
                
                optimizer.zero_grad()
                pred = model(obs, action)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
    
    return loss.item()


def evaluate(model, test_data):
    """Evaluate model."""
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    with torch.no_grad():
        for traj in test_data:
            for t in range(len(traj) - 1):
                obs = torch.FloatTensor(traj[t]).unsqueeze(0)
                action = torch.FloatTensor(np.random.randn(8) * 0.2).unsqueeze(0)
                target = torch.FloatTensor(traj[t + 1]).unsqueeze(0)
                pred = model(obs, action)
                losses.append(criterion(pred, target).item())
    return np.mean(losses)


def run_experiment():
    """Run H3.14: SSM + Invariant Combined experiment."""
    print("=" * 70)
    print("H3.14: SSM + Invariant Learning Combined")
    print("=" * 70)
    
    # Test 1: Long sequence performance (from H3.8)
    print("\n=== Test 1: Long Sequence Performance ===")
    source_dynamics = (0.5, 1.0, 0.1)
    long_seq_data = generate_trajectory_data(100, 30, source_dynamics, seed=42)
    test_long = generate_trajectory_data(50, 30, source_dynamics, seed=999)
    
    # Baseline
    baseline = BaselineModel(obs_dim=64, action_dim=8)
    train_baseline(baseline, long_seq_data, epochs=30)
    baseline_long_loss = evaluate(baseline, test_long)
    print(f"Baseline (30-step): {baseline_long_loss:.4f}")
    
    # SSM only
    ssm_only = SSMOnlyModel(obs_dim=64, action_dim=8, z_dim=128, n_layers=4)
    train_ssm_only(ssm_only, long_seq_data, epochs=30)
    ssm_long_loss = evaluate(ssm_only, test_long)
    print(f"SSM only (30-step): {ssm_long_loss:.4f}")
    
    # SSM + Invariant
    ssm_invariant = SSMInvariantModel(obs_dim=64, action_dim=8, z_dim=128, n_layers=4)
    train_ssm_invariant(ssm_invariant, long_seq_data, epochs=30)
    ssm_inv_long_loss = evaluate(ssm_invariant, test_long)
    print(f"SSM + Invariant (30-step): {ssm_inv_long_loss:.4f}")
    
    long_improvement_ssm = (baseline_long_loss - ssm_long_loss) / baseline_long_loss * 100
    long_improvement_combined = (baseline_long_loss - ssm_inv_long_loss) / baseline_long_loss * 100
    print(f"SSM improvement: {long_improvement_ssm:+.1f}%")
    print(f"SSM+Inv improvement: {long_improvement_combined:+.1f}%")
    
    # Test 2: Cross-dynamics transfer (from H1.8)
    print("\n=== Test 2: Cross-Dynamics Transfer ===")
    source_dynamics = (0.5, 1.0, 0.1)
    target_dynamics = (0.8, 1.5, 0.2)
    
    train_source = generate_trajectory_data(100, 10, source_dynamics, seed=42)
    train_target = generate_trajectory_data(100, 10, target_dynamics, seed=43)
    test_target = generate_trajectory_data(50, 10, target_dynamics, seed=999)
    
    # Baseline transfer
    baseline_transfer = BaselineModel(obs_dim=64, action_dim=8)
    train_baseline(baseline_transfer, train_source, epochs=30)
    baseline_transfer_loss = evaluate(baseline_transfer, test_target)
    print(f"Baseline transfer: {baseline_transfer_loss:.4f}")
    
    # SSM only transfer
    ssm_only_transfer = SSMOnlyModel(obs_dim=64, action_dim=8, z_dim=128, n_layers=4)
    train_ssm_only(ssm_only_transfer, train_source, epochs=30)
    ssm_transfer_loss = evaluate(ssm_only_transfer, test_target)
    print(f"SSM only transfer: {ssm_transfer_loss:.4f}")
    
    # SSM + Invariant transfer
    ssm_inv_transfer = SSMInvariantModel(obs_dim=64, action_dim=8, z_dim=128, n_layers=4)
    train_ssm_invariant(ssm_inv_transfer, train_source, epochs=30)
    ssm_inv_transfer_loss = evaluate(ssm_inv_transfer, test_target)
    print(f"SSM + Invariant transfer: {ssm_inv_transfer_loss:.4f}")
    
    transfer_improvement_ssm = (baseline_transfer_loss - ssm_transfer_loss) / baseline_transfer_loss * 100
    transfer_improvement_combined = (baseline_transfer_loss - ssm_inv_transfer_loss) / baseline_transfer_loss * 100
    print(f"SSM transfer improvement: {transfer_improvement_ssm:+.1f}%")
    print(f"SSM+Inv transfer improvement: {transfer_improvement_combined:+.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("H3.14 Results Summary")
    print("=" * 70)
    print(f"Long Sequence (30-step):")
    print(f"  - Baseline: {baseline_long_loss:.4f}")
    print(f"  - SSM only: {ssm_long_loss:.4f} ({long_improvement_ssm:+.1f}%)")
    print(f"  - SSM+Inv: {ssm_inv_long_loss:.4f} ({long_improvement_combined:+.1f}%)")
    print(f"\nCross-Dynamics Transfer:")
    print(f"  - Baseline: {baseline_transfer_loss:.4f}")
    print(f"  - SSM only: {ssm_transfer_loss:.4f} ({transfer_improvement_ssm:+.1f}%)")
    print(f"  - SSM+Inv: {ssm_inv_transfer_loss:.4f} ({transfer_improvement_combined:+.1f}%)")
    
    # Determine status
    long_supported = bool(ssm_long_loss < baseline_long_loss)
    combined_long_supported = bool(ssm_inv_long_loss < baseline_long_loss)
    transfer_supported = bool(ssm_inv_transfer_loss < baseline_transfer_loss)
    
    status = "SUPPORTED" if (long_supported and combined_long_supported and transfer_supported) else "PARTIAL"
    
    result = {
        "hypothesis": "H3.14",
        "experiment": "SSM + Invariant Combined",
        "status": status,
        "long_sequence": {
            "baseline": float(baseline_long_loss),
            "ssm_only": float(ssm_long_loss),
            "ssm_invariant": float(ssm_inv_long_loss),
            "ssm_improvement": float(long_improvement_ssm),
            "combined_improvement": float(long_improvement_combined)
        },
        "transfer": {
            "baseline": float(baseline_transfer_loss),
            "ssm_only": float(ssm_transfer_loss),
            "ssm_invariant": float(ssm_inv_transfer_loss),
            "ssm_improvement": float(transfer_improvement_ssm),
            "combined_improvement": float(transfer_improvement_combined)
        },
        "long_supported": long_supported,
        "transfer_supported": transfer_supported
    }
    
    print("\n" + json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()