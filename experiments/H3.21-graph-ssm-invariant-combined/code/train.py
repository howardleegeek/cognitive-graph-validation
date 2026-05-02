"""
H3.21: Graph + SSM + Invariant Combined Architecture
Tests combined architecture for both temporal reasoning AND transfer
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import json


class SSMProcessor(nn.Module):
    """Mamba-style SSM for long sequences"""
    def __init__(self, input_dim=64, state_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.ssm_A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.ssm_B = nn.Linear(state_dim, state_dim)
        self.ssm_C = nn.Linear(state_dim, state_dim)
        self.gate = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(state_dim, input_dim)
        
    def forward(self, x):
        # x: [batch, seq, features]
        h = torch.zeros(x.shape[0], self.ssm_A.shape[0], device=x.device)
        outputs = []
        
        for t in range(x.shape[1]):
            inp = self.input_proj(x[:, t])
            # SSM update: h = A*h + B*x
            h = torch.matmul(h, self.ssm_A.t()) + self.ssm_B(inp)
            # Gated output
            gate = self.gate(h)
            out = self.ssm_C(h * gate)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)


class InvariantLearner(nn.Module):
    """Bisimulation-inspired invariant learning"""
    def __init__(self, input_dim=64, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, x, dynamics_params=None):
        z = self.encoder(x)
        
        if dynamics_params is not None and self.training:
            # Invariant loss: minimize dependence on dynamics_params
            z_aug = z + torch.randn_like(z) * 0.1
            proj_z = self.projector(z)
            proj_z_aug = self.projector(z_aug)
            
            # Invariant loss (simplified)
            invariant_loss = torch.abs(proj_z - proj_z_aug).mean()
            return z, invariant_loss
        
        return z, None


class CombinedArchitecture(nn.Module):
    """SSM + Invariant Combined (simplified from Graph+SSM+Invariant)"""
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        super().__init__()
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.ssm = SSMProcessor(hidden_dim, hidden_dim)
        self.invariant = InvariantLearner(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index=None, dynamics_params=None):
        # x: [batch, seq, features]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Embed input
        embedded = self.input_embed(x)
        
        # SSM processing
        ssm_out = self.ssm(embedded)
        
        # Invariant learning on last timestep
        invariant_out, invariant_loss = self.invariant(ssm_out[:, -1], dynamics_params)
        
        # Output - repeat for each timestep
        out = self.output(invariant_out)  # [batch, output_dim]
        out = out.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, output_dim]
        
        return out, invariant_loss


def generate_temporal_data(num_samples=200, num_steps=20, num_objects=3):
    """Generate temporal reasoning data - simplified to flat representation"""
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_samples):
        # Flat state representation (64-dim)
        state = torch.randn(64)
        
        episode_states = []
        episode_actions = []
        episode_next = []
        
        for _ in range(num_steps):
            # Random action
            action = torch.randn(4)
            
            # State transition (simplified)
            next_state = state + torch.randn(64) * 0.1 + action[:1] * 0.5
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_next.append(next_state)
            
            state = next_state
        
        states.append(torch.stack(episode_states))
        actions.append(torch.stack(episode_actions))
        next_states.append(torch.stack(episode_next))
    
    return torch.stack(states), torch.stack(actions), torch.stack(next_states)


def generate_transfer_data(num_samples=200, friction_range=(0.05, 0.5), mass_range=(0.5, 2.0)):
    """Generate cross-dynamics transfer data"""
    states = []
    actions = []
    next_states = []
    dynamics = []
    
    for _ in range(num_samples):
        friction = torch.rand(1) * (friction_range[1] - friction_range[0]) + friction_range[0]
        mass = torch.rand(1) * (mass_range[1] - mass_range[0]) + mass_range[0]
        
        state = torch.randn(64)
        
        episode_states = [state]
        episode_actions = []
        episode_next = []
        
        for _ in range(10):
            action = torch.randn(4)
            # Dynamics-dependent transition (simplified)
            action_effect = (action.sum() * friction / mass).item()
            next_state = state + action_effect + torch.randn(64) * 0.01
            episode_states.append(next_state)
            episode_actions.append(action)
            episode_next.append(next_state)
            state = next_state
        
        states.append(torch.stack(episode_states[:-1]))
        actions.append(torch.stack(episode_actions))
        next_states.append(torch.stack(episode_next))
        dynamics.append(torch.cat([friction, mass]))
    
    return torch.stack(states), torch.stack(actions), torch.stack(next_states), torch.stack(dynamics)


def train_and_evaluate():
    """Main training and evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Edge index for graph (fully connected objects)
    edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 0, 2, 2, 0, 1]], dtype=torch.long)
    
    # Generate data
    print("Generating temporal data...")
    temp_states, temp_actions, temp_next = generate_temporal_data(200, 20, 3)
    
    print("Generating transfer data...")
    trans_states, trans_actions, trans_next, trans_dynamics = generate_transfer_data(200)
    
    # Split into train/test
    train_size = 150
    temp_train_states = temp_states[:train_size].to(device)
    temp_train_next = temp_next[:train_size].to(device)
    temp_test_states = temp_states[train_size:].to(device)
    temp_test_next = temp_next[train_size:].to(device)
    
    trans_train_states = trans_states[:train_size].to(device)
    trans_train_next = trans_next[:train_size].to(device)
    trans_train_dynamics = trans_dynamics[:train_size].to(device)
    trans_test_states = trans_states[train_size:].to(device)
    trans_test_next = trans_next[train_size:].to(device)
    trans_test_dynamics = trans_dynamics[train_size:].to(device)
    
    # Train model
    model = CombinedArchitecture().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training combined architecture...")
    losses = []
    
    for epoch in range(200):
        model.train()
        
        # Temporal task
        temp_pred, _ = model(temp_train_states, edge_index.to(device), None)
        temp_loss = criterion(temp_pred, temp_train_next[:, :, :32])
        
        # Transfer task
        trans_pred, inv_loss = model(trans_train_states, edge_index.to(device), trans_train_dynamics.to(device))
        trans_loss = criterion(trans_pred, trans_train_next[:, :, :32])
        
        # Combined loss
        total_loss = temp_loss + trans_loss + (inv_loss if inv_loss is not None else 0)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
    
    # Evaluate
    model.eval()
    
    with torch.no_grad():
        # Temporal evaluation
        temp_pred_test, _ = model(temp_test_states, edge_index.to(device), None)
        temp_mse = criterion(temp_pred_test, temp_test_next[:, :, :32]).item()
        
        # Transfer evaluation
        trans_pred_test, _ = model(trans_test_states, edge_index.to(device), trans_test_dynamics.to(device))
        trans_mse = criterion(trans_pred_test, trans_test_next[:, :, :32]).item()
        
        # Baseline (predict mean of input)
        baseline_temp = temp_test_states.mean(dim=2, keepdim=True).expand(-1, temp_test_states.shape[1], 32)
        baseline_trans = trans_test_states.mean(dim=2, keepdim=True).expand(-1, trans_test_states.shape[1], 32)
        baseline_temp_mse = criterion(baseline_temp, temp_test_next[:, :, :32]).item()
        baseline_trans_mse = criterion(baseline_trans, trans_test_next[:, :, :32]).item()
    
    # Calculate improvements
    temp_improvement = (baseline_temp_mse - temp_mse) / baseline_temp_mse * 100
    trans_improvement = (baseline_trans_mse - trans_mse) / baseline_trans_mse * 100
    
    results = {
        "temporal_mse": temp_mse,
        "temporal_baseline": baseline_temp_mse,
        "temporal_improvement": temp_improvement,
        "transfer_mse": trans_mse,
        "transfer_baseline": baseline_trans_mse,
        "transfer_improvement": trans_improvement,
        "combined_score": (temp_improvement + trans_improvement) / 2
    }
    
    print("\n=== H3.21 Results ===")
    print(f"Temporal MSE: {temp_mse:.4f} (baseline: {baseline_temp_mse:.4f})")
    print(f"Temporal Improvement: {temp_improvement:.1f}%")
    print(f"Transfer MSE: {trans_mse:.4f} (baseline: {baseline_trans_mse:.4f})")
    print(f"Transfer Improvement: {trans_improvement:.1f}%")
    print(f"Combined Score: {results['combined_score']:.1f}%")
    
    return results


if __name__ == "__main__":
    results = train_and_evaluate()
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")