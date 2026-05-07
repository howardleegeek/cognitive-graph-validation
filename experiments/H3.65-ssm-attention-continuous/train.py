"""
H3.65: SSM + Attention Hybrid on Continuous Control
Based on H3.8 (SSM +93%), H3.9 (Mamba +93%), H3.56 (inconclusive +5.2% attn -4.7% graph)
Test if combining SSM with attention overcomes H3.62 (causal attention -45%)
Simplified implementation focusing on key comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExperimentConfig:
    hidden_dim = 256
    state_dim = 16
    action_dim = 8
    dropout = 0.1


class SSMAttention(nn.Module):
    """Simple SSM with attention - both outputs match hidden_dim."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        # SSM-like dynamics (hidden_dim -> hidden_dim)
        self.ssm_dynamics = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Attention
        self.num_heads = 4
        self.attn = nn.MultiheadAttention(config.hidden_dim, self.num_heads, dropout=config.dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
    def forward(self, state, action):
        s = self.state_proj(state)
        a = self.action_proj(action)
        x = s + a  # (batch, hidden_dim)
        
        # SSM-like dynamics (outputs hidden_dim)
        ssm_out = self.ssm_dynamics(x)
        
        # Attention (outputs hidden_dim)
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(0)
        
        # Combine (both in hidden_dim)
        out = x + 0.5 * ssm_out + 0.5 * attn_out
        
        return self.fc(out)


class SSMOnly(nn.Module):
    """SSM only."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        self.ssm_dynamics = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
    def forward(self, state, action):
        s = self.state_proj(state)
        a = self.action_proj(action)
        x = s + a
        
        ssm_out = self.ssm_dynamics(x)
        out = x + ssm_out
        
        return self.fc(out)


class AttentionOnly(nn.Module):
    """Attention only."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        self.num_heads = 4
        self.attn = nn.MultiheadAttention(config.hidden_dim, self.num_heads, dropout=config.dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
    def forward(self, state, action):
        s = self.state_proj(state)
        a = self.action_proj(action)
        x = s + a
        
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x = x + attn_out.squeeze(0)
        
        return self.fc(x)


class Baseline(nn.Module):
    """Simple concatenation baseline."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.fusion = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.fusion(x)
        return self.fc(x)


def generate_continuous_control_data(num_samples, seq_len, dynamics="default"):
    """Generate continuous control task data."""
    
    if dynamics == "default":
        friction, mass = 0.2, 1.0
    elif dynamics == "high_friction":
        friction, mass = 0.5, 1.0
    elif dynamics == "low_friction":
        friction, mass = 0.05, 1.0
    elif dynamics == "heavy_mass":
        friction, mass = 0.2, 2.0
    else:
        friction, mass = 0.2, 1.0
    
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_samples):
        state = np.random.randn(16).astype(np.float32) * 0.1
        
        for t in range(seq_len):
            action = np.random.randn(8).astype(np.float32) * 0.1
            
            next_state = state.copy()
            next_state[:3] += action[:3] * friction / mass
            next_state[3:6] += action[3:6] / mass
            
            states.append(state.copy())
            actions.append(action.copy())
            next_states.append(next_state.copy())
            
            state = next_state
    
    return {
        'state': torch.tensor(np.array(states), dtype=torch.float32),
        'action': torch.tensor(np.array(actions), dtype=torch.float32),
        'next_state': torch.tensor(np.array(next_states), dtype=torch.float32)
    }


def train_model(model, data, num_epochs=150):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(data['state'], data['action'])
        loss = criterion(pred, data['next_state'])
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")
    
    return loss.item()


def evaluate_model(model, seq_lens=[30, 50, 75, 100]):
    results = {}
    model.eval()
    
    with torch.no_grad():
        for seq_len in seq_lens:
            data = generate_continuous_control_data(100, seq_len)
            pred = model(data['state'], data['action'])
            mse = F.mse_loss(pred, data['next_state']).item()
            results[seq_len] = mse
    
    return results


def evaluate_transfer(model):
    results = {}
    target_dynamics = ["high_friction", "low_friction", "heavy_mass"]
    
    model.eval()
    with torch.no_grad():
        for dyn in target_dynamics:
            data = generate_continuous_control_data(100, 30, dyn)
            pred = model(data['state'], data['action'])
            mse = F.mse_loss(pred, data['next_state']).item()
            results[dyn] = mse
    
    return results


def run_experiment():
    print("\n" + "="*60)
    print("H3.65: SSM + Attention Hybrid on Continuous Control")
    print("="*60 + "\n")
    
    config = ExperimentConfig()
    
    models = [
        ('SSM + Attention', SSMAttention(config)),
        ('SSM Only', SSMOnly(config)),
        ('Attention Only', AttentionOnly(config)),
        ('Baseline', Baseline(config)),
    ]
    
    all_temporal_results = {}
    all_transfer_results = {}
    
    for name, model in models:
        print(f"\n--- Testing: {name} ---")
        
        train_data = generate_continuous_control_data(500, 30)
        train_model(model, train_data)
        
        temporal = evaluate_model(model)
        all_temporal_results[name] = temporal
        
        transfer = evaluate_transfer(model)
        all_transfer_results[name] = transfer
        
        avg_temporal = np.mean(list(temporal.values()))
        avg_transfer = np.mean(list(transfer.values()))
        print(f"  Avg Temporal: {avg_temporal:.4f}, Avg Transfer: {avg_transfer:.4f}")
    
    print("\n" + "="*60)
    print("TEMPORAL REASONING RESULTS")
    print("="*60)
    
    baseline_temporal = np.mean(list(all_temporal_results["Baseline"].values()))
    for name, temporal in all_temporal_results.items():
        avg = np.mean(list(temporal.values()))
        improvement = (baseline_temporal - avg) / baseline_temporal * 100
        print(f"\n{name}: {improvement:+.1f}% vs baseline")
        for seq_len, mse in temporal.items():
            print(f"  {seq_len}-step: MSE={mse:.4f}")
    
    print("\n" + "="*60)
    print("CROSS-DYNAMICS TRANSFER RESULTS")
    print("="*60)
    
    baseline_transfer = np.mean(list(all_transfer_results["Baseline"].values()))
    for name, transfer in all_transfer_results.items():
        avg = np.mean(list(transfer.values()))
        improvement = (baseline_transfer - avg) / baseline_transfer * 100
        print(f"\n{name}: {improvement:+.1f}% vs baseline")
        for dyn, mse in transfer.items():
            print(f"  {dyn}: MSE={mse:.4f}")
    
    results = {
        'temporal': {k: {str(s): v for s, v in t.items()} for k, t in all_temporal_results.items()},
        'transfer': {k: {str(d): v for d, v in t.items()} for k, t in all_transfer_results.items()}
    }
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    
    import json
    import os
    os.makedirs("results", exist_ok=True)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
