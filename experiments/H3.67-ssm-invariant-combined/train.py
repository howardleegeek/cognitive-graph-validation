"""
H3.67: SSM + Invariant Combined Architecture
Based on H3.66 (+27.9% SSM-only) and H1.8 (+5.4% Invariant transfer)
Test if combining SSM with invariant learning solves both temporal and transfer.
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


class SSMInvariant(nn.Module):
    """SSM + Invariant combined architecture."""
    
    def __init__(self, config, use_invariant=True):
        super().__init__()
        self.config = config
        self.use_invariant = use_invariant
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        # SSM dynamics
        self.ssm_dynamics = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Invariant branch (bisimulation-inspired)
        if use_invariant:
            self.invariant_encoder = nn.Sequential(
                nn.Linear(config.state_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.invariant_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
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
        x = s + a
        
        # SSM branch
        ssm_out = self.ssm_dynamics(x)
        
        # Invariant branch (match dimensions)
        if self.use_invariant:
            inv = self.invariant_encoder(state)
            inv_out = self.invariant_proj(inv)  # (batch, hidden_dim)
            ssm_out = ssm_out + 0.3 * inv_out  # Weighted combination
        
        # Attention branch
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(0)
        
        # Combine
        out = x + ssm_out + 0.3 * attn_out
        
        return self.fc(out)


class SSMOnly(nn.Module):
    """SSM only baseline."""
    
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


class InvariantOnly(nn.Module):
    """Invariant only baseline."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        self.invariant_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
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
        
        inv = self.invariant_encoder(state)
        out = x + inv
        
        return self.fc(out)


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


def generate_task_data(num_samples, seq_len, dynamics="default"):
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


def evaluate_temporal(model, seq_lens=[30, 50, 75]):
    results = {}
    model.eval()
    
    with torch.no_grad():
        for seq_len in seq_lens:
            data = generate_task_data(100, seq_len)
            pred = model(data['state'], data['action'])
            mse = F.mse_loss(pred, data['next_state']).item()
            results[seq_len] = mse
    
    return results


def evaluate_transfer(model, dynamics_list=["high_friction", "low_friction", "heavy_mass"]):
    results = {}
    model.eval()
    
    with torch.no_grad():
        for dyn in dynamics_list:
            data = generate_task_data(100, 30, dyn)
            pred = model(data['state'], data['action'])
            mse = F.mse_loss(pred, data['next_state']).item()
            results[dyn] = mse
    
    return results


def run_experiment():
    print("\n" + "="*60)
    print("H3.67: SSM + Invariant Combined Architecture")
    print("="*60 + "\n")
    
    config = ExperimentConfig()
    
    models = [
        ('SSM + Invariant', SSMInvariant(config, use_invariant=True)),
        ('SSM Only', SSMOnly(config)),
        ('Invariant Only', InvariantOnly(config)),
        ('Baseline', Baseline(config)),
    ]
    
    all_temporal = {}
    all_transfer = {}
    
    for name, model in models:
        print(f"\n--- Testing: {name} ---")
        
        train_data = generate_task_data(500, 30)
        train_model(model, train_data)
        
        temporal = evaluate_temporal(model)
        all_temporal[name] = temporal
        
        transfer = evaluate_transfer(model)
        all_transfer[name] = transfer
        
        avg_t = np.mean(list(temporal.values()))
        avg_tr = np.mean(list(transfer.values()))
        print(f"  Temporal: {avg_t:.4f}, Transfer: {avg_tr:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    baseline_temp = np.mean(list(all_temporal["Baseline"].values()))
    baseline_tr = np.mean(list(all_transfer["Baseline"].values()))
    
    print("\nTemporal Reasoning:")
    for name, temporal in all_temporal.items():
        avg = np.mean(list(temporal.values()))
        improvement = (baseline_temp - avg) / baseline_temp * 100
        print(f"  {name}: {improvement:+.1f}% vs baseline")
    
    print("\nCross-Dynamics Transfer:")
    for name, transfer in all_transfer.items():
        avg = np.mean(list(transfer.values()))
        improvement = (baseline_tr - avg) / baseline_tr * 100
        print(f"  {name}: {improvement:+.1f}% vs baseline")
    
    results = {
        'temporal': {k: {str(s): v for s, v in t.items()} for k, t in all_temporal.items()},
        'transfer': {k: {str(d): v for d, v in t.items()} for k, t in all_transfer.items()}
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
