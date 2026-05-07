"""
H1.138: SSM on Very Long Sequences (100+ timesteps)
Based on H3.66 (+27.9% SSM-only best), H1.136 (decay scaling)
Test SSM on ultra-long sequences where standard attention may struggle.
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


class SSMModel(nn.Module):
    """SSM with dynamics-based temporal modeling."""
    
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        # Stacked SSM layers
        self.ssm_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            for _ in range(num_layers)
        ])
        
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
        
        # Apply SSM layers
        for layer in self.ssm_layers:
            x = x + layer(x)
        
        return self.fc(x)


class AttentionModel(nn.Module):
    """Standard attention for comparison."""
    
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


class SSMAttention(nn.Module):
    """SSM + Attention combined."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        self.ssm = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
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
        
        ssm_out = self.ssm(x)
        
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        
        out = x + 0.5 * ssm_out + 0.5 * attn_out.squeeze(0)
        
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


def generate_long_task_data(num_samples, seq_len, dynamics="default"):
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


def evaluate_model(model, seq_lens=[50, 75, 100, 125]):
    results = {}
    model.eval()
    
    with torch.no_grad():
        for seq_len in seq_lens:
            data = generate_long_task_data(100, seq_len)
            pred = model(data['state'], data['action'])
            mse = F.mse_loss(pred, data['next_state']).item()
            results[seq_len] = mse
    
    return results


def run_experiment():
    print("\n" + "="*60)
    print("H1.138: SSM on Very Long Sequences (100+ timesteps)")
    print("="*60 + "\n")
    
    config = ExperimentConfig()
    
    models = [
        ('SSM (2 layers)', SSMModel(config, num_layers=2)),
        ('SSM (3 layers)', SSMModel(config, num_layers=3)),
        ('Attention', AttentionModel(config)),
        ('SSM + Attention', SSMAttention(config)),
        ('Baseline', Baseline(config)),
    ]
    
    all_results = {}
    
    for name, model in models:
        print(f"\n--- Testing: {name} ---")
        
        train_data = generate_long_task_data(500, 75)
        train_model(model, train_data)
        
        results = evaluate_model(model)
        all_results[name] = results
        
        avg = np.mean(list(results.values()))
        print(f"  Average MSE: {avg:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    baseline_mse = np.mean(list(all_results['Baseline'].values()))
    for name, results in all_results.items():
        avg = np.mean(list(results.values()))
        improvement = (baseline_mse - avg) / baseline_mse * 100
        print(f"\n{name}: {improvement:+.1f}% vs baseline")
        for seq_len, mse in results.items():
            print(f"  {seq_len}-step: MSE={mse:.4f}")
    
    results = {
        name: {str(s): v for s, v in r.items()}
        for name, r in all_results.items()
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
