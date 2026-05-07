"""
H3.66: Adaptive SSM with Learned Mode Selection
Based on H3.65 results - test adaptive switching between SSM/attention/concat modes
Based on H3.64 (decay scaling +19.6%)
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


class AdaptiveModel(nn.Module):
    """Adaptive mode selection between SSM, attention, and concat."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        # SSM branch
        self.ssm_dynamics = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Attention branch
        self.num_heads = 4
        self.attn = nn.MultiheadAttention(config.hidden_dim, self.num_heads, dropout=config.dropout)
        
        # Mode selector
        self.mode_selector = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 0=concat, 1=SSM, 2=attention
            nn.Softmax(dim=-1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
    def forward(self, state, action, return_mode=False):
        s = self.state_proj(state)
        a = self.action_proj(action)
        x = s + a
        
        # Compute mode weights
        mode_weights = self.mode_selector(x)
        mode = torch.argmax(mode_weights, dim=-1)
        
        # SSM branch
        ssm_out = self.ssm_dynamics(x)
        
        # Attention branch
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(0)
        
        # Weighted combination
        w_concat = mode_weights[:, 0:1]
        w_ssm = mode_weights[:, 1:2]
        w_attn = mode_weights[:, 2:3]
        
        out = w_concat * x + w_ssm * ssm_out + w_attn * attn_out
        out = self.fc(out)
        
        if return_mode:
            return out, mode, mode_weights
        return out


class FixedSSMAttention(nn.Module):
    """Fixed SSM + Attention without adaptive selection."""
    
    def __init__(self, config, mode='ssm_attn'):
        super().__init__()
        self.config = config
        self.mode = mode
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        # SSM
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
        x = s + a
        
        if self.mode == 'ssm':
            ssm_out = self.ssm_dynamics(x)
            out = x + ssm_out
        elif self.mode == 'attn':
            x_seq = x.unsqueeze(0)
            attn_out, _ = self.attn(x_seq, x_seq, x_seq)
            out = x + attn_out.squeeze(0)
        elif self.mode == 'ssm_attn':
            ssm_out = self.ssm_dynamics(x)
            x_seq = x.unsqueeze(0)
            attn_out, _ = self.attn(x_seq, x_seq, x_seq)
            out = x + 0.5 * ssm_out + 0.5 * attn_out.squeeze(0)
        else:
            out = x
        
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


def evaluate_model(model, seq_lens=[20, 35, 50, 75]):
    results = {}
    model.eval()
    
    with torch.no_grad():
        for seq_len in seq_lens:
            data = generate_task_data(100, seq_len)
            pred = model(data['state'], data['action'])
            mse = F.mse_loss(pred, data['next_state']).item()
            results[seq_len] = mse
    
    return results


def run_experiment():
    print("\n" + "="*60)
    print("H3.66: Adaptive SSM with Learned Mode Selection")
    print("="*60 + "\n")
    
    config = ExperimentConfig()
    
    models = [
        ('adaptive', AdaptiveModel(config)),
        ('ssm_attn', FixedSSMAttention(config, mode='ssm_attn')),
        ('ssm', FixedSSMAttention(config, mode='ssm')),
        ('attn', FixedSSMAttention(config, mode='attn')),
        ('concat', Baseline(config)),
    ]
    
    all_results = {}
    
    for name, model in models:
        print(f"\n--- Testing: {name} ---")
        
        train_data = generate_task_data(500, 30)
        train_model(model, train_data)
        
        results = evaluate_model(model)
        all_results[name] = results
        
        avg = np.mean(list(results.values()))
        print(f"  Average MSE: {avg:.4f}")
        
        # For adaptive model, check mode distribution
        if name == 'adaptive':
            model.eval()
            with torch.no_grad():
                test_data = generate_task_data(100, 30)
                _, mode, weights = model(test_data['state'][:10], test_data['action'][:10], return_mode=True)
                print(f"  Mode distribution: concat={weights[:, 0].mean():.2f}, ssm={weights[:, 1].mean():.2f}, attn={weights[:, 2].mean():.2f}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    baseline_mse = np.mean(list(all_results['concat'].values()))
    for name, results in all_results.items():
        avg = np.mean(list(results.values()))
        improvement = (baseline_mse - avg) / baseline_mse * 100
        print(f"\n{name}: {improvement:+.1f}% vs baseline (avg MSE: {avg:.4f})")
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
