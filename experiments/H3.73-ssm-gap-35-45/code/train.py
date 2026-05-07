#!/usr/bin/env python3
"""
H3.73: SSM gap test - SSM 35-45 timesteps where H3.72 showed high variance
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_data(n_samples, seq_len, seed=42):
    """Generate trajectory data."""
    np.random.seed(seed)
    states = []
    actions = []
    
    for _ in range(n_samples):
        pos = np.random.randn(2) * 0.3
        s_seq = []
        a_seq = []
        
        for _ in range(seq_len):
            # 7D state
            s = np.concatenate([pos, [0.5, 0.5, 0.0, 0.5, 1.0]])
            s_seq.append(s)
            
            target = np.random.randn(2) * 0.3
            action = target - pos
            a_seq.append(action)
            
            pos = pos + action * 0.1 + np.random.randn(2) * 0.01
        
        states.append(np.array(s_seq))
        actions.append(np.array(a_seq))
    
    return {'states': np.array(states), 'actions': np.array(actions)}


class SSM(nn.Module):
    """SSM with gated recurrence."""
    
    def __init__(self, input_dim=7, hidden_dim=256):
        super().__init__()
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # x: (batch, seq, 7)
        seq_len = x.shape[1]
        h = torch.zeros(x.shape[0], self.x_proj.out_features, device=x.device)
        
        for t in range(seq_len):
            z = F.silu(self.x_proj(x[:, t, :]))
            h = h * 0.9 + z * 0.1
        
        return self.out(h)


class Baseline(nn.Module):
    """Simple MLP baseline - uses both first and last timesteps."""
    
    def __init__(self):
        super().__init__()
        # Input: first + last timestep = 14
        self.net = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        # x: (batch, seq, 7), use first and last
        x_first = x[:, 0, :]
        x_last = x[:, -1, :]
        x_comb = torch.cat([x_first, x_last], dim=-1)
        return self.net(x_comb)


def train(model, train_data, epochs=100):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    
    X = torch.tensor(train_data['states'], dtype=torch.float32)
    Y = torch.tensor(train_data['actions'], dtype=torch.float32)
    
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = crit(pred, Y[:, -1, :])
        loss.backward()
        opt.step()


def eval(model, val_data):
    model.eval()
    crit = nn.MSELoss()
    
    X = torch.tensor(val_data['states'], dtype=torch.float32)
    Y = torch.tensor(val_data['actions'], dtype=torch.float32)
    
    with torch.no_grad():
        pred = model(X)
        loss = crit(pred, Y[:, -1, :]).item()
    
    return loss


def run():
    print("=" * 60)
    print("H3.73: SSM Gap Test (35-45 timesteps)")
    print("=" * 60)
    
    results = {}
    
    for seq_len in [35, 40, 45]:
        print(f"\nSeq_len={seq_len}:")
        
        train_d = gen_data(n_samples=50, seq_len=seq_len, seed=42)
        val_d = gen_data(n_samples=20, seq_len=seq_len, seed=999)
        
        # SSM
        ssm = SSM()
        train(ssm, train_d)
        ssm_loss = eval(ssm, val_d)
        
        # Baseline
        base = Baseline()
        train(base, train_d)
        base_loss = eval(base, val_d)
        
        delta = (base_loss - ssm_loss) / base_loss * 100
        
        results[f'{seq_len}'] = {
            'baseline': base_loss,
            'ssm': ssm_loss,
            'delta': delta
        }
        
        print(f"  Baseline: {base_loss:.4f}, SSM: {ssm_loss:.4f}, Δ={delta:+.1f}%")
    
    avg_delta = np.mean([v['delta'] for v in results.values()])
    status = "SUPPORTED" if avg_delta > 10 else "REFUTED" if avg_delta < -10 else "INCONCLUSIVE"
    
    print(f"\nStatus: {status}, Avg Δ: {avg_delta:+.1f}%")
    
    output = {
        'experiment': 'H3.73',
        'status': status,
        'avg_improvement': avg_delta,
        'results': results
    }
    print(json.dumps(output, indent=2))
    
    return output


if __name__ == "__main__":
    run()