#!/usr/bin/env python3
"""
H3.81: Temporal Attention Focus on Important Timesteps
Based on H1.174 (+98.2% transfer), test if attention that focuses on
the most informative timesteps helps generalization.
"""
import torch
import numpy as np
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

def generate_data(n_samples, friction=0.2, mass=1.0, seq_len=20):
    states, targets = [], []
    for _ in range(n_samples):
        obj = np.random.randn(3) * 0.2
        goal = np.random.randn(3) * 0.2
        
        traj = [obj.copy()]
        for t in range(seq_len):
            force = (goal - obj) * 0.15
            damping = -friction * obj
            accel = (force + damping) / mass
            obj = obj + accel + np.random.randn(3) * 0.01
            traj.append(obj.copy())
        
        feat = [np.concatenate([traj[t], traj[t+1] - traj[t]]) for t in range(seq_len)]
        tgt = [traj[t+1] for t in range(seq_len)]
        
        states.append(np.array(feat))
        targets.append(np.array(tgt))
    
    return np.array(states), np.array(targets)

class TemporalAttention(nn.Module):
    """Attention that learns which timesteps are important."""
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.temporal_score = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Softmax(dim=1)
        )
        self.out = nn.Linear(hidden, 3)
        
    def forward(self, seq):
        x = self.enc(seq)
        
        # Learn importance weights for each timestep
        weights = self.temporal_score(x)  # (batch, seq, 1)
        
        # Weighted sum of all timesteps
        ctx = torch.sum(x * weights, dim=1)  # (batch, hidden)
        
        return self.out(ctx)

class LastStepAttention(nn.Module):
    """Focus on last few timesteps."""
    def __init__(self, hidden=64, lookback=5):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.lookback = lookback
        self.fc = nn.Sequential(
            nn.Linear(hidden * lookback, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )
        
    def forward(self, seq):
        x = self.enc(seq[:, -self.lookback:, :])
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class WeightedAttention(nn.Module):
    """Linear attention with importance weighting."""
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, 3)
        
    def forward(self, seq):
        x = self.enc(seq)
        # Exponential decay weighting: recent timesteps more important
        seq_len = x.size(1)
        decay = torch.exp(-torch.arange(seq_len, device=x.device).float() * 0.1)
        decay = decay / decay.sum()
        weights = decay.unsqueeze(0).unsqueeze(-1).expand_as(x)
        
        # Weighted sum
        ctx = torch.sum(x * weights, dim=1)
        return self.out(ctx)

class ConcatBaseline(nn.Module):
    def __init__(self, seq_len=20, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(hidden * seq_len, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, 3)
        )
        self._seq_len = seq_len
        
    def forward(self, seq):
        x = self.enc(seq)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

def train(model, states, targets, epochs=100):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        opt.zero_grad()
        tgt = torch.FloatTensor(targets[:, -1, :])
        pred = model(torch.FloatTensor(states))
        loss = nn.MSELoss()(pred, tgt)
        loss.backward()
        opt.step()

def evaluate(model, states, targets):
    with torch.no_grad():
        tgt = torch.FloatTensor(targets[:, -1, :])
        pred = model(torch.FloatTensor(states))
        return nn.MSELoss()(pred, tgt).item()

def run():
    print("="*60)
    print("H3.81: Temporal Attention Focus on Important Timesteps")
    print("="*60)
    
    # Training: mixed dynamics
    train_data = []
    for f, m in [(0.2, 1.0), (0.15, 0.8), (0.25, 1.2)]:
        s, t = generate_data(200, friction=f, mass=m)
        train_data.append((s, t))
    
    # Test: different dynamics
    test_scenarios = [
        ("high friction", 0.5, 1.0),
        ("low friction", 0.05, 1.0),
        ("heavy mass", 0.2, 2.0),
        ("light mass", 0.2, 0.5)
    ]
    
    # Train models
    m_concat = ConcatBaseline(seq_len=20)
    m_temporal = TemporalAttention()
    m_last = LastStepAttention()
    m_weighted = WeightedAttention()
    
    for s, t in train_data:
        train(m_concat, s, t, epochs=50)
        train(m_temporal, s, t, epochs=50)
        train(m_last, s, t, epochs=50)
        train(m_weighted, s, t, epochs=50)
    
    results = []
    
    print("\n--- Generalization Results ---")
    
    for name, friction, mass in test_scenarios:
        test_s, test_t = generate_data(100, friction=friction, mass=mass)
        
        mse_concat = evaluate(m_concat, test_s, test_t)
        mse_temporal = evaluate(m_temporal, test_s, test_t)
        mse_last = evaluate(m_last, test_s, test_t)
        mse_weighted = evaluate(m_weighted, test_s, test_t)
        
        print(f"\n{name}:")
        print(f"  Concat: {mse_concat:.6f}")
        print(f"  Temporal Attn: {mse_temporal:.6f}")
        print(f"  Last-5 Attn: {mse_last:.6f}")
        print(f"  Weighted Attn: {mse_weighted:.6f}")
        
        results.append({
            'name': name,
            'concat': mse_concat,
            'temporal': mse_temporal,
            'last': mse_last,
            'weighted': mse_weighted
        })
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_concat = np.mean([r['concat'] for r in results])
    best_attn = min(np.mean([r['temporal'] for r in results]),
                    np.mean([r['last'] for r in results]),
                    np.mean([r['weighted'] for r in results]))
    
    improvement = (avg_concat - best_attn) / avg_concat * 100
    
    print(f"Concat avg: {avg_concat:.6f}")
    print(f"Best attention avg: {best_attn:.6f}")
    print(f"Improvement: {improvement:+.2f}%")
    
    if improvement > 10:
        status = "SUPPORTED"
    elif improvement > 0:
        status = "MARGINAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    return {'status': status, 'improvement': improvement}

if __name__ == "__main__":
    run()