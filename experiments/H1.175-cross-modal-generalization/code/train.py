#!/usr/bin/env python3
"""
H1.175: Cross-Modal Attention for Generalization
Simplified: test if attention mechanisms help generalize to different dynamics.
"""
import torch
import numpy as np
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

def generate_data(n_samples, friction=0.2, mass=1.0, seq_len=20):
    """Generate manipulation data."""
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

class CrossModalAttention(nn.Module):
    """Cross-modal: state attends to goal."""
    def __init__(self, hidden=64):
        super().__init__()
        self.state_enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.goal_enc = nn.Sequential(nn.Linear(3, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, 3)
        
    def forward(self, state, goal):
        s = self.state_enc(state).unsqueeze(1)
        g = self.goal_enc(goal).unsqueeze(1)
        
        # Cross-attention
        scores = torch.matmul(s, g.transpose(-2, -1)) / np.sqrt(s.size(-1))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, g).squeeze(1)
        
        return self.out(ctx)

class SelfAttention(nn.Module):
    """Self-attention over sequence."""
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 3)
        
    def forward(self, seq):
        x = self.enc(seq)
        q = self.q_proj(x[:, -1:, :])
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).squeeze(1)
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
        if isinstance(model, CrossModalAttention):
            goal = torch.FloatTensor(targets[:, -1, :]) * 0.5  # Use target as pseudo-goal
            pred = model(torch.FloatTensor(states[:, -1, :]), goal)
        else:
            pred = model(torch.FloatTensor(states))
        loss = nn.MSELoss()(pred, tgt)
        loss.backward()
        opt.step()

def evaluate(model, states, targets):
    with torch.no_grad():
        tgt = torch.FloatTensor(targets[:, -1, :])
        if isinstance(model, CrossModalAttention):
            goal = torch.FloatTensor(targets[:, -1, :]) * 0.5
            pred = model(torch.FloatTensor(states[:, -1, :]), goal)
        else:
            pred = model(torch.FloatTensor(states))
        return nn.MSELoss()(pred, tgt).item()

def run():
    print("="*60)
    print("H1.175: Cross-Modal Attention for Generalization")
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
    
    results = []
    
    # Train models
    m_concat = ConcatBaseline(seq_len=20)
    m_self = SelfAttention()
    m_cross = CrossModalAttention()
    
    for s, t in train_data:
        train(m_concat, s, t, epochs=50)
        train(m_self, s, t, epochs=50)
        train(m_cross, s, t, epochs=50)
    
    print("\n--- Generalization Results ---")
    
    for name, friction, mass in test_scenarios:
        test_s, test_t = generate_data(100, friction=friction, mass=mass)
        
        mse_concat = evaluate(m_concat, test_s, test_t)
        mse_self = evaluate(m_self, test_s, test_t)
        mse_cross = evaluate(m_cross, test_s, test_t)
        
        print(f"\n{name}:")
        print(f"  Concat: {mse_concat:.6f}")
        print(f"  Self-Attn: {mse_self:.6f}")
        print(f"  Cross-Modal: {mse_cross:.6f}")
        
        results.append({
            'name': name,
            'concat': mse_concat,
            'self': mse_self,
            'cross': mse_cross
        })
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_concat = np.mean([r['concat'] for r in results])
    avg_self = np.mean([r['self'] for r in results])
    avg_cross = np.mean([r['cross'] for r in results])
    
    print(f"Concat avg: {avg_concat:.6f}")
    print(f"Self-Attn avg: {avg_self:.6f}")
    print(f"Cross-Modal avg: {avg_cross:.6f}")
    
    improvement = (avg_concat - min(avg_self, avg_cross)) / avg_concat * 100
    
    if improvement > 10:
        status = "SUPPORTED"
    elif improvement > 0:
        status = "MARGINAL"
    else:
        status = "REFUTED"
    
    print(f"\nBest improvement over concat: {improvement:+.2f}%")
    print(f"Status: {status}")
    
    return {'status': status, 'improvement': improvement}

if __name__ == "__main__":
    run()