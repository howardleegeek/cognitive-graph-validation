#!/usr/bin/env python3
"""
H3.82: Hierarchical Temporal Abstraction with Attention
Based on H3.81 showing Last-5 attention (+56.0%), test if hierarchical
abstraction (coarse + fine temporal scales) helps.
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

class HierarchicalAttention(nn.Module):
    """Hierarchical: coarse (every 5) + fine (last 5) attention."""
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        
        # Coarse level: every 5 timesteps
        self.coarse_q = nn.Linear(hidden, hidden)
        self.coarse_k = nn.Linear(hidden, hidden)
        self.coarse_v = nn.Linear(hidden, hidden)
        
        # Fine level: last 5 timesteps
        self.fine_q = nn.Linear(hidden, hidden)
        self.fine_k = nn.Linear(hidden, hidden)
        self.fine_v = nn.Linear(hidden, hidden)
        
        self.out = nn.Linear(hidden * 2, 3)
        
    def forward(self, seq):
        x = self.enc(seq)
        
        # Coarse attention: every 5 timesteps
        coarse_indices = list(range(0, x.size(1), 5))
        if len(coarse_indices) < 2:
            coarse_indices = [0, x.size(1)-1]
        coarse_x = x[:, coarse_indices, :]
        
        q_coarse = self.coarse_q(x[:, -1:, :])
        k_coarse = self.coarse_k(coarse_x)
        v_coarse = self.coarse_v(coarse_x)
        scores_coarse = torch.matmul(q_coarse, k_coarse.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attn_coarse = torch.softmax(scores_coarse, dim=-1)
        ctx_coarse = torch.matmul(attn_coarse, v_coarse).squeeze(1)
        
        # Fine attention: last 5 timesteps
        fine_x = x[:, -5:, :]
        q_fine = self.fine_q(x[:, -1:, :])
        k_fine = self.fine_k(fine_x)
        v_fine = self.fine_v(fine_x)
        scores_fine = torch.matmul(q_fine, k_fine.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attn_fine = torch.softmax(scores_fine, dim=-1)
        ctx_fine = torch.matmul(attn_fine, v_fine).squeeze(1)
        
        # Combine
        ctx = torch.cat([ctx_coarse, ctx_fine], dim=-1)
        return self.out(ctx)

class MultiScaleAttention(nn.Module):
    """Multi-scale: 3 different window sizes."""
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        
        self.scales = [3, 5, 7]
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        
        self.out = nn.Linear(hidden * len(self.scales), 3)
        
    def forward(self, seq):
        x = self.enc(seq)
        
        contexts = []
        for scale in self.scales:
            start = max(0, x.size(1) - scale)
            window = x[:, start:, :]
            
            q = self.q_proj(x[:, -1:, :])
            k = self.k_proj(window)
            v = self.v_proj(window)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(x.size(-1))
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v).squeeze(1)
            contexts.append(ctx)
        
        ctx = torch.cat(contexts, dim=-1)
        return self.out(ctx)

class Last5Baseline(nn.Module):
    """Last-5 attention baseline from H3.81."""
    def __init__(self, hidden=64, lookback=5):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(hidden * lookback, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )
        self._lookback = lookback
        
    def forward(self, seq):
        x = self.enc(seq[:, -self._lookback:, :])
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

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
    print("H3.82: Hierarchical Temporal Abstraction with Attention")
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
    m_last5 = Last5Baseline()
    m_hier = HierarchicalAttention()
    m_multi = MultiScaleAttention()
    
    for s, t in train_data:
        train(m_concat, s, t, epochs=50)
        train(m_last5, s, t, epochs=50)
        train(m_hier, s, t, epochs=50)
        train(m_multi, s, t, epochs=50)
    
    results = []
    
    print("\n--- Generalization Results ---")
    
    for name, friction, mass in test_scenarios:
        test_s, test_t = generate_data(100, friction=friction, mass=mass)
        
        mse_concat = evaluate(m_concat, test_s, test_t)
        mse_last5 = evaluate(m_last5, test_s, test_t)
        mse_hier = evaluate(m_hier, test_s, test_t)
        mse_multi = evaluate(m_multi, test_s, test_t)
        
        print(f"\n{name}:")
        print(f"  Concat: {mse_concat:.6f}")
        print(f"  Last-5: {mse_last5:.6f}")
        print(f"  Hierarchical: {mse_hier:.6f}")
        print(f"  Multi-Scale: {mse_multi:.6f}")
        
        results.append({
            'name': name,
            'concat': mse_concat,
            'last5': mse_last5,
            'hier': mse_hier,
            'multi': mse_multi
        })
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_concat = np.mean([r['concat'] for r in results])
    avg_last5 = np.mean([r['last5'] for r in results])
    avg_hier = np.mean([r['hier'] for r in results])
    avg_multi = np.mean([r['multi'] for r in results])
    
    print(f"Concat avg: {avg_concat:.6f}")
    print(f"Last-5 avg: {avg_last5:.6f}")
    print(f"Hierarchical avg: {avg_hier:.6f}")
    print(f"Multi-Scale avg: {avg_multi:.6f}")
    
    best_attn = min(avg_last5, avg_hier, avg_multi)
    improvement = (avg_concat - best_attn) / avg_concat * 100
    
    print(f"\nBest improvement over concat: {improvement:+.2f}%")
    
    if improvement > avg_last5 / avg_concat * 100 - avg_concat:
        status = "SUPPORTED"
        print(f"Status: {status} - Hierarchical/multi-scale beats Last-5")
    elif improvement > 0:
        status = "MARGINAL"
        print(f"Status: {status} - Improvement over concat but not Last-5")
    else:
        status = "REFUTED"
        print(f"Status: {status} - Last-5 still best")
    
    return {'status': status, 'improvement': improvement}

if __name__ == "__main__":
    run()