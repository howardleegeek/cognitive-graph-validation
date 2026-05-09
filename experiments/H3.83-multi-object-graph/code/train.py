#!/usr/bin/env python3
"""
H3.83: Multi-Scale Attention on Multi-Object Tasks (Simplified)
Test multi-scale attention on multi-object tasks with varying dynamics.
"""
import torch
import numpy as np
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

def generate_data(n_samples, n_objects=2, seq_len=15, friction=0.2, mass=1.0):
    states, targets = [], []
    for _ in range(n_samples):
        objs = [np.random.randn(3) * 0.3 for _ in range(n_objects)]
        goals = [np.random.randn(3) * 0.3 for _ in range(n_objects)]
        
        traj = [[o.copy() for o in objs]]
        for t in range(seq_len):
            new_objs = []
            for i, (obj, goal) in enumerate(zip(objs, goals)):
                force = (goal - obj) * 0.15
                for j, other in enumerate(objs):
                    if i != j:
                        dist = np.linalg.norm(obj - other)
                        if dist < 0.5 and dist > 0:
                            force += (other - obj) * 0.2 / dist
                damping = -friction * obj
                accel = (force + damping) / mass
                obj = obj + accel + np.random.randn(3) * 0.01
                new_objs.append(obj)
            objs = new_objs
            traj.append([o.copy() for o in objs])
        
        feat = []
        for t in range(seq_len):
            f = []
            for i in range(n_objects):
                vel = traj[t+1][i] - traj[t][i] if t < seq_len-1 else np.zeros(3)
                f.extend(traj[t][i])
                f.extend(vel)
            feat.append(f)
        
        tgt = []
        for t in range(seq_len):
            f = []
            for i in range(n_objects):
                f.extend(traj[t+1][i])
            tgt.append(f)
        
        states.append(np.array(feat))
        targets.append(np.array(tgt))
    
    return np.array(states), np.array(targets)

class MultiScaleAttention(nn.Module):
    def __init__(self, n_objects=2, hidden=64):
        super().__init__()
        self.n_objects = n_objects
        self.feat_dim = n_objects * 6
        self.enc = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.ReLU())
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_objects * 3)
        
    def forward(self, seq):
        x = self.enc(seq)
        contexts = []
        for scale in [3, 5, 7]:
            start = max(0, x.size(1) - scale)
            window = x[:, start:, :]
            q = self.q_proj(x[:, -1:, :])
            k = self.k_proj(window)
            v = self.v_proj(window)
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(x.size(-1))
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v).squeeze(1)
            contexts.append(ctx)
        ctx = torch.stack(contexts, dim=-1).mean(dim=-1)
        return self.out(ctx)

class ConcatBaseline(nn.Module):
    def __init__(self, n_objects=2, seq_len=15, hidden=64):
        super().__init__()
        self.n_objects = n_objects
        self.feat_dim = n_objects * 6
        self.enc = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(hidden * seq_len, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, n_objects * 3)
        )
        self._seq_len = seq_len
        
    def forward(self, seq):
        x = self.enc(seq)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

def train(model, states, targets, epochs=80):
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
    print("H3.83: Multi-Scale Attention on Multi-Object Tasks")
    print("="*60)
    
    n_objs = 2
    train_states, train_targets = generate_data(400, n_objects=n_objs)
    
    test_scenarios = [
        ("high friction", 0.5, 1.0),
        ("low friction", 0.05, 1.0),
        ("heavy mass", 0.2, 2.0),
        ("light mass", 0.2, 0.5)
    ]
    
    m_concat = ConcatBaseline(n_objects=n_objs)
    m_multi = MultiScaleAttention(n_objects=n_objs)
    
    train(m_concat, train_states, train_targets)
    train(m_multi, train_states, train_targets)
    
    results = []
    
    print("\n--- Generalization Results ---")
    
    for name, friction, mass in test_scenarios:
        test_states, test_targets = generate_data(100, n_objects=n_objs, friction=friction, mass=mass)
        
        mse_concat = evaluate(m_concat, test_states, test_targets)
        mse_multi = evaluate(m_multi, test_states, test_targets)
        
        print(f"\n{name}:")
        print(f"  Concat: {mse_concat:.6f}")
        print(f"  Multi-Scale: {mse_multi:.6f}")
        
        results.append({
            'name': name,
            'concat': mse_concat,
            'multi': mse_multi
        })
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_concat = np.mean([r['concat'] for r in results])
    avg_multi = np.mean([r['multi'] for r in results])
    
    print(f"Concat avg: {avg_concat:.6f}")
    print(f"Multi-Scale avg: {avg_multi:.6f}")
    
    improvement = (avg_concat - avg_multi) / avg_concat * 100
    
    print(f"\nImprovement: {improvement:+.2f}%")
    
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