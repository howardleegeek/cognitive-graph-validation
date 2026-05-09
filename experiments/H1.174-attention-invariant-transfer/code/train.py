#!/usr/bin/env python3
"""
H1.174: Attention + Invariant on Cross-Dynamics Transfer
Simplified version focusing on the core question.
"""
import torch
import numpy as np
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

def generate_data(n_samples, friction=0.2, mass=1.0):
    """Generate manipulation data."""
    states, targets = [], []
    for _ in range(n_samples):
        obj = np.random.randn(3) * 0.2
        goal = np.random.randn(3) * 0.2
        
        # Simple dynamics
        force = (goal - obj) * 0.1
        damping = -friction * obj
        accel = (force + damping) / mass
        next_obj = obj + accel + np.random.randn(3) * 0.01
        
        state = np.concatenate([obj, np.zeros(3)])
        target = next_obj
        
        states.append(state)
        targets.append(target)
    
    return np.array(states), np.array(targets)

class AttentionInvariant(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, 3)
        
    def forward(self, x):
        h = self.enc(x)
        # Invariant: remove mean
        h = h - h.mean()
        return self.out(h)

class AttentionCrossModal(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.state_enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.goal_enc = nn.Sequential(nn.Linear(3, hidden), nn.ReLU())
        self.attn = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.out = nn.Linear(hidden, 3)
        
    def forward(self, state, goal):
        s = self.state_enc(state).unsqueeze(1)  # (batch, 1, hidden)
        g = self.goal_enc(goal).unsqueeze(1)    # (batch, 1, hidden)
        
        # Cross attention: state attends to goal
        out, _ = self.attn(s, g, g)
        return self.out(out.squeeze(1))

def train(model, states, targets, epochs=100):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        opt.zero_grad()
        if isinstance(model, AttentionCrossModal):
            pred = model(torch.FloatTensor(states), torch.FloatTensor(targets[:len(states)]))
        else:
            pred = model(torch.FloatTensor(states))
        loss = nn.MSELoss()(pred, torch.FloatTensor(targets))
        loss.backward()
        opt.step()

def eval_model(model, states, targets):
    with torch.no_grad():
        if isinstance(model, AttentionCrossModal):
            pred = model(torch.FloatTensor(states), torch.FloatTensor(targets[:len(states)]))
        else:
            pred = model(torch.FloatTensor(states))
        return nn.MSELoss()(pred, torch.FloatTensor(targets)).item()

def run():
    print("="*60)
    print("H1.174: Attention + Invariant on Cross-Dynamics Transfer")
    print("="*60)
    
    # Mixed training dynamics
    train_states, train_targets = [], []
    for f, m in [(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)]:
        s, t = generate_data(200, friction=f, mass=m)
        train_states.append(s)
        train_targets.append(t)
    train_states = np.concatenate(train_states)
    train_targets = np.concatenate(train_targets)
    
    # Test dynamics
    test_scenarios = [
        generate_data(100, friction=0.5, mass=1.0),
        generate_data(100, friction=0.05, mass=1.0),
        generate_data(100, friction=0.3, mass=1.5)
    ]
    
    # Reference: same dynamics
    ref_s, ref_t = generate_data(200, friction=0.2, mass=1.0)
    
    results = []
    
    for i, (test_s, test_t) in enumerate(test_scenarios):
        # Same dynamics reference
        m_ref = AttentionInvariant()
        train(m_ref, ref_s, ref_t)
        mse_ref = eval_model(m_ref, ref_s, ref_t)
        
        # Attention + invariant
        m_attn = AttentionCrossModal()
        train(m_attn, train_states, train_targets)
        mse_attn = eval_model(m_attn, test_s, test_t)
        
        # Baseline invariant
        m_inv = AttentionInvariant()
        train(m_inv, train_states, train_targets)
        mse_inv = eval_model(m_inv, test_s, test_t)
        
        delta_attn = (mse_ref - mse_attn) / mse_ref * 100
        delta_inv = (mse_ref - mse_inv) / mse_ref * 100
        
        print(f"\nTest {i+1}:")
        print(f"  Same dynamics ref: {mse_ref:.6f}")
        print(f"  Attention transfer: {mse_attn:.6f} ({delta_attn:+.1f}%)")
        print(f"  Invariant transfer: {mse_inv:.6f} ({delta_inv:+.1f}%)")
        
        results.append({
            'delta_attn': delta_attn,
            'delta_inv': delta_inv
        })
    
    avg_attn = np.mean([r['delta_attn'] for r in results])
    avg_inv = np.mean([r['delta_inv'] for r in results])
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Attention+Invariant avg transfer delta: {avg_attn:+.1f}%")
    print(f"Invariant only avg transfer delta: {avg_inv:+.1f}%")
    
    if avg_attn > avg_inv + 5:
        status = "SUPPORTED"
        print(f"\n{status}: Attention improves transfer over invariant alone")
    elif avg_attn > avg_inv:
        status = "MARGINAL"
        print(f"\n{status}: Attention marginally helps transfer")
    else:
        status = "REFUTED"
        print(f"\n{status}: Invariant alone is better for transfer")
    
    return {'status': status, 'avg_attn': avg_attn, 'avg_inv': avg_inv}

if __name__ == "__main__":
    run()