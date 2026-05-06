#!/usr/bin/env python3
"""H3.42: GWM-Style Action Nodes - Fast version"""

import numpy as np
import torch
import torch.nn as nn

class GWMEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_net = nn.Linear(state_dim, hidden_dim)
        self.action_net = nn.Linear(action_dim, hidden_dim)
        self.combine = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, state_dim))
        
    def forward(self, state, action):
        return self.combine(torch.cat([self.state_net(state), self.action_net(action)], dim=-1))

class BaselineEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, state_dim))
        
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

def run():
    print("H3.42: GWM Action Nodes")
    
    state_dim, action_dim = 16, 8
    num_samples = 50
    task_lengths = [5, 10, 15]
    results = []
    
    for ns in task_lengths:
        np.random.seed(ns); torch.manual_seed(ns)
        
        S = torch.randn(num_samples, ns, state_dim)
        A = torch.randn(num_samples, ns, action_dim)
        T = S + torch.randn(num_samples, ns, state_dim) * 0.1
        
        # GWM
        gwm = GWMEncoder(state_dim, action_dim)
        opt = torch.optim.Adam(gwm.parameters(), lr=1e-2)
        for _ in range(20):
            for i in range(num_samples):
                opt.zero_grad()
                loss = ((gwm(S[i,0], A[i,0]) - T[i,0])**2).mean()
                loss.backward(); opt.step()
        gwm_loss = sum(((gwm(S[i,0], A[i,0]) - T[i,0])**2).mean().item() for i in range(num_samples)) / num_samples
        
        # Baseline
        base = BaselineEncoder(state_dim, action_dim)
        opt = torch.optim.Adam(base.parameters(), lr=1e-2)
        for _ in range(20):
            for i in range(num_samples):
                opt.zero_grad()
                loss = ((base(S[i,0], A[i,0]) - T[i,0])**2).mean()
                loss.backward(); opt.step()
        base_loss = sum(((base(S[i,0], A[i,0]) - T[i,0])**2).mean().item() for i in range(num_samples)) / num_samples
        
        improvement = (base_loss - gwm_loss) / base_loss * 100
        print(f"{ns} steps: B={base_loss:.4f}, GWM={gwm_loss:.4f}, Δ={improvement:+.1f}%")
        results.append({'steps': ns, 'baseline': base_loss, 'gwm': gwm_loss, 'improvement': improvement})
    
    avg = np.mean([r['improvement'] for r in results])
    status = "SUPPORTED" if avg > 5 else "REFUTED" if avg < -5 else "INCONCLUSIVE"
    print(f"Avg: {avg:+.1f}% | Status: {status}")
    
    return results, status

if __name__ == "__main__":
    run()