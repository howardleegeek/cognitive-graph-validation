#!/usr/bin/env python3
"""H3.44: AGT-World Style Hierarchical - Simplified"""

import numpy as np
import torch
import torch.nn as nn

class Hierarchical(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.planner = nn.Linear(state_dim, action_dim * 2)
        self.executor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, state, action):
        subgoals = self.planner(state).chunk(2, dim=-1)
        out = state
        for sg in subgoals:
            out = self.executor(torch.cat([out, sg], dim=-1))
        return out

class Flat(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

def run():
    print("H3.44: Hierarchical Task Decomposition")
    
    state_dim, action_dim = 16, 8
    num_samples = 50
    task_lengths = [5, 10, 15]
    results = []
    
    for ns in task_lengths:
        np.random.seed(ns); torch.manual_seed(ns)
        
        S = torch.randn(num_samples, ns, state_dim)
        A = torch.randn(num_samples, ns, action_dim)
        T = S + torch.randn(num_samples, ns, state_dim) * 0.1
        
        # Hierarchical
        hier = Hierarchical(state_dim, action_dim)
        opt = torch.optim.Adam(hier.parameters(), lr=1e-2)
        for _ in range(20):
            for i in range(num_samples):
                opt.zero_grad()
                pred = hier(S[i,0], A[i,0])
                loss = ((pred - T[i,0])**2).mean()
                loss.backward(); opt.step()
        hier_loss = sum(((hier(S[i,0], A[i,0]) - T[i,0])**2).mean().item() for i in range(num_samples)) / num_samples
        
        # Flat
        flat = Flat(state_dim, action_dim)
        opt = torch.optim.Adam(flat.parameters(), lr=1e-2)
        for _ in range(20):
            for i in range(num_samples):
                opt.zero_grad()
                pred = flat(S[i,0], A[i,0])
                loss = ((pred - T[i,0])**2).mean()
                loss.backward(); opt.step()
        flat_loss = sum(((flat(S[i,0], A[i,0]) - T[i,0])**2).mean().item() for i in range(num_samples)) / num_samples
        
        improvement = (flat_loss - hier_loss) / flat_loss * 100
        print(f"{ns} steps: Flat={flat_loss:.4f}, Hier={hier_loss:.4f}, Δ={improvement:+.1f}%")
        results.append({'steps': ns, 'flat': flat_loss, 'hier': hier_loss, 'improvement': improvement})
    
    avg = np.mean([r['improvement'] for r in results])
    status = "SUPPORTED" if avg > 5 else "REFUTED" if avg < -5 else "INCONCLUSIVE"
    print(f"Avg: {avg:+.1f}% | Status: {status}")
    
    return results, status

if __name__ == "__main__":
    run()