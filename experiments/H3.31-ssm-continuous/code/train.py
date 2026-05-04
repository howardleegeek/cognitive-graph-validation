#!/usr/bin/env python3
"""H3.31: SSM + Continuous Control"""

import torch
import torch.nn as nn
import numpy as np
import json


class ConcatFusion(nn.Module):
    def __init__(self, s_dim=16, a_dim=7, h_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, s_dim),
        )
    
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class SSMFusion(nn.Module):
    def __init__(self, s_dim=16, a_dim=7, h_dim=256, d_state=16):
        super().__init__()
        self.d_state = d_state
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_state, a_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(h_dim, d_state) * 0.1)
        
        self.net = nn.Sequential(
            nn.Linear(s_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, s_dim),
        )
    
    def forward(self, s, a):
        B = s.shape[0]
        h = torch.randn(B, self.d_state, device=s.device) * 0.1
        a_proj = a @ self.B.T
        h = torch.tanh(h @ self.A + a_proj)
        hs = h @ self.C.T
        return self.net(torch.cat([s, hs], dim=-1))


def main():
    print("H3.31: SSM + Continuous Control")
    print("=" * 50)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    s_dim, a_dim = 16, 7
    n_samples = 500
    
    states = []
    actions = []
    next_states = []
    
    for _ in range(n_samples):
        s = np.random.randn(20, s_dim) * 0.1
        a = np.random.randn(20, a_dim)
        ns = np.roll(s, -1, axis=0) + np.random.randn(20, s_dim) * 0.01
        ns[-1] = s[-1]
        
        states.append(s[:-1])
        actions.append(a[:-1])
        next_states.append(ns[1:])
    
    X_s = torch.tensor(np.concatenate(states), dtype=torch.float32)
    X_a = torch.tensor(np.concatenate(actions), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(next_states), dtype=torch.float32)
    
    idx = int(0.8 * len(X_s))
    X_s_train, X_s_test = X_s[:idx], X_s[idx:]
    X_a_train, X_a_test = X_a[:idx], X_a[idx:]
    Y_train, Y_test = Y[:idx], Y[idx:]
    
    # Concat
    c_model = ConcatFusion(s_dim, a_dim, 256)
    opt = torch.optim.Adam(c_model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    
    for e in range(30):
        for i in range(0, len(X_s_train), 32):
            s = X_s_train[i:i+32]
            a = X_a_train[i:i+32]
            y = Y_train[i:i+32]
            loss = crit(c_model(s, a), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    c_model.eval()
    with torch.no_grad():
        c_loss = crit(c_model(X_s_test, X_a_test), Y_test).item()
    
    # SSM
    s_model = SSMFusion(s_dim, a_dim, 256)
    opt = torch.optim.Adam(s_model.parameters(), lr=0.01)
    
    for e in range(30):
        for i in range(0, len(X_s_train), 32):
            s = X_s_train[i:i+32]
            a = X_a_train[i:i+32]
            y = Y_train[i:i+32]
            loss = crit(s_model(s, a), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    s_model.eval()
    with torch.no_grad():
        s_loss = crit(s_model(X_s_test, X_a_test), Y_test).item()
    
    delta = (c_loss - s_loss) / c_loss * 100
    winner = "SSM" if delta > 0 else "CONCAT"
    
    print(f"Concat: {c_loss:.4f}")
    print(f"SSM: {s_loss:.4f}")
    print(f"Δ: {delta:+.1f}% {winner}")
    
    results = {"concat": c_loss, "ssm": s_loss, "delta": delta, "winner": winner}
    results["summary"] = {"status": winner}
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()