#!/usr/bin/env python3
"""H3.32: SSM Validation on Continuous Control"""

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


def generate_continuous_control_data(n_samples=500, seq_len=20, s_dim=16, a_dim=7):
    """Generate continuous control dynamics data"""
    states = []
    actions = []
    next_states = []
    
    for _ in range(n_samples):
        s = np.random.randn(seq_len, s_dim) * 0.1
        a = np.random.randn(seq_len, a_dim)
        
        ns = np.zeros_like(s)
        for t in range(seq_len - 1):
            action_effect = np.dot(a[t], np.random.randn(a_dim, s_dim)[:s_dim]) * 0.1
            ns[t] = s[t] + action_effect + np.random.randn(s_dim) * 0.01
        ns[-1] = s[-1]
        
        states.append(s[:-1])
        actions.append(a[:-1])
        next_states.append(ns[1:])
    
    return states, actions, next_states


def train_and_evaluate(model, X_s_train, X_a_train, Y_train, X_s_test, X_a_test, Y_test, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    
    for e in range(epochs):
        model.train()
        for i in range(0, len(X_s_train), 32):
            s = X_s_train[i:i+32]
            a = X_a_train[i:i+32]
            y = Y_train[i:i+32]
            loss = crit(model(s, a), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        loss = crit(model(X_s_test, X_a_test), Y_test).item()
    return loss


def main():
    print("H3.32: SSM Validation on Continuous Control")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    s_dim, a_dim = 16, 7
    results_all = []
    
    for seq_len in [10, 20, 30]:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        states, actions, next_states = generate_continuous_control_data(n_samples=500, seq_len=seq_len, s_dim=s_dim, a_dim=a_dim)
        
        X_s = torch.tensor(np.concatenate(states), dtype=torch.float32)
        X_a = torch.tensor(np.concatenate(actions), dtype=torch.float32)
        Y = torch.tensor(np.concatenate(next_states), dtype=torch.float32)
        
        idx = int(0.8 * len(X_s))
        X_s_train, X_s_test = X_s[:idx], X_s[idx:]
        X_a_train, X_a_test = X_a[:idx], X_a[idx:]
        Y_train, Y_test = Y[:idx], Y[idx:]
        
        c_model = ConcatFusion(s_dim, a_dim, 256)
        c_loss = train_and_evaluate(c_model, X_s_train, X_a_train, Y_train, X_s_test, X_a_test, Y_test)
        
        s_model = SSMFusion(s_dim, a_dim, 256, d_state=16)
        s_loss = train_and_evaluate(s_model, X_s_train, X_a_train, Y_train, X_s_test, X_a_test, Y_test)
        
        delta_ssm = (c_loss - s_loss) / c_loss * 100
        
        print(f"Concat: {c_loss:.4f}")
        print(f"SSM:    {s_loss:.4f} (Δ={delta_ssm:+.1f}%)")
        
        results_all.append({
            "seq_len": seq_len,
            "concat": c_loss,
            "ssm": s_loss,
            "ssm_delta": delta_ssm
        })
    
    avg_ssm = np.mean([r["ssm_delta"] for r in results_all])
    
    print(f"\n=== Summary ===")
    print(f"SSM avg Δ: {avg_ssm:+.1f}%")
    
    winner = "SSM" if avg_ssm > 0 else "CONCAT"
    
    results = {
        "experiments": results_all,
        "avg_ssm_delta": avg_ssm,
        "winner": winner,
        "status": "SUPPORTED" if avg_ssm > 10 else "REFUTED" if avg_ssm < -10 else "INCONCLUSIVE"
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()