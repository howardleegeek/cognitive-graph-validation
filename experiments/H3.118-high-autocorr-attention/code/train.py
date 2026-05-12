#!/usr/bin/env python3
"""H3.118: Attention with high autocorrelation (0.9-0.95) on 50-80 steps"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 64}

class Concat(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(48, hidden), nn.ReLU(), nn.Linear(hidden, D["action"]))
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(8, hidden)
        self.k = nn.Linear(8, hidden)
        self.v = nn.Linear(8, hidden)
        self.fc = nn.Linear(hidden, D["action"])
    def forward(self, x):
        obs = x[:8].unsqueeze(0)
        q = self.q(obs)
        k = self.k(obs)
        v = self.v(obs)
        attn = torch.softmax((q @ k.T) / np.sqrt(8), dim=-1)
        out = attn @ v
        return self.fc(out.squeeze(0))

def gen(n, seq_len, rho):
    X, Y = [], []
    for _ in range(n):
        s = np.random.randn(8).astype(np.float32) * 0.2
        for _ in range(seq_len):
            s = s * rho + np.random.randn(8).astype(np.float32) * (1-rho)
            l = np.zeros(32, dtype=np.float32); l[np.random.randint(4)] = 1.0
            X.append(np.concatenate([s, l, s + np.random.randn(8)*0.2]))
            Y.append(s[:7] + np.random.randn(7)*0.05)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("H3.118: Attention with High Autocorrelation (0.9-0.95) on 50-80 Steps")
seq_lengths = [50, 60, 70, 80]
autocorrs = [0.90, 0.93, 0.95]
results = {"hypothesis": "H3.118", "timestamp": datetime.now().isoformat()}

for rho in autocorrs:
    concat_mse, attn_mse = [], []
    for L in seq_lengths:
        X, Y = gen(50, L, rho); Xv, Yv = gen(10, L, rho)
        c = Concat(64); a = Attention(64)
        for m in [c, a]:
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(10):
                for i in np.random.permutation(len(X)):
                    opt.zero_grad(); loss = nn.MSELoss()(m(torch.from_numpy(X[i])), torch.from_numpy(Y[i])); loss.backward(); opt.step()
        cv = np.mean([nn.MSELoss()(c(torch.from_numpy(Xv[i])), torch.from_numpy(Yv[i])).item() for i in range(len(Xv))])
        av = np.mean([nn.MSELoss()(a(torch.from_numpy(Xv[i])), torch.from_numpy(Yv[i])).item() for i in range(len(Xv))])
        concat_mse.append(cv); attn_mse.append(av)
        print(f"rho={rho}, L={L}: Concat={cv:.5f} Attn={av:.5f}")
    
    avg_c = sum(concat_mse)/len(concat_mse)
    avg_a = sum(attn_mse)/len(attn_mse)
    imp = (avg_c - avg_a) / avg_c * 100
    wins = sum(a<c for c,a in zip(concat_mse, attn_mse))
    print(f"rho={rho}: Concat avg={avg_c:.5f}, Attn avg={avg_a:.5f}, Improvement={imp:.1f}%, Wins={wins}/4")

avg_imp = sum((sum(concat_mse)/4 - sum(attn_mse)/4) / (sum(concat_mse)/4) * 100 for concat_mse, attn_mse in [(concat_mse, attn_mse)]) / len(autocorrs)
results["metrics"] = {"autocorrs_tested": autocorrs, "avg_improvement": float(avg_imp)}
results["status"] = "SUPPORTED" if avg_imp > 5 else ("PARTIAL" if avg_imp > 0 else "REFUTED")
print(f"\nOverall: {avg_imp:.1f}% - {results['status']}")

import os
os.makedirs("experiments/H3.118-high-autocorr-attention/results", exist_ok=True)
with open("experiments/H3.118-high-autocorr-attention/results/metrics.json", "w") as f: json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))