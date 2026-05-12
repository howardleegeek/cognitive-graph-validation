#!/usr/bin/env python3
"""H1.222: Ultra-complex multi-step (80-120 steps) with goal conditioning"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 64}

class Baseline(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(56, hidden), nn.ReLU(), nn.Linear(hidden, D["action"]))
    def forward(self, x): return self.net(x)

class Unified(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.physical = nn.Sequential(nn.Linear(8, hidden//2), nn.ReLU())
        self.semantic = nn.Sequential(nn.Linear(32, hidden//2), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden+8, hidden), nn.ReLU(), nn.Linear(hidden, D["action"]))
    def forward(self, x):
        if x.dim() == 1:
            obs = x[:8]; lang = x[8:40]; goal = x[40:48]; curr = x[48:56]
        else:
            obs = x[:, :8]; lang = x[:, 8:40]; goal = x[:, 40:48]; curr = x[:, 48:56]
        fused = torch.cat([self.physical(obs), self.semantic(lang)], dim=-1)
        return self.fusion(torch.cat([fused, goal], dim=-1))

def gen(n, seq_len, rho=0.85):
    X, Y = [], []
    for _ in range(n):
        s = np.random.randn(8).astype(np.float32) * 0.2
        goal = s.copy()
        for _ in range(seq_len):
            s = s * rho + np.random.randn(8).astype(np.float32) * (1-rho)
            l = np.zeros(32, dtype=np.float32); l[np.random.randint(4)] = 1.0
            X.append(np.concatenate([s, l, goal, s + np.random.randn(8)*0.2]))
            Y.append(s[:7] + np.random.randn(7)*0.05)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("H1.222: Ultra-Complex Multi-Step (80-120 Steps) with Goal Conditioning")
seq_lengths = [80, 90, 100, 110, 120]
results = {"hypothesis": "H1.222", "timestamp": datetime.now().isoformat()}
baseline, unified = [], []

for L in seq_lengths:
    X, Y = gen(50, L); Xv, Yv = gen(10, L)
    b = Baseline(64); u = Unified(128)
    for m in [b, u]:
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(10):
            for i in np.random.permutation(len(X)):
                opt.zero_grad(); loss = nn.MSELoss()(m(torch.from_numpy(X[i])), torch.from_numpy(Y[i])); loss.backward(); opt.step()
    bv = np.mean([nn.MSELoss()(b(torch.from_numpy(Xv[i])), torch.from_numpy(Yv[i])).item() for i in range(len(Xv))])
    uv = np.mean([nn.MSELoss()(u(torch.from_numpy(Xv[i])), torch.from_numpy(Yv[i])).item() for i in range(len(Xv))])
    baseline.append(bv); unified.append(uv)
    print(f"L={L}: Baseline={bv:.5f} Unified={uv:.5f}")

avg = lambda x: sum(x)/len(x)
unified_imp = (avg(baseline)-avg(unified))/avg(baseline)*100
unified_wins = sum(u<b for b,u in zip(baseline,unified))

results["metrics"] = {"baseline_avg": float(avg(baseline)), "unified_avg": float(avg(unified)), "unified_improvement": float(unified_imp), "unified_wins": int(unified_wins)}
results["status"] = "SUPPORTED" if unified_imp > 5 else ("PARTIAL" if unified_imp > 0 else "REFUTED")
print(f"\nUnified={unified_imp:.1f}% ({unified_wins}/5) - {results['status']}")

import os
os.makedirs("experiments/H1.222-ultra-complex-goal/results", exist_ok=True)
with open("experiments/H1.222-ultra-complex-goal/results/metrics.json", "w") as f: json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))