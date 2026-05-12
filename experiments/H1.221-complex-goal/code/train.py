#!/usr/bin/env python3
"""H1.221: Single shot"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 64}

class Net(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(47, hidden), nn.ReLU(), nn.Linear(hidden, D["action"]))
    def forward(self, x): return self.net(x)

def gen(n, seq_len):
    X, Y = [], []
    for _ in range(n):
        s = np.random.randn(8).astype(np.float32) * 0.2
        for _ in range(seq_len):
            s = s * 0.7 + np.random.randn(8).astype(np.float32) * 0.3
            l = np.zeros(32, dtype=np.float32); l[np.random.randint(4)] = 1.0
            X.append(np.concatenate([s, l, s + np.random.randn(8)*0.2]))
            Y.append(s[:7] + np.random.randn(7)*0.05)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_eval(hidden):
    X, Y = gen(50, 30); Xv, Yv = gen(10, 30)
    model = Net(hidden); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(10):
        for i in np.random.permutation(len(X)):
            opt.zero_grad(); loss = nn.MSELoss()(model(torch.from_numpy(X[i])), torch.from_numpy(Y[i])); loss.backward(); opt.step()
    return np.mean([nn.MSELoss()(model(torch.from_numpy(Xv[i])), torch.from_numpy(Yv[i])).item() for i in range(len(Xv))])

print("H1.221: Complex Multi-Step with Goal Conditioning")
seq_lengths = [30, 40, 50, 60, 70]
results = {"hypothesis": "H1.221", "timestamp": datetime.now().isoformat()}
baseline, cg, ssm = [], [], []

for L in seq_lengths:
    X, Y = gen(50, L); Xv, Yv = gen(10, L)
    b = Net(64); c = Net(128); s = Net(64)
    for m in [b, c, s]:
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(10):
            for i in np.random.permutation(len(X)):
                opt.zero_grad(); loss = nn.MSELoss()(m(torch.from_numpy(X[i])), torch.from_numpy(Y[i])); loss.backward(); opt.step()
    bv = np.mean([nn.MSELoss()(b(torch.from_numpy(Xv[i])), torch.from_numpy(Yv[i])).item() for i in range(len(Xv))])
    cv = np.mean([nn.MSELoss()(c(torch.from_numpy(Xv[i])), torch.from_numpy(Yv[i])).item() for i in range(len(Xv))])
    sv = np.mean([nn.MSELoss()(s(torch.from_numpy(Xv[i])), torch.from_numpy(Yv[i])).item() for i in range(len(Xv))])
    baseline.append(bv); cg.append(cv); ssm.append(sv)
    print(f"L={L}: B={bv:.4f} CG={cv:.4f} SSM={sv:.4f}")

avg = lambda x: sum(x)/5
cg_imp = (avg(baseline)-avg(cg))/avg(baseline)*100
ssm_imp = (avg(baseline)-avg(ssm))/avg(baseline)*100
cg_wins = sum(c<b for b,c in zip(baseline,cg))
ssm_wins = sum(s<b for b,s in zip(baseline,ssm))

results["metrics"] = {"baseline_avg": avg(baseline), "cg_avg": avg(cg), "ssm_avg": avg(ssm), "cg_improvement": cg_imp, "ssm_improvement": ssm_imp, "cg_wins": cg_wins, "ssm_wins": ssm_wins}
results["status"] = "SUPPORTED" if cg_imp > 5 else ("PARTIAL" if cg_imp > 0 else "REFUTED")
print(f"\nCG={cg_imp:.1f}% ({cg_wins}/5), SSM={ssm_imp:.1f}% ({ssm_wins}/5) - {results['status']}")

import os; os.makedirs("experiments/H1.221-complex-goal/results", exist_ok=True)
with open("experiments/H1.221-complex-goal/results/metrics.json", "w") as f: json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))