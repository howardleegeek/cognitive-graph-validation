#!/usr/bin/env python3
"""H3.123: Attention on 80-100 steps with maximum autocorrelation (rho=0.98)

Pushing the boundary further - can attention work on 100 step sequences?
"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 64}

class Concat(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D["obs"] + D["lang"], hidden), nn.ReLU(), nn.Linear(hidden, D["action"]))
    def forward(self, obs, lang): return self.net(torch.cat([obs, lang], dim=-1))

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(D["obs"], hidden); self.k = nn.Linear(D["obs"], hidden); self.v = nn.Linear(D["obs"], hidden)
        self.fc = nn.Linear(hidden, D["action"])
    def forward(self, obs, lang):
        q, k, v = self.q(obs).unsqueeze(0), self.k(obs).unsqueeze(0), self.v(obs).unsqueeze(0)
        attn = torch.softmax((q @ k.T) / np.sqrt(D["obs"]), dim=-1)
        return self.fc((attn @ v).squeeze(0))

def gen_task(n, seq_len, rho):
    X_o, X_l, Y = [], [], []
    for _ in range(n):
        state = np.random.randn(D["obs"]).astype(np.float32) * 0.2
        for _ in range(seq_len):
            state = state * rho + np.random.randn(D["obs"]).astype(np.float32) * (1 - rho) * 0.3
            lang = np.zeros(D["lang"], dtype=np.float32); lang[np.random.randint(4)] = 1.0
            X_o.append(state.copy()); X_l.append(lang.copy()); Y.append((state[:D["action"]] + np.random.randn(D["action"]) * 0.05).copy())
    return np.array(X_o, dtype=np.float32), np.array(X_l, dtype=np.float32), np.array(Y, dtype=np.float32)

print("H3.123: Attention on 80-100 Steps with Maximum Autocorrelation (rho=0.98)")
print("=" * 70)

rho = 0.98
seq_lengths = [80, 85, 90, 95, 100]
results = {"hypothesis": "H3.123", "rho": rho, "timestamp": datetime.now().isoformat(), "experiments": []}

for L in seq_lengths:
    X_o, X_l, Y = gen_task(30, L, rho)
    X_o_val, X_l_val, Y_val = gen_task(10, L, rho)
    
    concat = Concat(64)
    attention = Attention(64)
    
    for model in [concat, attention]:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            idx = np.random.permutation(len(X_o))
            for i in idx[:50]:
                opt.zero_grad()
                loss = nn.MSELoss()(model(torch.from_numpy(X_o[i]), torch.from_numpy(X_l[i])), torch.from_numpy(Y[i]))
                loss.backward()
                opt.step()
    
    concat_losses = [nn.MSELoss()(concat(torch.from_numpy(X_o_val[i]), torch.from_numpy(X_l_val[i])), 
                                torch.from_numpy(Y_val[i])).item() for i in range(len(X_o_val))]
    attn_losses = [nn.MSELoss()(attention(torch.from_numpy(X_o_val[i]), torch.from_numpy(X_l_val[i])), 
                               torch.from_numpy(Y_val[i])).item() for i in range(len(X_o_val))]
    
    concat_mse = np.mean(concat_losses)
    attn_mse = np.mean(attn_losses)
    improvement = (concat_mse - attn_mse) / concat_mse * 100
    
    print(f"  L={L}: Concat={concat_mse:.6f}, Attention={attn_mse:.6f}, Δ={improvement:+.1f}%")
    
    results["experiments"].append({
        "seq_len": L, "concat_mse": float(concat_mse), "attn_mse": float(attn_mse),
        "improvement": float(improvement), "attn_wins": bool(attn_mse < concat_mse)
    })

concat_avg = np.mean([e["concat_mse"] for e in results["experiments"]])
attn_avg = np.mean([e["attn_mse"] for e in results["experiments"]])
avg_improvement = (concat_avg - attn_avg) / concat_avg * 100
attn_wins = sum(1 for e in results["experiments"] if e["attn_wins"])

print(f"\n{'='*70}")
print(f"Overall: Concat avg={concat_avg:.6f}, Attention avg={attn_avg:.6f}")
print(f"Average improvement: {avg_improvement:+.1f}%")
print(f"Attention wins: {attn_wins}/{len(seq_lengths)}")

results["summary"] = {"concat_avg": float(concat_avg), "attn_avg": float(attn_avg),
                      "avg_improvement": float(avg_improvement), "attn_wins": attn_wins, "total_tests": len(seq_lengths)}
results["status"] = "SUPPORTED" if avg_improvement > 5 else ("PARTIAL" if avg_improvement > 0 else "REFUTED")
print(f"\nStatus: {results['status']}")

import os
os.makedirs("experiments/H3.123-attention-80-100-rho098/results", exist_ok=True)
with open("experiments/H3.123-attention-80-100-rho098/results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved.")