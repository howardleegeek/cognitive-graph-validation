#!/usr/bin/env python3
"""H3.120: Attention on 20-40 steps with optimal autocorrelation (rho=0.93)

Key insight from H3.118: rho=0.93 is optimal (+14.7% improvement)
This test focuses on the 20-40 step range where attention previously failed.
"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 64}

class Concat(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D["obs"] + D["lang"], hidden),
            nn.ReLU(),
            nn.Linear(hidden, D["action"])
        )
    def forward(self, obs, lang):
        return self.net(torch.cat([obs, lang], dim=-1))

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(D["obs"], hidden)
        self.k = nn.Linear(D["obs"], hidden)
        self.v = nn.Linear(D["obs"], hidden)
        self.fc = nn.Linear(hidden, D["action"])
    def forward(self, obs, lang):
        q = self.q(obs).unsqueeze(0)
        k = self.k(obs).unsqueeze(0)
        v = self.v(obs).unsqueeze(0)
        attn = torch.softmax((q @ k.T) / np.sqrt(D["obs"]), dim=-1)
        out = attn @ v
        return self.fc(out.squeeze(0))

def gen_task(n, seq_len, rho):
    X_obs, X_lang, Y = [], [], []
    for _ in range(n):
        state = np.random.randn(D["obs"]).astype(np.float32) * 0.2
        for _ in range(seq_len):
            state = state * rho + np.random.randn(D["obs"]).astype(np.float32) * (1 - rho) * 0.3
            
            lang = np.zeros(D["lang"], dtype=np.float32)
            lang[np.random.randint(min(4, D["lang"]))] = 1.0
            
            action = state[:D["action"]] + np.random.randn(D["action"]) * 0.05
            
            X_obs.append(state.copy())
            X_lang.append(lang.copy())
            Y.append(action.copy())
    
    return (np.array(X_obs, dtype=np.float32), np.array(X_lang, dtype=np.float32), 
            np.array(Y, dtype=np.float32))

print("H3.120: Attention on 20-40 Steps with Optimal Autocorrelation (rho=0.93)")
print("=" * 70)

rho = 0.93  # Optimal from H3.118
seq_lengths = [20, 25, 30, 35, 40]
results = {"hypothesis": "H3.120", "rho": rho, "timestamp": datetime.now().isoformat(), "experiments": []}

for L in seq_lengths:
    print(f"\n--- Sequence Length: {L} steps ---")
    
    X_o, X_l, Y = gen_task(30, L, rho)  # Reduced from 60
    X_o_val, X_l_val, Y_val = gen_task(10, L, rho)  # Reduced from 15
    
    concat = Concat(64)
    attention = Attention(64)
    
    for model, name in [(concat, "Concat"), (attention, "Attention")]:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):  # Reduced from 12
            idx = np.random.permutation(len(X_o))
            for i in idx[:50]:  # Reduced from full batch
                opt.zero_grad()
                loss = nn.MSELoss()(model(torch.from_numpy(X_o[i]), torch.from_numpy(X_l[i])), 
                                   torch.from_numpy(Y[i]))
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
        "seq_len": L,
        "concat_mse": float(concat_mse),
        "attn_mse": float(attn_mse),
        "improvement": float(improvement),
        "attn_wins": bool(attn_mse < concat_mse)
    })

concat_avg = np.mean([e["concat_mse"] for e in results["experiments"]])
attn_avg = np.mean([e["attn_mse"] for e in results["experiments"]])
avg_improvement = (concat_avg - attn_avg) / concat_avg * 100
attn_wins = sum(1 for e in results["experiments"] if e["attn_wins"])

print(f"\n{'='*70}")
print(f"Overall: Concat avg={concat_avg:.6f}, Attention avg={attn_avg:.6f}")
print(f"Average improvement: {avg_improvement:+.1f}%")
print(f"Attention wins: {attn_wins}/{len(seq_lengths)}")

results["summary"] = {
    "concat_avg": float(concat_avg),
    "attn_avg": float(attn_avg),
    "avg_improvement": float(avg_improvement),
    "attn_wins": attn_wins,
    "total_tests": len(seq_lengths)
}

if avg_improvement > 5:
    results["status"] = "SUPPORTED"
elif avg_improvement > 0:
    results["status"] = "PARTIAL"
else:
    results["status"] = "REFUTED"

results["note"] = "Testing optimal rho=0.93 on 20-40 step range"

print(f"\nStatus: {results['status']}")

import os
os.makedirs("experiments/H3.120-attention-optimal-rho/results", exist_ok=True)
with open("experiments/H3.120-attention-optimal-rho/results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to experiments/H3.120-attention-optimal-rho/results/metrics.json")