#!/usr/bin/env python3
"""H3.121: Attention on 40-60 steps with extreme autocorrelation (0.95-0.98)

Key insight from H3.117: Autocorrelation unlocks attention in the 30-50 step "death zone"
This test pushes to even higher autocorrelation to find the ceiling.
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

print("H3.121: Attention on 40-60 Steps with Extreme Autocorrelation (0.95-0.98)")
print("=" * 70)

rhos = [0.95, 0.96, 0.97, 0.98]
seq_lengths = [40, 45, 50, 55, 60]
results = {"hypothesis": "H3.121", "timestamp": datetime.now().isoformat(), "experiments": []}

for rho in rhos:
    print(f"\n=== Autocorrelation rho={rho} ===")
    rho_results = []
    
    for L in seq_lengths:
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
        
        print(f"  rho={rho}, L={L}: Concat={concat_mse:.6f}, Attn={attn_mse:.6f}, Δ={improvement:+.1f}%")
        
        rho_results.append({
            "rho": rho,
            "seq_len": L,
            "concat_mse": float(concat_mse),
            "attn_mse": float(attn_mse),
            "improvement": float(improvement),
            "attn_wins": bool(attn_mse < concat_mse)
        })
    
    results["experiments"].extend(rho_results)
    
    rho_concat_avg = np.mean([r["concat_mse"] for r in rho_results])
    rho_attn_avg = np.mean([r["attn_mse"] for r in rho_results])
    rho_imp = (rho_concat_avg - rho_attn_avg) / rho_concat_avg * 100
    rho_wins = sum(1 for r in rho_results if r["attn_wins"])
    print(f"  rho={rho} avg: Concat={rho_concat_avg:.6f}, Attn={rho_attn_avg:.6f}, Δ={rho_imp:+.1f}%, Wins={rho_wins}/5")

all_concat = [e["concat_mse"] for e in results["experiments"]]
all_attn = [e["attn_mse"] for e in results["experiments"]]
avg_improvement = (np.mean(all_concat) - np.mean(all_attn)) / np.mean(all_concat) * 100
attn_wins = sum(1 for e in results["experiments"] if e["attn_wins"])

print(f"\n{'='*70}")
print(f"Overall: Concat avg={np.mean(all_concat):.6f}, Attention avg={np.mean(all_attn):.6f}")
print(f"Average improvement: {avg_improvement:+.1f}%")
print(f"Attention wins: {attn_wins}/{len(results['experiments'])}")

results["summary"] = {
    "concat_avg": float(np.mean(all_concat)),
    "attn_avg": float(np.mean(all_attn)),
    "avg_improvement": float(avg_improvement),
    "attn_wins": attn_wins,
    "total_tests": len(results["experiments"])
}

if avg_improvement > 5:
    results["status"] = "SUPPORTED"
elif avg_improvement > 0:
    results["status"] = "PARTIAL"
else:
    results["status"] = "REFUTED"

results["note"] = "Testing extreme autocorrelation (0.95-0.98) on 40-60 step sequences"

print(f"\nStatus: {results['status']}")

import os
os.makedirs("experiments/H3.121-attention-extreme-rho/results", exist_ok=True)
with open("experiments/H3.121-attention-extreme-rho/results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to experiments/H3.121-attention-extreme-rho/results/metrics.json")