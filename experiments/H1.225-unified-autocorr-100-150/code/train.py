#!/usr/bin/env python3
"""H1.225: Unified architecture with autocorrelation injection on 100-150 step sequences

Key insight: Adding autocorrelation to training data might enable
unified architecture to work better on ultra-long sequences.
"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 128}

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D["obs"] + D["lang"], D["hidden"]), nn.ReLU(),
                                 nn.Linear(D["hidden"], D["hidden"]), nn.ReLU(), nn.Linear(D["hidden"], D["action"]))
    def forward(self, obs, lang): return self.net(torch.cat([obs, lang], dim=-1))

class Unified(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(D["obs"] + D["lang"], D["hidden"]), nn.ReLU(),
                                    nn.Linear(D["hidden"], D["hidden"]), nn.ReLU())
        self.physical = nn.Sequential(nn.Linear(D["hidden"], D["hidden"]//2), nn.ReLU())
        self.semantic = nn.Sequential(nn.Linear(D["hidden"], D["hidden"]//2), nn.ReLU())
        self.output = nn.Linear(D["hidden"], D["action"])
    def forward(self, obs, lang):
        fused = self.fusion(torch.cat([obs, lang], dim=-1))
        return self.output(torch.cat([self.physical(fused), self.semantic(fused)], dim=-1))

def gen_task(n, seq_len, rho):
    X_o, X_l, Y = [], [], []
    for _ in range(n):
        state = np.random.randn(D["obs"]).astype(np.float32) * 0.2
        for _ in range(seq_len):
            state = state * rho + np.random.randn(D["obs"]).astype(np.float32) * (1 - rho) * 0.3
            lang = np.zeros(D["lang"], dtype=np.float32); lang[np.random.randint(4)] = 1.0
            X_o.append(state.copy()); X_l.append(lang.copy()); Y.append((state[:D["action"]] + np.random.randn(D["action"]) * 0.05).copy())
    return np.array(X_o, dtype=np.float32), np.array(X_l, dtype=np.float32), np.array(Y, dtype=np.float32)

print("H1.225: Unified Architecture with Autocorrelation on 100-150 Steps")
print("=" * 70)

rho = 0.95  # High autocorrelation from H3.121 success
seq_lengths = [100, 110, 120, 130, 140, 150]
results = {"hypothesis": "H1.225", "rho": rho, "timestamp": datetime.now().isoformat(), "experiments": []}

for L in seq_lengths:
    X_o, X_l, Y = gen_task(30, L, rho)
    X_o_val, X_l_val, Y_val = gen_task(10, L, rho)
    
    baseline = Baseline()
    unified = Unified()
    
    for model in [baseline, unified]:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            idx = np.random.permutation(len(X_o))
            for i in idx[:50]:
                opt.zero_grad()
                loss = nn.MSELoss()(model(torch.from_numpy(X_o[i]), torch.from_numpy(X_l[i])), torch.from_numpy(Y[i]))
                loss.backward()
                opt.step()
    
    baseline_losses = [nn.MSELoss()(baseline(torch.from_numpy(X_o_val[i]), torch.from_numpy(X_l_val[i])), 
                                torch.from_numpy(Y_val[i])).item() for i in range(len(X_o_val))]
    unified_losses = [nn.MSELoss()(unified(torch.from_numpy(X_o_val[i]), torch.from_numpy(X_l_val[i])), 
                               torch.from_numpy(Y_val[i])).item() for i in range(len(X_o_val))]
    
    baseline_mse = np.mean(baseline_losses)
    unified_mse = np.mean(unified_losses)
    improvement = (baseline_mse - unified_mse) / baseline_mse * 100
    
    print(f"  L={L}: Baseline={baseline_mse:.6f}, Unified={unified_mse:.6f}, Δ={improvement:+.1f}%")
    
    results["experiments"].append({
        "seq_len": L, "baseline_mse": float(baseline_mse), "unified_mse": float(unified_mse),
        "improvement": float(improvement), "unified_wins": bool(unified_mse < baseline_mse)
    })

baseline_avg = np.mean([e["baseline_mse"] for e in results["experiments"]])
unified_avg = np.mean([e["unified_mse"] for e in results["experiments"]])
avg_improvement = (baseline_avg - unified_avg) / baseline_avg * 100
unified_wins = sum(1 for e in results["experiments"] if e["unified_wins"])

print(f"\n{'='*70}")
print(f"Overall: Baseline avg={baseline_avg:.6f}, Unified avg={unified_avg:.6f}")
print(f"Average improvement: {avg_improvement:+.1f}%")
print(f"Unified wins: {unified_wins}/{len(seq_lengths)}")

results["summary"] = {"baseline_avg": float(baseline_avg), "unified_avg": float(unified_avg),
                      "avg_improvement": float(avg_improvement), "unified_wins": unified_wins, "total_tests": len(seq_lengths)}
results["status"] = "SUPPORTED" if avg_improvement > 5 else ("PARTIAL" if avg_improvement > 0 else "REFUTED")
print(f"\nStatus: {results['status']}")

import os
os.makedirs("experiments/H1.225-unified-autocorr-100-150/results", exist_ok=True)
with open("experiments/H1.225-unified-autocorr-100-150/results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved.")