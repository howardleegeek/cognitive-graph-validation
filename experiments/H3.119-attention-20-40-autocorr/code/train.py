#!/usr/bin/env python3
"""H3.119: Attention on 20-40 steps WITH autocorrelation (combining H3.117's finding)"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 64, "seq": 10}

class ConcatModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(D["obs"] + D["lang"], hidden),
            nn.ReLU(),
            nn.Linear(hidden, D["action"])
        )
    def forward(self, phys, sem):
        x = torch.cat([phys, sem], dim=-1)
        x = x.mean(dim=1)
        return self.fc(x)

class AttentionModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        total_dim = D["obs"] + D["lang"]
        self.qkv = nn.Linear(total_dim, total_dim * 3)
        self.proj = nn.Linear(total_dim, total_dim)
        self.fc = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D["action"])
        )
    def forward(self, phys, sem):
        B, T, D = phys.shape
        h = torch.cat([phys, sem], dim=-1)
        qkv = self.qkv(h)
        qkv = qkv.view(B, T, 3, -1)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn = torch.softmax(attn, dim=-2)
        h = torch.matmul(attn, v)
        h = h.mean(dim=1)
        return self.fc(h)

def gen_with_autocorr(n, seq_len, rho):
    X_phys, X_sem, Y = [], [], []
    for _ in range(n):
        state = np.random.randn(seq_len, D["obs"]).astype(np.float32) * 0.2
        for t in range(1, seq_len):
            state[t] = state[t-1] * rho + state[t] * (1-rho) * 0.3
        lang = np.zeros((seq_len, D["lang"]), dtype=np.float32)
        for t in range(seq_len):
            lang[t, np.random.randint(4)] = 1.0
        y = state[:, :D["action"]] + np.random.randn(seq_len, D["action"]) * 0.02
        X_phys.append(state)
        X_sem.append(lang)
        Y.append(y)
    return np.array(X_phys), np.array(X_sem), np.array(Y)

print("H3.119: Attention on 20-40 Steps WITH Autocorrelation")
seq_lengths = [20, 25, 30, 35, 40]
autocorrs = [0.85, 0.90, 0.93]
results = {"hypothesis": "H3.119", "timestamp": datetime.now().isoformat(), "experiments": []}

for rho in autocorrs:
    for L in seq_lengths:
        X_phys, X_sem, Y = gen_with_autocorr(50, L, rho)
        n_train = int(len(X_phys) * 0.8)
        X_phys_train, X_sem_train, Y_train = X_phys[:n_train], X_sem[:n_train], Y[:n_train]
        X_phys_val, X_sem_val, Y_val = X_phys[n_train:], X_sem[n_train:], Y[n_train:]
        
        for Model, name in [(ConcatModel, "concat"), (AttentionModel, "attention")]:
            model = Model(D["hidden"])
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(50):
                indices = np.random.permutation(len(X_phys_train))
                for i in range(0, len(indices), 8):
                    batch_idx = indices[i:i+8]
                    phys = torch.tensor(X_phys_train[batch_idx])
                    sem = torch.tensor(X_sem_train[batch_idx])
                    y = torch.tensor(Y_train[batch_idx])
                    opt.zero_grad()
                    pred = model(phys, sem)
                    loss = ((pred - y.mean(dim=1))**2).mean()
                    loss.backward()
                    opt.step()
            
            with torch.no_grad():
                phys = torch.tensor(X_phys_val)
                sem = torch.tensor(X_sem_val)
                pred = model(phys, sem)
                mse = ((pred - Y_val.mean(axis=1))**2).mean().item()
            
            print(f"  rho={rho}, L={L}: {name} MSE = {mse:.6f}")
            results["experiments"].append({"rho": rho, "length": L, "model": name, "mse": float(mse)})

concat_mses = {L: [] for L in seq_lengths}
attn_mses = {L: [] for L in seq_lengths}

for e in results["experiments"]:
    if e["model"] == "concat":
        concat_mses[e["length"]].append(e["mse"])
    else:
        attn_mses[e["length"]].append(e["mse"])

improvements = []
for L in seq_lengths:
    avg_concat = np.mean(concat_mses[L])
    avg_attn = np.mean(attn_mses[L])
    improvement = (avg_concat - avg_attn) / avg_concat * 100
    improvements.append(improvement)
    print(f"Length {L}: Concat={avg_concat:.6f}, Attn={avg_attn:.6f}, Δ={improvement:+.1f}%")

avg_improvement = np.mean(improvements)
attn_wins = sum(1 for imp in improvements if imp > 0)

results["summary"] = {
    "avg_improvement": float(avg_improvement),
    "attn_wins": attn_wins,
    "total_tests": len(seq_lengths)
}

status = "SUPPORTED" if avg_improvement > 5 and attn_wins >= 3 else "REFUTED"
if avg_improvement > 0 and attn_wins >= 2:
    status = "PARTIAL"
results["status"] = status

print(f"\nSummary: Avg improvement={avg_improvement:+.1f}%, Attention wins {attn_wins}/{len(seq_lengths)}")
print(f"Status: {status}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to results.json")