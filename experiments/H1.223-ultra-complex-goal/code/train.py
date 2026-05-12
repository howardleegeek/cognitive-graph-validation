#!/usr/bin/env python3
"""H1.223: Unified architecture on ultra-complex multi-step tasks (100-150 steps) with goal conditioning"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 128}

class Baseline(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(48, hidden), nn.ReLU(), nn.Linear(hidden, D["action"]))
    def forward(self, x): return self.net(x)

class Unified(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.physical = nn.Linear(8, hidden // 4)
        self.semantic = nn.Linear(32, hidden * 3 // 4)
        self.fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, D["action"]))
    def forward(self, x):
        obs = x[:, :8]  
        lang = x[:, 8:40]
        phys = self.physical(obs)
        sem = self.semantic(lang)
        combined = torch.cat([phys, sem], dim=-1)
        return self.fusion(combined)

class UnifiedWithGoal(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.physical = nn.Linear(8, hidden // 4)
        self.semantic = nn.Linear(32, hidden // 4)
        self.goal = nn.Linear(8, hidden // 4)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3 // 4, hidden), nn.ReLU(), nn.Linear(hidden, D["action"]))
    def forward(self, x, goal_state):
        obs = x[:, :8]  
        lang = x[:, 8:40]
        phys = self.physical(obs)
        sem = self.semantic(lang)
        g = self.goal(goal_state)
        combined = torch.cat([phys, sem, g], dim=-1)
        return self.fusion(combined)

def gen_complex_task(n, seq_len, rho=0.85):
    X, Y, goals = [], [], []
    for _ in range(n):
        goal = np.random.randn(8).astype(np.float32) * 0.5
        state = np.random.randn(8).astype(np.float32) * 0.2
        for t in range(seq_len):
            state = state * rho + np.random.randn(8).astype(np.float32) * (1-rho) * 0.3
            lang = np.zeros(32, dtype=np.float32)
            lang[np.random.randint(4)] = 1.0
            x = np.concatenate([state, lang, state + np.random.randn(8)*0.1])
            y = state[:7] + np.random.randn(7)*0.02
            X.append(x)
            Y.append(y)
            goals.append(goal)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(goals, dtype=np.float32)

print("H1.223: Unified Architecture on Ultra-Complex Multi-Step Tasks (100-150 Steps) with Goal Conditioning")
seq_lengths = [100, 120, 150]
results = {"hypothesis": "H1.223", "timestamp": datetime.now().isoformat(), "experiments": []}

for L in seq_lengths:
    X, Y, goals = gen_complex_task(50, L)
    n_train = int(len(X) * 0.8)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]
    goals_train, goals_val = goals[:n_train], goals[n_train:]
    
    for Model, name in [(Baseline, "baseline"), (Unified, "unified"), (UnifiedWithGoal, "unified_goal")]:
        model = Model(D["hidden"])
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        if name == "unified_goal":
            for _ in range(100):
                for i in range(0, len(X_train), 32):
                    idx = slice(i, i+32)
                    opt.zero_grad()
                    if name == "unified_goal":
                        loss = ((model(torch.tensor(X_train[idx]), torch.tensor(goals_train[idx])) - torch.tensor(Y_train[idx]))**2).mean()
                    else:
                        loss = ((model(torch.tensor(X_train[idx])) - torch.tensor(Y_train[idx]))**2).mean()
                    loss.backward()
                    opt.step()
            preds = []
            for i in range(0, len(X_val), 32):
                idx = slice(i, i+32)
                with torch.no_grad():
                    if name == "unified_goal":
                        pred = model(torch.tensor(X_val[idx]), torch.tensor(goals_val[idx])).numpy()
                    else:
                        pred = model(torch.tensor(X_val[idx])).numpy()
                    preds.append(pred)
            mse = ((np.concatenate(preds) - Y_val)**2).mean()
        else:
            for _ in range(100):
                for i in range(0, len(X_train), 32):
                    idx = slice(i, i+32)
                    opt.zero_grad()
                    loss = ((model(torch.tensor(X_train[idx])) - torch.tensor(Y_train[idx]))**2).mean()
                    loss.backward()
                    opt.step()
            preds = []
            for i in range(0, len(X_val), 32):
                idx = slice(i, i+32)
                with torch.no_grad():
                    pred = model(torch.tensor(X_val[idx])).numpy()
                    preds.append(pred)
            mse = ((np.concatenate(preds) - Y_val)**2).mean()
        
        print(f"  Length {L}: {name} MSE = {mse:.6f}")
        results["experiments"].append({"length": L, "model": name, "mse": float(mse)})

baseline_mses = [e["mse"] for e in results["experiments"] if e["model"] == "baseline"]
unified_mses = [e["mse"] for e in results["experiments"] if e["model"] == "unified"]
unified_goal_mses = [e["mse"] for e in results["experiments"] if e["model"] == "unified_goal"]

avg_baseline = np.mean(baseline_mses)
avg_unified = np.mean(unified_mses)
avg_unified_goal = np.mean(unified_goal_mses)

improvement_unified = (avg_baseline - avg_unified) / avg_baseline * 100
improvement_goal = (avg_baseline - avg_unified_goal) / avg_baseline * 100

results["summary"] = {
    "avg_baseline": float(avg_baseline),
    "avg_unified": float(avg_unified),
    "avg_unified_goal": float(avg_unified_goal),
    "unified_improvement": float(improvement_unified),
    "goal_improvement": float(improvement_goal)
}

print(f"\nSummary: Baseline={avg_baseline:.6f}, Unified={avg_unified:.6f} ({improvement_unified:+.1f}%), Unified+Goal={avg_unified_goal:.6f} ({improvement_goal:+.1f}%)")

status = "SUPPORTED" if improvement_unified > 5 or improvement_goal > 5 else "REFUTED"
results["status"] = status
print(f"Status: {status}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to results.json")