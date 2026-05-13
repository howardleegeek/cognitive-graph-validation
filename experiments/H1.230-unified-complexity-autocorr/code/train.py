import numpy as np
import torch
import torch.nn as nn
import json

class UnifiedPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.physical = nn.Sequential(nn.Linear(input_dim // 2, hidden_dim // 2), nn.ReLU())
        self.semantic = nn.Sequential(nn.Linear(input_dim // 2, hidden_dim // 2), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        mid = x.shape[-1] // 2
        phys = self.physical(x[..., :mid])
        sem = self.semantic(x[..., mid:])
        return self.fusion(torch.cat([phys, sem], dim=-1)).squeeze(-1)

class BaselinePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def generate_complexity_autocorr(seq_len, n_samples, rho=0.95, complexity=1.0):
    X, y = [], []
    for _ in range(n_samples):
        state = np.random.randn(8)
        states, targets = [], []
        for t in range(seq_len):
            states.append(state.copy())
            state = state * rho + np.random.randn(8) * np.sqrt(1 - rho**2)
            target = np.sum(state[:4])
            if complexity > 0.5:
                target += np.sin(t * 0.1) * complexity
            if complexity > 0.8:
                target += np.cos(t * 0.05) * (complexity - 0.8) * 2
            targets.append(target)
        X.append(np.array(states))
        y.append(np.array(targets))
    return np.array(X), np.array(y)

results = {}
unified_wins = 0

complexities = [0.3, 0.5, 0.7, 0.9, 1.0]
for complexity in complexities:
    print(f"\n=== Testing complexity={complexity} ===")
    X, y = generate_complexity_autocorr(50, 80, 0.95, complexity)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    n_train = 60
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    unified = UnifiedPredictor(X.shape[-1])
    baseline = BaselinePredictor(X.shape[-1])
    
    opt_u = torch.optim.Adam(unified.parameters(), lr=0.01)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=0.01)
    
    for _ in range(150):
        opt_u.zero_grad()
        loss = ((unified(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_u.step()
        
        opt_b.zero_grad()
        loss = ((baseline(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_b.step()
    
    with torch.no_grad():
        unified_loss = ((unified(X_val) - y_val) ** 2).mean().item()
        baseline_loss = ((baseline(X_val) - y_val) ** 2).mean().item()
    
    delta = (baseline_loss - unified_loss) / baseline_loss * 100
    results[complexity] = {"unified": unified_loss, "baseline": baseline_loss, "delta": delta}
    if delta > 0:
        unified_wins += 1
    print(f"Complexity={complexity}: Unified={unified_loss:.4f}, Baseline={baseline_loss:.4f}, Δ={delta:+.1f}%")

avg_delta = np.mean([r["delta"] for r in results.values()])
print(f"\n=== SUMMARY ===")
print(f"Avg improvement: {avg_delta:+.1f}%")
print(f"Unified wins: {unified_wins}/5")

status = "SUPPORTED" if avg_delta > 0 and unified_wins >= 3 else "REFUTED"
print(f"Status: {status}")

with open("results.json", "w") as f:
    json.dump({"results": results, "avg_delta": avg_delta, "unified_wins": unified_wins, "status": status}, f, indent=2)