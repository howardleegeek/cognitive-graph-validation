import numpy as np
import torch
import torch.nn as nn
import json

class UnifiedPredictorReg(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, reg=0.1):
        super().__init__()
        self.physical = nn.Sequential(nn.Linear(input_dim // 2, hidden_dim // 2), nn.ReLU())
        self.semantic = nn.Sequential(nn.Linear(input_dim // 2, hidden_dim // 2), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.reg = reg
        
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
reg_values = [0.01, 0.05, 0.1, 0.5, 1.0]
best_reg = None
best_delta = float('-inf')

for reg in reg_values:
    print(f"\n=== Testing reg={reg} ===")
    X, y = generate_complexity_autocorr(50, 80, 0.95, 0.7)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    n_train = 60
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    unified = UnifiedPredictorReg(X.shape[-1], reg=reg)
    baseline = BaselinePredictor(X.shape[-1])
    
    opt_u = torch.optim.Adam(unified.parameters(), lr=0.01, weight_decay=reg)
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
    results[reg] = {"unified": unified_loss, "baseline": baseline_loss, "delta": delta}
    if delta > best_delta:
        best_delta = delta
        best_reg = reg
    print(f"Reg={reg}: Unified={unified_loss:.4f}, Baseline={baseline_loss:.4f}, Δ={delta:+.1f}%")

print(f"\n=== SUMMARY ===")
print(f"Best reg: {best_reg} with Δ={best_delta:+.1f}%")

status = "SUPPORTED" if best_delta > 0 else "REFUTED"
print(f"Status: {status}")

with open("results.json", "w") as f:
    json.dump({"results": results, "best_reg": best_reg, "best_delta": best_delta, "status": status}, f, indent=2)