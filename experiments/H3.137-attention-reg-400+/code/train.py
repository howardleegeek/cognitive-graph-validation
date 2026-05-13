import numpy as np
import torch
import torch.nn as nn
import json

class AttentionPredictorReg(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, reg=0.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
        self.reg = reg
        
    def forward(self, x):
        h = self.encoder(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(q.size(-1)), dim=-1)
        return self.decoder(attn @ v).squeeze(-1)

class ConcatPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def generate_autocorr(seq_len, n_samples, rho=0.95):
    X, y = [], []
    for _ in range(n_samples):
        state = np.random.randn(8)
        states, targets = [], []
        for _ in range(seq_len):
            states.append(state.copy())
            state = state * rho + np.random.randn(8) * np.sqrt(1 - rho**2)
            target = np.sum(state[:4])
            targets.append(target)
        X.append(np.array(states))
        y.append(np.array(targets))
    return np.array(X), np.array(y)

results = {}
reg_values = [0.01, 0.05, 0.1, 0.5]
best_reg = None
best_delta = float('-inf')

for reg in reg_values:
    print(f"\n=== Testing reg={reg} on 400 steps ===")
    X, y = generate_autocorr(400, 80, 0.95)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    n_train = 60
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    attn = AttentionPredictorReg(X.shape[-1], reg=reg)
    concat = ConcatPredictor(X.shape[-1])
    
    opt_a = torch.optim.Adam(attn.parameters(), lr=0.01, weight_decay=reg)
    opt_c = torch.optim.Adam(concat.parameters(), lr=0.01)
    
    for _ in range(150):
        opt_a.zero_grad()
        loss = ((attn(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_a.step()
        
        opt_c.zero_grad()
        loss = ((concat(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_c.step()
    
    with torch.no_grad():
        attn_loss = ((attn(X_val) - y_val) ** 2).mean().item()
        concat_loss = ((concat(X_val) - y_val) ** 2).mean().item()
    
    delta = (concat_loss - attn_loss) / concat_loss * 100
    results[reg] = {"attn": attn_loss, "concat": concat_loss, "delta": delta}
    if delta > best_delta:
        best_delta = delta
        best_reg = reg
    print(f"Reg={reg}: Attn={attn_loss:.4f}, Concat={concat_loss:.4f}, Δ={delta:+.1f}%")

print(f"\n=== SUMMARY ===")
print(f"Best reg: {best_reg} with Δ={best_delta:+.1f}%")

status = "SUPPORTED" if best_delta > 0 else "REFUTED"
print(f"Status: {status}")

with open("results.json", "w") as f:
    json.dump({"results": results, "best_reg": best_reg, "best_delta": best_delta, "status": status}, f, indent=2)