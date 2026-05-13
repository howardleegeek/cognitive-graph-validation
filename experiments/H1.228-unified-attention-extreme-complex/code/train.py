import numpy as np
import torch
import torch.nn as nn
import json

class UnifiedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.phys = nn.Sequential(nn.Linear(input_dim//2, hidden_dim//4), nn.ReLU())
        self.sem = nn.Sequential(nn.Linear(input_dim//2, hidden_dim*3//4), nn.ReLU())
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h = torch.cat([self.phys(x[..., :x.shape[-1]//2]), self.sem(x[..., x.shape[-1]//2:])], dim=-1)
        q, k, v = self.q(h), self.k(h), self.v(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(q.size(-1)), dim=-1)
        return self.out(attn @ v).squeeze(-1)

class Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def gen_data(seq_len, n_samples):
    X, y = [], []
    for _ in range(n_samples):
        state = np.random.randn(16)
        states, targets = [], []
        for _ in range(seq_len):
            states.append(state.copy())
            state = state * 0.98 + np.random.randn(16) * 0.1
            targets.append(np.sum(state[:8]))
        X.append(np.array(states))
        y.append(np.array(targets))
    return np.array(X), np.array(y)

def run_exp(seq_len):
    X, y = gen_data(seq_len, 100)
    X_t, y_t = torch.FloatTensor(X[:80]), torch.FloatTensor(y[:80])
    X_v, y_v = torch.FloatTensor(X[80:]), torch.FloatTensor(y[80:])
    
    unified = UnifiedAttention(X.shape[-1])
    baseline = Baseline(X.shape[-1])
    
    opt_u = torch.optim.Adam(unified.parameters(), lr=0.01)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=0.01)
    
    for _ in range(200):
        opt_u.zero_grad()
        loss = ((unified(X_t) - y_t) ** 2).mean()
        loss.backward()
        opt_u.step()
        
        opt_b.zero_grad()
        loss = ((baseline(X_t) - y_t) ** 2).mean()
        loss.backward()
        opt_b.step()
    
    with torch.no_grad():
        u_loss = ((unified(X_v) - y_v) ** 2).mean().item()
        b_loss = ((baseline(X_v) - y_v) ** 2).mean().item()
    
    return u_loss, b_loss

results = []
for seq_len in [100, 150, 200]:
    u_l, b_l = run_exp(seq_len)
    delta = (b_l - u_l) / b_l * 100
    results.append({"seq_len": seq_len, "unified": u_l, "baseline": b_l, "delta": delta})
    print(f"Seq={seq_len}: Unified={u_l:.4f}, Baseline={b_l:.4f}, Δ={delta:+.1f}%")

wins = sum(1 for r in results if r["delta"] > 0)
avg = np.mean([r["delta"] for r in results])
print(f"\n{'SUPPORTED' if wins >= 2 else 'REFUTED'}: {wins}/3 wins, avg={avg:+.1f}%")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)