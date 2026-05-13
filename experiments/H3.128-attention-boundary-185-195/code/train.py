import numpy as np
import torch
import torch.nn as nn
import json

class AttentionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(q.size(-1)), dim=-1)
        out = attn @ v
        return self.decoder(out).squeeze(-1)

class ConcatPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
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

def run_exp(seq_len, rho):
    X, y = generate_autocorr(seq_len, 100, rho)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    n_train = 80
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    attn = AttentionPredictor(X.shape[-1])
    concat = ConcatPredictor(X.shape[-1])
    
    opt_a = torch.optim.Adam(attn.parameters(), lr=0.01)
    opt_c = torch.optim.Adam(concat.parameters(), lr=0.01)
    
    for _ in range(200):
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
    
    return attn_loss, concat_loss

results = []
for seq_len in [185, 190, 195]:
    for rho in [0.95, 0.97]:
        attn_l, concat_l = run_exp(seq_len, rho)
        delta = (concat_l - attn_l) / concat_l * 100
        results.append({"seq_len": seq_len, "rho": rho, "attn": attn_l, "concat": concat_l, "delta": delta})
        print(f"Seq={seq_len}, rho={rho}: Δ={delta:+.1f}%")

wins = sum(1 for r in results if r["delta"] > 0)
avg = np.mean([r["delta"] for r in results])
print(f"\n{'SUPPORTED' if wins >= 3 else 'REFUTED'}: {wins}/6 wins, avg={avg:+.1f}%")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)