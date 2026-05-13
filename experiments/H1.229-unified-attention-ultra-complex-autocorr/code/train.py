import numpy as np
import torch
import torch.nn as nn
import json

class UnifiedAttentionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(q.size(-1)), dim=-1)
        return self.decoder(attn @ v).squeeze(-1)

class UnifiedConcatPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def generate_ultra_complex_autocorr(seq_len, n_samples, rho=0.95, n_steps=5):
    X, y = [], []
    for _ in range(n_samples):
        state = np.random.randn(8)
        states, targets = [], []
        for t in range(seq_len):
            states.append(state.copy())
            for _ in range(n_steps):
                state = state * rho + np.random.randn(8) * np.sqrt(1 - rho**2)
            target = np.sum(state[:4]) + np.sin(t * 0.1) * 0.5
            targets.append(target)
        X.append(np.array(states))
        y.append(np.array(targets))
    return np.array(X), np.array(y)

results = {}
unified_wins = 0
unified_attn_wins = 0

for seq_len in [100, 120, 150, 180, 200]:
    print(f"\n=== Testing seq_len={seq_len} ===")
    X, y = generate_ultra_complex_autocorr(seq_len, 80, 0.95, 5)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    n_train = 60
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    unified_attn = UnifiedAttentionPredictor(X.shape[-1])
    unified_concat = UnifiedConcatPredictor(X.shape[-1])
    
    opt_a = torch.optim.Adam(unified_attn.parameters(), lr=0.01)
    opt_c = torch.optim.Adam(unified_concat.parameters(), lr=0.01)
    
    for _ in range(150):
        opt_a.zero_grad()
        loss = ((unified_attn(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_a.step()
        
        opt_c.zero_grad()
        loss = ((unified_concat(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_c.step()
    
    with torch.no_grad():
        unified_attn_loss = ((unified_attn(X_val) - y_val) ** 2).mean().item()
        unified_concat_loss = ((unified_concat(X_val) - y_val) ** 2).mean().item()
    
    delta_attn = (unified_concat_loss - unified_attn_loss) / unified_concat_loss * 100
    results[seq_len] = {
        "unified_attn": unified_attn_loss, 
        "unified_concat": unified_concat_loss, 
        "delta": delta_attn
    }
    if delta_attn > 0:
        unified_attn_wins += 1
    print(f"Seq={seq_len}: Unified+Attn={unified_attn_loss:.4f}, Unified={unified_concat_loss:.4f}, Δ={delta_attn:+.1f}%")

avg_delta = np.mean([r["delta"] for r in results.values()])
print(f"\n=== SUMMARY ===")
print(f"Avg improvement: {avg_delta:+.1f}%")
print(f"Unified+Attention wins: {unified_attn_wins}/5")

status = "SUPPORTED" if avg_delta > 0 and unified_attn_wins >= 3 else "REFUTED"
print(f"Status: {status}")

with open("results.json", "w") as f:
    json.dump({"results": results, "avg_delta": avg_delta, "unified_attn_wins": unified_attn_wins, "status": status}, f, indent=2)