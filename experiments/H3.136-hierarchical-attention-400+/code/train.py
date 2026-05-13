import numpy as np
import torch
import torch.nn as nn
import json

class HierarchicalAttentionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, chunk_size=50):
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.chunk_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        batch, seq, dim = x.shape
        h = self.encoder(x)
        
        if seq > self.chunk_size:
            chunks = []
            for i in range(0, seq, self.chunk_size):
                chunk = h[:, i:i+self.chunk_size]
                chunk_avg = self.chunk_encoder(chunk.mean(dim=1, keepdim=True).expand(-1, chunk.size(1), -1))
                chunks.append(chunk_avg)
            h = torch.cat(chunks, dim=1)
            if h.shape[1] > seq:
                h = h[:, :seq, :]
        
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
hier_wins = 0

for seq_len in [400, 450, 500, 550, 600]:
    print(f"\n=== Testing seq_len={seq_len} ===")
    X, y = generate_autocorr(seq_len, 80, 0.95)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    n_train = 60
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    hier = HierarchicalAttentionPredictor(X.shape[-1], chunk_size=50)
    concat = ConcatPredictor(X.shape[-1])
    
    opt_h = torch.optim.Adam(hier.parameters(), lr=0.01)
    opt_c = torch.optim.Adam(concat.parameters(), lr=0.01)
    
    for _ in range(150):
        opt_h.zero_grad()
        loss = ((hier(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_h.step()
        
        opt_c.zero_grad()
        loss = ((concat(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_c.step()
    
    with torch.no_grad():
        hier_loss = ((hier(X_val) - y_val) ** 2).mean().item()
        concat_loss = ((concat(X_val) - y_val) ** 2).mean().item()
    
    delta = (concat_loss - hier_loss) / concat_loss * 100
    results[seq_len] = {"hier": hier_loss, "concat": concat_loss, "delta": delta}
    if delta > 0:
        hier_wins += 1
    print(f"Seq={seq_len}: Hier={hier_loss:.4f}, Concat={concat_loss:.4f}, Δ={delta:+.1f}%")

avg_delta = np.mean([r["delta"] for r in results.values()])
print(f"\n=== SUMMARY ===")
print(f"Avg improvement: {avg_delta:+.1f}%")
print(f"Hierarchical wins: {hier_wins}/5")

status = "SUPPORTED" if avg_delta > 0 and hier_wins >= 3 else "REFUTED"
print(f"Status: {status}")

with open("results.json", "w") as f:
    json.dump({"results": results, "avg_delta": avg_delta, "hier_wins": hier_wins, "status": status}, f, indent=2)