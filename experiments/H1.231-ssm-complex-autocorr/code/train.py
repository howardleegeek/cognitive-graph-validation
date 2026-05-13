import numpy as np
import torch
import torch.nn as nn
import json

class SSMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, state_dim=16):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, state_dim, batch_first=True)
        self.decoder = nn.Linear(state_dim, 1)
        
    def forward(self, x):
        h = torch.relu(self.encoder(x))
        out, _ = self.rnn(h)
        return self.decoder(out).squeeze(-1)

class ConcatPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def generate_complex_autocorr(seq_len, n_samples, rho=0.95, complexity=1.0):
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
ssm_wins = 0

for complexity in [0.3, 0.5, 0.7, 0.9, 1.0]:
    print(f"\n=== Testing complexity={complexity} ===")
    X, y = generate_complex_autocorr(50, 80, 0.95, complexity)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    n_train = 60
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    ssm = SSMPredictor(X.shape[-1])
    concat = ConcatPredictor(X.shape[-1])
    
    opt_s = torch.optim.Adam(ssm.parameters(), lr=0.01)
    opt_c = torch.optim.Adam(concat.parameters(), lr=0.01)
    
    for _ in range(150):
        opt_s.zero_grad()
        loss = ((ssm(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_s.step()
        
        opt_c.zero_grad()
        loss = ((concat(X_train) - y_train) ** 2).mean()
        loss.backward()
        opt_c.step()
    
    with torch.no_grad():
        ssm_loss = ((ssm(X_val) - y_val) ** 2).mean().item()
        concat_loss = ((concat(X_val) - y_val) ** 2).mean().item()
    
    delta = (concat_loss - ssm_loss) / concat_loss * 100
    results[complexity] = {"ssm": ssm_loss, "concat": concat_loss, "delta": delta}
    if delta > 0:
        ssm_wins += 1
    print(f"Complexity={complexity}: SSM={ssm_loss:.4f}, Concat={concat_loss:.4f}, Δ={delta:+.1f}%")

avg_delta = np.mean([r["delta"] for r in results.values()])
print(f"\n=== SUMMARY ===")
print(f"Avg improvement: {avg_delta:+.1f}%")
print(f"SSM wins: {ssm_wins}/5")

status = "SUPPORTED" if avg_delta > 0 and ssm_wins >= 3 else "REFUTED"
print(f"Status: {status}")

with open("results.json", "w") as f:
    json.dump({"results": results, "avg_delta": avg_delta, "ssm_wins": ssm_wins, "status": status}, f, indent=2)