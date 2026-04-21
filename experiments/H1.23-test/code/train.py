"""H1.23: Quick dimension scaling test to 64k."""

import torch
import torch.nn as nn
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class UnifiedModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 4, alpha: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(alpha),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(alpha),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return self.net(x)

def generate_batch(n_samples: int, input_dim: int, action_dim: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randn(n_samples, action_dim).astype(np.float32) * 0.1
    return X, y

def train_and_evaluate(input_dim: int, hidden_dim: int, n_samples: int = 200, epochs: int = 300):
    X, y = generate_batch(n_samples, input_dim, 4, seed)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    
    model = UnifiedModel(input_dim, hidden_dim, 4, alpha=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    
    for epoch in range(epochs):
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        final_loss = crit(model(X_t), y_t).item()
    return final_loss

dims_to_test = [
    (12, 4096, "H1.20 reference"),
    (12, 16384, "H1.20 16k"),
    (12, 32768, "H1.20 32k"),
    (12, 65536, "64k test"),
    (12, 131072, "128k test"),
]

print("\n" + "=" * 60)
print("H1.23: Quick Dimension Scaling Test (64k+)")
print("=" * 60)

results = []
for input_dim, hidden_dim, label in dims_to_test:
    print(f"\nTesting {label} ({hidden_dim} hidden)...", flush=True)
    loss = train_and_evaluate(input_dim, hidden_dim, n_samples=200)
    results.append((hidden_dim, loss, label))
    print(f"  MSE: {loss:.6f}")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

results.sort(key=lambda x: x[1])
for hidden_dim, loss, label in results:
    print(f"  {label:>20}: MSE={loss:.6f}")

best = results[0]
print(f"\nBest: {best[2]} with MSE={best[1]:.6f}")