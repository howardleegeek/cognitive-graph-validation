"""
H3.22: SSM Dimension Scaling Test (Fast version)
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SSMBlock(nn.Module):
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.x_proj = nn.Linear(dim, state_dim)
        self.conv = nn.Conv1d(state_dim, state_dim, kernel_size=3, padding=1)
        self.gate = nn.Linear(state_dim, dim)
        self.norm = nn.LayerNorm(state_dim)
        self.act = nn.SiLU()
        
    def forward(self, x):
        s = self.x_proj(x)
        s = s.transpose(1, 2)
        s = self.conv(s).transpose(1, 2)
        s = self.norm(s)
        s = self.act(s)
        return x * torch.sigmoid(self.gate(s))

class SSMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ssm_blocks = nn.ModuleList([SSMBlock(hidden_dim, state_dim) for _ in range(2)])
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.ssm_blocks:
            h = block(h)
        return self.output_proj(h[:, -1, :])

def generate_data(n_samples, seq_len, input_dim):
    np.random.seed(42)
    X, Y = [], []
    for _ in range(n_samples):
        x = torch.randn(seq_len, input_dim) * 0.1
        for i in range(1, seq_len):
            x[i] += x[i-1] * 0.5 + torch.randn(input_dim) * 0.05
        y = x[-1] + torch.randn(input_dim) * 0.1
        X.append(x)
        Y.append(y)
    return torch.stack(X), torch.stack(Y)

def main():
    print("=" * 50)
    print("H3.22: SSM Dimension Scaling (Fast)")
    print("=" * 50)
    
    results = {}
    
    # Quick test: just 3 configs
    configs = [
        (8, 128),
        (16, 256),
        (32, 512),
    ]
    input_dim = 32
    n_samples = 100
    seq_len = 15
    
    X, Y = generate_data(n_samples, seq_len, input_dim)
    X, Y = X.to(device), Y.to(device)
    
    for state_dim, hidden_dim in configs:
        config_name = f"state={state_dim}_hidden={hidden_dim}"
        print(f"\n--- {config_name} ---")
        
        torch.manual_seed(42)
        model = SSMModel(input_dim, hidden_dim, state_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            mse = criterion(model(X), Y).item()
        
        results[config_name] = {"mse": mse}
        print(f"  MSE: {mse:.6f}")
    
    # Find best
    best = min(results, key=lambda k: results[k]["mse"])
    print(f"\n=== BEST: {best} ===")
    
    # Save
    output = {
        "experiment": "H3.22",
        "results": results,
        "best": best
    }
    os.makedirs("experiments/H3.22-ssm-dim-scaling", exist_ok=True)
    with open("experiments/H3.22-ssm-dim-scaling/results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Results saved")

if __name__ == "__main__":
    main()