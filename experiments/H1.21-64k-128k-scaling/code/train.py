"""H1.21: Test 64k-128k dimension scaling with optimal alpha."""

import torch
import torch.nn as nn
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    seed: int = 42
    n_samples: int = 500
    n_timesteps: int = 8
    n_objects: int = 3
    action_dim: int = 4
    hidden_dim: int = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class UnifiedModel(nn.Module):
    def __init__(self, total_dim: int, physical_pct: float = 0.22, alpha: float = 0.3, action_dim: int = 4):
        super().__init__()
        self.total_dim = total_dim
        self.physical_dim = int(total_dim * physical_pct)
        self.semantic_dim = total_dim - self.physical_dim
        self.alpha = alpha
        self.action_dim = action_dim
        
        self.physical_encoder = nn.Sequential(
            nn.Linear(self.physical_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
        )
        
        self.semantic_encoder = nn.Sequential(
            nn.Linear(self.semantic_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim * 2, total_dim),
            nn.ReLU(),
            nn.Dropout(alpha),
            nn.Linear(total_dim, total_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, action_dim),
        )

    def forward(self, physical, semantic):
        p_enc = self.physical_encoder(physical)
        s_enc = self.semantic_encoder(semantic)
        combined = torch.cat([p_enc, s_enc], dim=-1)
        fused = self.fusion(combined)
        return self.decoder(fused)

def generate_data(n_samples: int, n_timesteps: int, n_objects: int, action_dim: int, seed: int):
    set_seed(seed)
    X_physical = []
    X_semantic = []
    y_actions = []
    
    for _ in range(n_samples):
        for t in range(n_timesteps):
            obs = np.random.randn(n_objects, 7).astype(np.float32)
            language = np.random.randn(50).astype(np.float32)
            
            physical = obs[:, :4].flatten()
            semantics = np.concatenate([
                obs[:, 4:].flatten(),
                language[:20]
            ])
            
            action = np.sum(obs[:n_objects, :action_dim], axis=0) + np.random.randn(action_dim) * 0.01
            
            X_physical.append(physical)
            X_semantic.append(semantics)
            y_actions.append(action)
    
    X_physical = np.array(X_physical)
    X_semantic = np.array(X_semantic)
    y_actions = np.array(y_actions)
    
    return X_physical, X_semantic, y_actions

def train_model(model, X_physical, X_semantic, y_actions, epochs: int = 500, lr: float = 0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_phys = torch.tensor(X_physical, dtype=torch.float32).to(device)
    X_sem = torch.tensor(X_semantic, dtype=torch.float32).to(device)
    y = torch.tensor(y_actions, dtype=torch.float32).to(device)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_phys, X_sem)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return np.mean(losses[-20:])

def test_config(dim: int, alpha: float, n_samples: int = 500, n_runs: int = 3) -> List[float]:
    results = []
    for seed in range(42, 42 + n_runs):
        set_seed(seed)
        
        X_phys, X_sem, y = generate_data(n_samples, 8, 3, 4, seed)
        
        model = UnifiedModel(dim, alpha=alpha, action_dim=4).to(device)
        loss = train_model(model, X_phys, X_sem, y)
        results.append(loss)
    
    return results

def main():
    print("=" * 60)
    print("H1.21: 64k-128k Dimension Scaling")
    print("=" * 60)
    
    configs = [
        (32768, 0.3),
        (65536, 0.3),
        (65536, 0.5),
        (131072, 0.3),
        (131072, 0.5),
    ]
    
    results = {}
    for dim, alpha in configs:
        print(f"\nTesting {dim} dims with α={alpha}...", flush=True)
        losses = test_config(dim, alpha, n_samples=200, n_runs=3)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        results[(dim, alpha)] = (mean_loss, std_loss)
        print(f"  MSE: {mean_loss:.4f} ± {std_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])
    for (dim, alpha), (loss, std) in sorted_results:
        print(f"  {dim:>6} dims, α={alpha}: MSE={loss:.4f} ± {std:.4f}")
    
    best = sorted_results[0]
    print(f"\nBest: {best[0][0]} dims, α={best[0][1]} → MSE={best[1][0]:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()