#!/usr/bin/env python3
"""
H1.233: Stronger Regularization on Complex Tasks

Hypothesis: Stronger regularization (reg=0.2-0.5) can enable unified architecture 
to handle complex tasks that previously failed.

Based on: H1.232 showed reg=0.1 helps (+9.0% on complex tasks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

class UnifiedWithReg(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=1, total_dim=128, reg=0.3):
        super().__init__()
        self.reg = reg
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, total_dim), nn.LayerNorm(total_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, total_dim), nn.LayerNorm(total_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(total_dim * 2, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        z = torch.cat([z_obs, z_lang], dim=-1)
        out = self.fusion(z)
        return out.mean(dim=1)  # Mean over sequence
    
    def get_regularization(self):
        l2 = 0
        for p in self.parameters():
            l2 += (p ** 2).sum()
        return self.reg * l2

class Baseline(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        out = self.net(torch.cat([obs, lang], dim=-1))
        return out.mean(dim=1)

def generate_data(n_samples, seq_len, complexity=0.8, rho=0.95):
    np.random.seed(42)
    torch.manual_seed(42)
    
    observations = []
    languages = []
    actions = []
    
    for _ in range(n_samples):
        base = np.cumsum(np.random.randn(seq_len) * 0.1)
        trajectory = np.zeros(seq_len)
        trajectory[0] = base[0] + np.random.randn() * 0.1
        for t in range(1, seq_len):
            trajectory[t] = rho * trajectory[t-1] + (1-rho) * base[t] + np.random.randn() * 0.05
        
        obs = np.column_stack([
            trajectory,
            np.roll(trajectory, 1),
            np.sin(np.linspace(0, 4*np.pi, seq_len)),
            np.cos(np.linspace(0, 4*np.pi, seq_len)),
            np.random.rand(seq_len) * complexity,
            np.random.rand(seq_len),
            np.random.rand(seq_len) * 0.5,
            np.random.rand(seq_len) * 0.5,
        ]).astype(np.float32)
        
        lang = np.tile(np.random.rand(32).astype(np.float32), (seq_len, 1))
        action = trajectory.mean() * 0.5 + 0.25
        
        observations.append(obs)
        languages.append(lang)
        actions.append([action])
    
    return (
        torch.tensor(np.stack(observations), dtype=torch.float32),
        torch.tensor(np.stack(languages), dtype=torch.float32),
        torch.tensor(np.stack(actions), dtype=torch.float32)
    )

def train_and_evaluate(model_class, train_data, val_data, reg=0.0, epochs=30):
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    
    if reg > 0:
        model = model_class(reg)
    else:
        model = model_class()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for obs, lang, act in train_loader:
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, act.squeeze(-1))
            if reg > 0:
                loss = loss + model.get_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, lang, act in val_loader:
                pred = model(obs, lang)
                val_losses.append(criterion(pred, act.squeeze(-1)).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss

def main():
    print("=" * 70)
    print("H1.233: Stronger Regularization on Complex Tasks")
    print("=" * 70)
    
    results = {}
    
    for seq_len in [50, 80, 100]:
        print(f"\n--- Testing sequence length: {seq_len} ---")
        
        train_obs, train_lang, train_act = generate_data(15, seq_len)
        val_obs, val_lang, val_act = generate_data(5, seq_len)
        
        train_data = TensorDataset(train_obs, train_lang, train_act)
        val_data = TensorDataset(val_obs, val_lang, val_act)
        
        baseline_loss = train_and_evaluate(Baseline, train_data, val_data, reg=0)
        print(f"  Baseline: {baseline_loss:.6f}")
        
        for reg in [0.1, 0.2, 0.3, 0.5]:
            unified_loss = train_and_evaluate(lambda r=reg: UnifiedWithReg(reg=r), train_data, val_data, reg=reg)
            improvement = (baseline_loss - unified_loss) / baseline_loss * 100
            results[f"seq_{seq_len}_reg_{reg}"] = {
                "unified": unified_loss,
                "baseline": baseline_loss,
                "improvement": improvement
            }
            print(f"  reg={reg}: {unified_loss:.6f} ({improvement:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    improvements = []
    for key, val in results.items():
        print(f"{key}: {val['improvement']:+.1f}%")
        improvements.append(val['improvement'])
    
    avg_improvement = np.mean(improvements)
    print(f"\nAverage improvement: {avg_improvement:+.1f}%")
    
    if avg_improvement > 5:
        status = "SUPPORTED"
    elif avg_improvement > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    output = {
        "experiment": "H1.233",
        "hypothesis": "Stronger regularization on complex tasks",
        "results": results,
        "avg_improvement": avg_improvement,
        "status": status
    }
    
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.233-strong-reg-ultra-complex/results/metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    main()