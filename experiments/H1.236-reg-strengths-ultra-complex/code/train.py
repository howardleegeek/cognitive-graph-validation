#!/usr/bin/env python3
"""
H1.236: Regularization Strengths on Ultra-Complex Tasks

Hypothesis: Testing different regularization strengths (0.05-0.5) on ultra-complex
tasks to find optimal regularization for maximum complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

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
        return out

class UnifiedAttentionModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=1, total_dim=64, reg=0.1):
        super().__init__()
        self.reg = reg
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, total_dim), nn.LayerNorm(total_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, total_dim), nn.LayerNorm(total_dim)
        )
        self.attn = nn.MultiheadAttention(total_dim, num_heads=2, batch_first=True)
        self.attn_norm = nn.LayerNorm(total_dim)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        combined = torch.stack([z_obs, z_lang], dim=1)
        attn_out, _ = self.attn(combined, combined, combined)
        z = self.attn_norm(attn_out.mean(dim=1))
        return self.decoder(z)
    
    def get_regularization(self):
        l2 = 0
        for p in self.parameters():
            l2 += (p ** 2).sum()
        return self.reg * l2

def generate_ultra_complex_data(n_samples, seq_len, n_steps=15, complexity=1.5):
    np.random.seed(42)
    torch.manual_seed(42)
    
    observations = []
    languages = []
    actions = []
    
    for _ in range(n_samples):
        trajectory = np.zeros(seq_len)
        step_size = max(1, seq_len // n_steps)
        rho = 0.95
        
        for step in range(n_steps):
            start = step * step_size
            end = min((step + 1) * step_size, seq_len)
            if end <= start:
                continue
            
            base_val = np.random.rand() * 0.5 + 0.25
            noise_scale = 0.02 * complexity
            for t in range(start, min(end, seq_len)):
                if t == start:
                    trajectory[t] = base_val
                else:
                    trajectory[t] = rho * trajectory[t-1] + (1-rho) * base_val + np.random.randn() * noise_scale
        
        step_size = max(1, seq_len // n_steps)
        progress = np.array([i * step_size / seq_len for i in range(seq_len)], dtype=np.float32)
        
        obs = np.column_stack([
            trajectory,
            np.roll(trajectory, 1),
            np.sin(np.linspace(0, n_steps * np.pi, seq_len)),
            np.cos(np.linspace(0, n_steps * np.pi, seq_len)),
            progress,
            np.random.rand(seq_len) * 0.8 * complexity,
            np.random.rand(seq_len),
            np.random.rand(seq_len) * 0.5,
        ]).astype(np.float32)
        
        lang = np.zeros((seq_len, 32), dtype=np.float32)
        for step in range(n_steps):
            start = step * step_size
            end = min((step + 1) * step_size, seq_len)
            lang[start:end, :4] = step / n_steps
        
        action = trajectory.mean() * 0.5 + 0.25
        
        observations.append(obs)
        languages.append(lang)
        actions.append([action])
    
    return (
        torch.tensor(np.stack(observations), dtype=torch.float32),
        torch.tensor(np.stack(languages), dtype=torch.float32),
        torch.tensor(np.stack(actions), dtype=torch.float32)
    )

def train_and_evaluate(model, train_data, val_data, reg=0.0, epochs=30):
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for obs, lang, act in train_loader:
            B, T, _ = obs.shape
            obs_flat = obs.view(B * T, -1)
            lang_flat = lang.view(B * T, -1)
            pred = model(obs_flat, lang_flat)
            loss = criterion(pred, act.squeeze(-1).repeat(B * T, 1))
            if reg > 0 and hasattr(model, 'get_regularization'):
                loss = loss + model.get_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, lang, act in val_loader:
                B, T, _ = obs.shape
                obs_flat = obs.view(B * T, -1)
                lang_flat = lang.view(B * T, -1)
                pred = model(obs_flat, lang_flat)
                val_losses.append(criterion(pred, act.squeeze(-1).repeat(B * T, 1)).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss

def main():
    print("=" * 70)
    print("H1.236: Regularization Strengths on Ultra-Complex Tasks")
    print("=" * 70)
    
    results = {}
    
    reg_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    test_configs = [
        (80, 12, 1.2),
        (100, 15, 1.5),
        (120, 18, 1.8),
    ]
    
    for seq_len, n_steps, complexity in test_configs:
        print(f"\n--- Testing: seq_len={seq_len}, n_steps={n_steps}, complexity={complexity} ---")
        
        train_obs, train_lang, train_act = generate_ultra_complex_data(15, seq_len, n_steps=n_steps, complexity=complexity)
        val_obs, val_lang, val_act = generate_ultra_complex_data(5, seq_len, n_steps=n_steps, complexity=complexity)
        
        train_data = TensorDataset(train_obs, train_lang, train_act)
        val_data = TensorDataset(val_obs, val_lang, val_act)
        
        # Baseline
        baseline_model = Baseline()
        baseline_loss = train_and_evaluate(baseline_model, train_data, val_data, reg=0)
        print(f"  Baseline: {baseline_loss:.6f}")
        
        best_reg = None
        best_improvement = float('-inf')
        best_loss = float('inf')
        
        for reg in reg_values:
            model = UnifiedAttentionModel(reg=reg)
            loss = train_and_evaluate(model, train_data, val_data, reg=reg)
            improv = (baseline_loss - loss) / baseline_loss * 100
            print(f"  reg={reg}: {loss:.6f} ({improv:+.1f}%)")
            
            if improv > best_improvement:
                best_improvement = improv
                best_reg = reg
                best_loss = loss
        
        results[f"seq{seq_len}_steps{n_steps}_comp{complexity}"] = {
            "baseline": baseline_loss,
            "best_reg": best_reg,
            "best_loss": best_loss,
            "best_improvement": best_improvement
        }
        print(f"  -> Best: reg={best_reg} ({best_improvement:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    improvements = []
    best_regs = []
    for key, val in results.items():
        improvements.append(val["best_improvement"])
        best_regs.append(val["best_reg"])
        print(f"{key}: best_reg={val['best_reg']} ({val['best_improvement']:+.1f}%)")
    
    avg_improvement = np.mean(improvements)
    most_common_reg = max(set(best_regs), key=best_regs.count)
    print(f"\nAverage improvement: {avg_improvement:+.1f}%")
    print(f"Most common best reg: {most_common_reg}")
    
    if avg_improvement > 30:
        status = "SUPPORTED"
    elif avg_improvement > 10:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    output = {
        "experiment": "H1.236",
        "hypothesis": "Regularization strengths on ultra-complex tasks",
        "results": results,
        "avg_improvement": avg_improvement,
        "most_common_best_reg": most_common_reg,
        "status": status
    }
    
    import os
    os.makedirs("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.236-reg-strengths-ultra-complex/results", exist_ok=True)
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.236-reg-strengths-ultra-complex/results/metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    main()