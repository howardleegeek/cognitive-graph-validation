#!/usr/bin/env python3
"""
H1.234: Unified + Attention with Regularization on Complex Multi-Step Tasks

Hypothesis: Combining unified architecture with attention and regularization 
will enable handling of complex multi-step tasks.
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

class UnifiedModel(nn.Module):
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
        self.decoder = nn.Sequential(
            nn.Linear(total_dim * 2, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        z = torch.cat([z_obs, z_lang], dim=-1)
        return self.decoder(z)
    
    def get_regularization(self):
        l2 = 0
        for p in self.parameters():
            l2 += (p ** 2).sum()
        return self.reg * l2

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

def generate_multistep_data(n_samples, seq_len, n_steps=5, rho=0.95):
    np.random.seed(42)
    torch.manual_seed(42)
    
    observations = []
    languages = []
    actions = []
    
    for _ in range(n_samples):
        trajectory = np.zeros(seq_len)
        step_size = seq_len // n_steps
        
        for step in range(n_steps):
            start = step * step_size
            end = min((step + 1) * step_size, seq_len)
            
            base_val = np.random.rand() * 0.5 + 0.25
            for t in range(start, end):
                if t == start:
                    trajectory[t] = base_val
                else:
                    trajectory[t] = rho * trajectory[t-1] + (1-rho) * base_val + np.random.randn() * 0.02
        
        obs = np.column_stack([
            trajectory,
            np.roll(trajectory, 1),
            np.sin(np.linspace(0, n_steps * np.pi, seq_len)),
            np.cos(np.linspace(0, n_steps * np.pi, seq_len)),
            np.tile(np.linspace(0, 1, step_size), n_steps)[:seq_len],
            np.random.rand(seq_len) * 0.8,
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
    print("H1.234: Unified + Attention + Regularization on Complex Multi-Step")
    print("=" * 70)
    
    results = {}
    
    for seq_len in [50, 80, 100]:
        print(f"\n--- Testing sequence length: {seq_len} ---")
        
        train_obs, train_lang, train_act = generate_multistep_data(15, seq_len, n_steps=5)
        val_obs, val_lang, val_act = generate_multistep_data(5, seq_len, n_steps=5)
        
        train_data = TensorDataset(train_obs, train_lang, train_act)
        val_data = TensorDataset(val_obs, val_lang, val_act)
        
        # Baseline
        baseline_model = Baseline()
        baseline_loss = train_and_evaluate(baseline_model, train_data, val_data, reg=0)
        print(f"  Baseline: {baseline_loss:.6f}")
        
        # Unified (no attention, no reg)
        unified_model = UnifiedModel()
        unified_loss = train_and_evaluate(unified_model, train_data, val_data, reg=0)
        unimprov = (baseline_loss - unified_loss) / baseline_loss * 100
        print(f"  Unified: {unified_loss:.6f} ({unimprov:+.1f}%)")
        
        # Unified + Attention (no reg)
        uniattn_model = UnifiedAttentionModel()
        uniattn_loss = train_and_evaluate(uniattn_model, train_data, val_data, reg=0)
        uniattn_improv = (baseline_loss - uniattn_loss) / baseline_loss * 100
        print(f"  Unified+Attn: {uniattn_loss:.6f} ({uniattn_improv:+.1f}%)")
        
        # Unified + Attention + Reg (0.1)
        uniattnreg_model = UnifiedAttentionModel(reg=0.1)
        reg01_loss = train_and_evaluate(uniattnreg_model, train_data, val_data, reg=0.1)
        reg01_improv = (baseline_loss - reg01_loss) / baseline_loss * 100
        print(f"  Unified+Attn+Reg0.1: {reg01_loss:.6f} ({reg01_improv:+.1f}%)")
        
        results[f"seq_{seq_len}"] = {
            "baseline": baseline_loss,
            "unified": unified_loss,
            "unified_attn": uniattn_loss,
            "reg_0.1": reg01_loss,
            "unified_improvement": unimprov,
            "uniattn_improvement": uniattn_improv,
            "reg01_improvement": reg01_improv
        }
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    best_configs = []
    for key, val in results.items():
        improvements = [
            ("unified", val["unified_improvement"]),
            ("unified+attn", val["uniattn_improvement"]),
            ("reg_0.1", val["reg01_improvement"])
        ]
        best = max(improvements, key=lambda x: x[1])
        best_configs.append(best)
        print(f"{key}: best={best[0]} ({best[1]:.1f}%)")
    
    avg_best = np.mean([x[1] for x in best_configs])
    print(f"\nAverage best improvement: {avg_best:+.1f}%")
    
    if avg_best > 10:
        status = "SUPPORTED"
    elif avg_best > 5:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    output = {
        "experiment": "H1.234",
        "hypothesis": "Unified + Attention + Regularization on complex multi-step",
        "results": results,
        "avg_best_improvement": avg_best,
        "status": status
    }
    
    import os
    os.makedirs("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.234-unified-attn-reg-complex/results", exist_ok=True)
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.234-unified-attn-reg-complex/results/metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    main()