#!/usr/bin/env python3
"""
H3.48: Semantic Reasoning Hub (SRH) + Attention on Extreme Long Sequences
Building on H3.47 success (+74.4%), tests if SRH + attention helps on 100+ timesteps
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class SemanticReasoningHub(nn.Module):
    """MIND-V style semantic reasoning hub"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hub_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hub_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, hub_dim))
        self.hub = nn.Sequential(nn.Linear(hub_dim * 2, hub_dim), nn.ReLU(), nn.Linear(hub_dim, hub_dim))
        self.decoder = nn.Sequential(nn.Linear(hub_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
        
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        z_hub = self.hub(torch.cat([z_obs, z_lang], dim=-1))
        return self.decoder(z_hub)


class SRHWithAttention(nn.Module):
    """SRH + attention for extreme long sequences"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hub_dim=256, num_heads=8):
        super().__init__()
        self.hub_dim = hub_dim
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hub_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, hub_dim))
        self.hub = nn.Sequential(nn.Linear(hub_dim * 2, hub_dim), nn.ReLU(), nn.Linear(hub_dim, hub_dim))
        self.attn = nn.MultiheadAttention(hub_dim, num_heads, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hub_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
        
    def forward(self, obs_seq, lang_seq):
        """Process sequences with attention"""
        # obs_seq: (T, obs_dim), lang_seq: (T, lang_dim)
        T = obs_seq.size(0)
        
        # Encode each timestep
        z_obs = self.obs_encoder(obs_seq)  # (T, hub_dim)
        z_lang = self.lang_encoder(lang_seq)  # (T, hub_dim)
        
        # Concatenate and pass through hub
        z_combined = torch.cat([z_obs, z_lang], dim=-1)  # (T, 2*hub_dim)
        z_hub = self.hub(z_combined)  # (T, hub_dim)
        
        # Apply self-attention across time
        z_expanded = z_hub.unsqueeze(1)  # (T, 1, hub_dim) - need seq dim
        attn_out, _ = self.attn(z_expanded, z_expanded, z_expanded)
        attn_out = attn_out.squeeze(1)  # (T, hub_dim)
        
        # Pool and decode
        z_pooled = attn_out.mean(0)  # (hub_dim,)
        return self.decoder(z_pooled)


class Baseline(nn.Module):
    """Simple concatenation baseline"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.fusion = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, action_dim))
        
    def forward(self, obs_seq, lang_seq):
        """Process sequences - pool then fuse"""
        # Pool over time, then fuse
        obs_pooled = obs_seq.mean(0)
        lang_pooled = lang_seq.mean(0)
        return self.fusion(torch.cat([self.obs_encoder(obs_pooled), self.lang_encoder(lang_pooled)], dim=-1))


def generate_extreme_long_sequence_data(n_samples=200, seq_len=120, obs_dim=8, lang_dim=32):
    """Generate extreme long sequence data (100+ timesteps)"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    sequences = []
    for _ in range(n_samples):
        T = np.random.randint(seq_len - 20, seq_len + 1)
        traj = np.random.randn(T, obs_dim + lang_dim + 7).astype(np.float32) * 0.1
        sequences.append(traj)
    
    return sequences


def train_model(model, train_data, epochs=30, lr=3e-4):
    """Train and evaluate model"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for traj in train_data[:len(train_data)//2]:
            if len(traj) < 10:
                continue
            opt.zero_grad()
            
            obs = torch.from_numpy(traj[:, :8])
            lang = torch.from_numpy(traj[:, 8:40])
            action = torch.from_numpy(traj[:, -7:])
            
            pred = model(obs, lang)
            loss = crit(pred, action.mean(0))
            loss.backward()
            opt.step()
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for traj in train_data[len(train_data)//2:]:
            if len(traj) < 10:
                continue
            obs = torch.from_numpy(traj[:, :8])
            lang = torch.from_numpy(traj[:, 8:40])
            action = torch.from_numpy(traj[:, -7:])
            
            pred = model(obs, lang)
            val_losses.append(crit(pred, action.mean(0)).item())
    
    return np.mean(val_losses) if val_losses else 1.0


def main():
    print("=" * 60)
    print("H3.48: SRH + Attention on Extreme Long Sequences (100+ timesteps)")
    print("=" * 60)
    
    seq_lens = [100, 120, 150, 200]
    results = {"baseline": [], "srh": [], "srh_attn": []}
    
    for seq_len in seq_lens:
        print(f"\nseq_len={seq_len}...", flush=True)
        train_data = generate_extreme_long_sequence_data(n_samples=200, seq_len=seq_len)
        
        base = Baseline()
        base_loss = train_model(base, train_data)
        results["baseline"].append(base_loss)
        
        srh = SemanticReasoningHub()
        srh_loss = train_model(srh, train_data)
        results["srh"].append(srh_loss)
        
        srh_attn = SRHWithAttention()
        srh_attn_loss = train_model(srh_attn, train_data)
        results["srh_attn"].append(srh_attn_loss)
        
        print(f"  baseline={base_loss:.4f}, srh={srh_loss:.4f}, srh_attn={srh_attn_loss:.4f}")
    
    base_avg = np.mean(results["baseline"])
    srh_avg = np.mean(results["srh"])
    srh_attn_avg = np.mean(results["srh_attn"])
    
    srh_imp = (base_avg - srh_avg) / base_avg * 100
    srh_attn_imp = (base_avg - srh_attn_avg) / base_avg * 100
    
    output = {
        "hypothesis": "H3.48",
        "description": "SRH + Attention on extreme long sequences (100+ timesteps)",
        "seq_lengths_tested": seq_lens,
        "baseline_avg_mse": float(base_avg),
        "srh_avg_mse": float(srh_avg),
        "srh_attn_avg_mse": float(srh_attn_avg),
        "srh_improvement_percent": float(srh_imp),
        "srh_attn_improvement_percent": float(srh_attn_imp),
        "srh_wins": bool(srh_avg < base_avg),
        "srh_attn_wins": bool(srh_attn_avg < base_avg),
        "status": "SUPPORTED" if srh_attn_imp > 0 else "REFUTED"
    }
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(json.dumps(output, indent=2))
    print("=" * 60)
    
    return output


if __name__ == "__main__":
    result = main()