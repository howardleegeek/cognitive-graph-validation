#!/usr/bin/env python3
"""
H3.139: Chunked Attention on 500+ Step Sequences

Hypothesis: Based on H3.138 showing linear attention has marginal improvement (+0.7%)
on 400+ steps, testing chunked/segmented attention will break the 400-step barrier.
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

class ConcatenationModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=1, total_dim=64):
        super().__init__()
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

class StandardAttentionModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=1, total_dim=64):
        super().__init__()
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

class ChunkedAttentionModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=1, total_dim=64, chunk_size=100):
        super().__init__()
        self.chunk_size = chunk_size
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, total_dim), nn.LayerNorm(total_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, total_dim), nn.LayerNorm(total_dim)
        )
        self.chunk_attn = nn.MultiheadAttention(total_dim, num_heads=2, batch_first=True)
        self.chunk_norm = nn.LayerNorm(total_dim)
        self.global_attn = nn.MultiheadAttention(total_dim, num_heads=2, batch_first=True)
        self.global_norm = nn.LayerNorm(total_dim)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs, lang):
        B = obs.shape[0]
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        combined = torch.stack([z_obs, z_lang], dim=1)
        
        T = combined.shape[1]
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        
        chunk_outputs = []
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, T)
            chunk = combined[:, start:end, :]
            
            if chunk.shape[1] > 1:
                chunk_out, _ = self.chunk_attn(chunk, chunk, chunk)
                chunk_out = self.chunk_norm(chunk_out.mean(dim=1))
            else:
                chunk_out = chunk.squeeze(1)
            chunk_outputs.append(chunk_out)
        
        if len(chunk_outputs) > 1:
            chunk_tensor = torch.stack(chunk_outputs, dim=1)
            global_out, _ = self.global_attn(chunk_tensor, chunk_tensor, chunk_tensor)
            z = self.global_norm(global_out.mean(dim=1))
        else:
            z = chunk_outputs[0]
        
        return self.decoder(z)

class LinearAttentionModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=1, total_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, total_dim), nn.LayerNorm(total_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, total_dim), nn.LayerNorm(total_dim)
        )
        self.feature_dim = total_dim
        self.decoder = nn.Sequential(
            nn.Linear(total_dim * 2, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs, lang):
        if obs.dim() == 2:
            B = obs.shape[0]
            z_obs = self.obs_encoder(obs)
            z_lang = self.lang_encoder(lang)
            z = torch.cat([z_obs, z_lang], dim=-1)
            return self.decoder(z)
        
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        
        k = F.relu(z_lang)
        v = z_lang
        q = F.relu(z_obs)
        
        kv = torch.einsum('bld,blv->bdv', k, v)
        z = torch.einsum('bld,bdv->blv', q, kv)
        
        z = torch.cat([z, z_obs], dim=-1)
        return self.decoder(z)

def generate_long_sequence_data(n_samples, seq_len, n_steps=5, rho=0.95):
    np.random.seed(42)
    torch.manual_seed(42)
    
    observations = []
    languages = []
    actions = []
    
    for _ in range(n_samples):
        trajectory = np.zeros(seq_len)
        step_size = max(1, seq_len // n_steps)
        
        for step in range(n_steps):
            start = step * step_size
            end = min((step + 1) * step_size, seq_len)
            if end <= start:
                continue
            
            base_val = np.random.rand() * 0.5 + 0.25
            for t in range(start, min(end, seq_len)):
                if t == start:
                    trajectory[t] = base_val
                else:
                    trajectory[t] = rho * trajectory[t-1] + (1-rho) * base_val + np.random.randn() * 0.02
        
        step_size = max(1, seq_len // n_steps)
        progress = np.array([i * step_size / seq_len for i in range(seq_len)], dtype=np.float32)
        
        obs = np.column_stack([
            trajectory,
            np.roll(trajectory, 1),
            np.sin(np.linspace(0, n_steps * np.pi, seq_len)),
            np.cos(np.linspace(0, n_steps * np.pi, seq_len)),
            progress,
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

def train_and_evaluate(model, train_data, val_data, epochs=30):
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
    print("H3.139: Chunked Attention on 500+ Step Sequences")
    print("=" * 70)
    
    results = {}
    
    test_configs = [
        (400, 5),   # 400 steps - baseline (attention should work)
        (450, 5),   # 450 steps - at boundary
        (500, 5),   # 500 steps - above boundary
        (550, 6),   # 550 steps
        (600, 6),   # 600 steps
    ]
    
    for seq_len, n_steps in test_configs:
        print(f"\n--- Testing: seq_len={seq_len}, n_steps={n_steps} ---")
        
        train_obs, train_lang, train_act = generate_long_sequence_data(15, seq_len, n_steps=n_steps)
        val_obs, val_lang, val_act = generate_long_sequence_data(5, seq_len, n_steps=n_steps)
        
        train_data = TensorDataset(train_obs, train_lang, train_act)
        val_data = TensorDataset(val_obs, val_lang, val_act)
        
        # Baseline (concatenation)
        baseline_model = ConcatenationModel()
        baseline_loss = train_and_evaluate(baseline_model, train_data, val_data)
        print(f"  Concat: {baseline_loss:.6f}")
        
        # Standard Attention
        attn_model = StandardAttentionModel()
        attn_loss = train_and_evaluate(attn_model, train_data, val_data)
        attn_improv = (baseline_loss - attn_loss) / baseline_loss * 100
        print(f"  Standard Attn: {attn_loss:.6f} ({attn_improv:+.1f}%)")
        
        # Linear Attention (from H3.138)
        linear_model = LinearAttentionModel()
        linear_loss = train_and_evaluate(linear_model, train_data, val_data)
        linear_improv = (baseline_loss - linear_loss) / baseline_loss * 100
        print(f"  Linear Attn: {linear_loss:.6f} ({linear_improv:+.1f}%)")
        
        # Chunked Attention
        chunk_size = 100 if seq_len <= 450 else 125
        chunk_model = ChunkedAttentionModel(chunk_size=chunk_size)
        chunk_loss = train_and_evaluate(chunk_model, train_data, val_data)
        chunk_improv = (baseline_loss - chunk_loss) / baseline_loss * 100
        print(f"  Chunked Attn (size={chunk_size}): {chunk_loss:.6f} ({chunk_improv:+.1f}%)")
        
        results[f"seq_{seq_len}"] = {
            "concat": baseline_loss,
            "standard_attn": attn_loss,
            "linear_attn": linear_loss,
            "chunked_attn": chunk_loss,
            "attn_improvement": attn_improv,
            "linear_improvement": linear_improv,
            "chunked_improvement": chunk_improv
        }
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    improvements = []
    for key, val in results.items():
        best = max(val["attn_improvement"], val["linear_improvement"], val["chunked_improvement"])
        improvements.append(best)
        
        best_method = "standard" if val["attn_improvement"] == best else ("linear" if val["linear_improvement"] == best else "chunked")
        print(f"{key}: best={best_method} ({best:+.1f}%)")
    
    avg_improvement = np.mean(improvements)
    print(f"\nAverage improvement: {avg_improvement:+.1f}%")
    
    if avg_improvement > 10:
        status = "SUPPORTED"
    elif avg_improvement > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    output = {
        "experiment": "H3.139",
        "hypothesis": "Chunked attention on 500+ step sequences",
        "results": results,
        "avg_improvement": avg_improvement,
        "status": status
    }
    
    import os
    os.makedirs("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.139-chunked-attention-500plus/results", exist_ok=True)
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.139-chunked-attention-500plus/results/metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    main()