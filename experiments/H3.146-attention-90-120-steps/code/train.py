#!/usr/bin/env python3
"""
H3.146: Attention on 90-120 Step Sequences
Test attention mechanism on very long sequences (90-120 timesteps)
Building on H3.145's success (+4.9% causal attention on 60-80 steps)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

class ConcatenationBaseline(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=512):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs, lang):
        combined = torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1)
        return self.fusion(combined[:, -1, :])

class AttentionArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=512, use_causal=True):
        super().__init__()
        self.use_causal = use_causal
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(latent_dim * 2, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        
        if self.use_causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=combined.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
        else:
            mask = None
        
        attn_out, _ = self.cross_attn(combined, combined, combined, attn_mask=mask)
        
        return self.decoder(attn_out[:, -1, :])

class CausalAttentionArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=512):
        super().__init__()
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=latent_dim * 2, nhead=8, dim_feedforward=512, batch_first=True)
            for _ in range(2)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        
        for layer in self.layers:
            combined = layer(combined)
        
        return self.decoder(combined[:, -1, :])

def generate_long_sequence_data(n_samples, seq_len, autocorrelation=0.9):
    """Generate long sequence data with high autocorrelation (real robot characteristic)."""
    np.random.seed(42)
    obs_dim, lang_dim, action_dim = 8, 32, 7
    
    observations = []
    languages = []
    actions = []
    
    for _ in range(n_samples):
        obs = np.zeros((seq_len, obs_dim), dtype=np.float32)
        lang = np.zeros((seq_len, lang_dim), dtype=np.float32)
        action = np.zeros((seq_len, action_dim), dtype=np.float32)
        
        for i in range(seq_len):
            obs[i] = np.random.randn(obs_dim) * 0.5
            lang[i] = np.tanh(np.random.randn(lang_dim) * 0.3)
            
            if i == 0:
                action[i] = np.random.randn(action_dim) * 0.5
            else:
                action[i] = autocorrelation * action[i-1] + (1 - autocorrelation) * np.random.randn(action_dim) * 0.5
        
        observations.append(obs)
        languages.append(lang)
        actions.append(action)
    
    return list(zip(observations, languages, actions))

def train_and_eval(model, train_data, val_data, epochs=20):
    train_obs = torch.tensor(np.stack([x[0] for x in train_data]), dtype=torch.float32)
    train_lang = torch.tensor(np.stack([x[1] for x in train_data]), dtype=torch.float32)
    train_act = torch.tensor(np.stack([x[2] for x in train_data]), dtype=torch.float32)
    
    val_obs = torch.tensor(np.stack([x[0] for x in val_data]), dtype=torch.float32)
    val_lang = torch.tensor(np.stack([x[1] for x in val_data]), dtype=torch.float32)
    val_act = torch.tensor(np.stack([x[2] for x in val_data]), dtype=torch.float32)
    
    train_ds = TensorDataset(train_obs, train_lang, train_act)
    val_ds = TensorDataset(val_obs, val_lang, val_act)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for obs, lang, act in train_loader:
            opt.zero_grad()
            pred = model(obs, lang)
            loss = crit(pred, act[:, -1, :])
            loss.backward()
            opt.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, lang, act in val_loader:
                pred = model(obs, lang)
                val_losses.append(crit(pred, act[:, -1, :]).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss

def run_experiment():
    print("=" * 70)
    print("H3.146: Attention on 90-120 Step Sequences")
    print("=" * 70)
    
    results = {}
    
    test_configs = [
        {"seq_len": 90, "autocorrelation": 0.90},
        {"seq_len": 100, "autocorrelation": 0.92},
        {"seq_len": 110, "autocorrelation": 0.95},
        {"seq_len": 120, "autocorrelation": 0.98},
    ]
    
    all_concat_losses = []
    all_attn_losses = []
    all_causal_losses = []
    
    for config in test_configs:
        seq_len = config["seq_len"]
        autocorr = config["autocorrelation"]
        
        print(f"\n--- Testing seq_len={seq_len}, autocorrelation={autocorr} ---")
        
        train_data = generate_long_sequence_data(100, seq_len, autocorr)
        val_data = generate_long_sequence_data(30, seq_len, autocorr)
        
        print("Training Concatenation...")
        concat_model = ConcatenationBaseline()
        concat_loss = train_and_eval(concat_model, train_data, val_data)
        all_concat_losses.append(concat_loss)
        
        print("Training Standard Attention...")
        attn_model = AttentionArchitecture(use_causal=False)
        attn_loss = train_and_eval(attn_model, train_data, val_data)
        all_attn_losses.append(attn_loss)
        
        print("Training Causal Attention...")
        causal_model = CausalAttentionArchitecture()
        causal_loss = train_and_eval(causal_model, train_data, val_data)
        all_causal_losses.append(causal_loss)
        
        attn_improvement = (concat_loss - attn_loss) / concat_loss * 100
        causal_improvement = (concat_loss - causal_loss) / concat_loss * 100
        
        print(f"Seq {seq_len}: Concat={concat_loss:.4f}, Attn={attn_loss:.4f} ({attn_improvement:+.1f}%), Causal={causal_loss:.4f} ({causal_improvement:+.1f}%)")
        
        results[f"seq_{seq_len}"] = {
            "concat_loss": concat_loss,
            "attn_loss": attn_loss,
            "causal_loss": causal_loss,
            "attn_improvement": attn_improvement,
            "causal_improvement": causal_improvement
        }
    
    avg_concat = np.mean(all_concat_losses)
    avg_attn = np.mean(all_attn_losses)
    avg_causal = np.mean(all_causal_losses)
    
    attn_overall = (avg_concat - avg_attn) / avg_concat * 100
    causal_overall = (avg_concat - avg_causal) / avg_concat * 100
    
    print(f"\n{'=' * 70}")
    print(f"Overall: Concat={avg_concat:.4f}, Attn={avg_attn:.4f} ({attn_overall:+.1f}%), Causal={avg_causal:.4f} ({causal_overall:+.1f}%)")
    print(f"{'=' * 70}")
    
    final_results = {
        "concat_loss": float(avg_concat),
        "attention_loss": float(avg_attn),
        "causal_attention_loss": float(avg_causal),
        "attention_improvement_percent": float(attn_overall),
        "causal_improvement_percent": float(causal_overall),
        "attention_wins": bool(attn_overall > 0),
        "causal_wins": bool(causal_overall > 0),
        "config": {"test_configs": test_configs},
        "detailed_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    }
    
    print("\n" + json.dumps(final_results, indent=2))
    return final_results

if __name__ == "__main__":
    results = run_experiment()
    
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.146-attention-90-120-steps/results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)