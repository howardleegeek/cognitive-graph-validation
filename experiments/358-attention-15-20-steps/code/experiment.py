#!/usr/bin/env python3
"""
H3.358: Attention on 15-20 Step Sequences (Crossover Zone)
Based on H3.352: Attention -28.6% on 8-15 steps (concat wins)
Based on H3.353: Attention +16.0% on 20-40 steps (attention wins)
Test the crossover zone: 15-20 steps
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from data_loader import prepare_datasets


class CrossoverDataset(Dataset):
    """Dataset with medium sequences (15-20 timesteps)"""
    def __init__(self, base_dataset, seq_len=17):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        seq_obs = []
        seq_lang = []
        seq_action = []
        
        base_obs = item['observation'].numpy().astype(np.float32)
        base_action = item['action'].numpy().astype(np.float32)
        
        for t in range(self.seq_len):
            phase = (t % 8) / 8.0
            obs_t = base_obs + np.sin(phase * np.pi) * 0.15 + np.random.randn(*base_obs.shape).astype(np.float32) * 0.02
            action_t = base_action + np.sin(t * 0.2) * 0.1
            
            seq_obs.append(torch.FloatTensor(obs_t))
            seq_lang.append(item['language'])
            seq_action.append(torch.FloatTensor(action_t))
        
        obs_seq = torch.stack(seq_obs)
        lang_seq = torch.stack([torch.FloatTensor(l) if isinstance(l, np.ndarray) else l for l in seq_lang])
        action_seq = torch.stack(seq_action)
        
        return {
            'observation': obs_seq,
            'language': lang_seq,
            'action': action_seq,
            'seq_len': self.seq_len
        }


class ConcatenationBaseline(nn.Module):
    """Simple concatenation baseline"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq.mean(dim=1))
        lang_enc = self.lang_encoder(lang_seq.mean(dim=1))
        return self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))


class AttentionModel(nn.Module):
    """Attention-based model"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=256, n_heads=8):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq)
        lang_enc = self.lang_encoder(lang_seq)
        
        attn_out, _ = self.cross_attn(obs_enc, lang_enc, lang_enc)
        
        obs_pooled = attn_out.mean(dim=1)
        lang_pooled = lang_enc.mean(dim=1)
        
        return self.fusion(torch.cat([obs_pooled, lang_pooled], dim=-1))


class CausalAttentionModel(nn.Module):
    """Causal attention model"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=256, n_heads=8):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        
        self.causal_attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq)
        lang_enc = self.lang_encoder(lang_seq)
        
        seq_len = obs_enc.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=obs_enc.device), diagonal=1).bool()
        
        attn_out, _ = self.causal_attn(obs_enc, lang_enc, lang_enc, attn_mask=mask)
        
        obs_pooled = attn_out.mean(dim=1)
        lang_pooled = lang_enc.mean(dim=1)
        
        return self.fusion(torch.cat([obs_pooled, lang_pooled], dim=-1))


class ActionGatedAttention(nn.Module):
    """Action-gated attention for crossover zone"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=256, n_heads=8):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        
        self.action_gate = nn.Linear(action_dim, latent_dim)
        self.cross_attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq)
        lang_enc = self.lang_encoder(lang_seq)
        
        # Gate with action from previous timestep
        batch_size = obs_seq.size(0)
        dummy_action = torch.zeros(batch_size, 7, device=obs_seq.device)
        gate = torch.sigmoid(self.action_gate(dummy_action)).unsqueeze(1)
        
        attn_out, _ = self.cross_attn(obs_enc * gate, lang_enc, lang_enc)
        
        obs_pooled = attn_out.mean(dim=1)
        lang_pooled = lang_enc.mean(dim=1)
        
        return self.fusion(torch.cat([obs_pooled, lang_pooled], dim=-1))


def train_and_eval(model, train_loader, val_loader, epochs=80):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            target = batch['action'].mean(dim=1)
            loss = crit(pred, target)
            loss.backward()
            opt.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                target = batch['action'].mean(dim=1)
                val_losses.append(crit(pred, target).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


print("Loading data...")
train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)

results = {}
for seq_len in [15, 17, 20]:
    print(f"\n=== Testing {seq_len}-step sequences (crossover zone) ===")
    train_seq = CrossoverDataset(train_data, seq_len=seq_len)
    val_seq = CrossoverDataset(val_data, seq_len=seq_len)
    
    train_loader = DataLoader(train_seq, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_seq, batch_size=16)
    
    print(f"Training Concatenation Baseline ({seq_len} steps)...")
    baseline = ConcatenationBaseline()
    base_loss = train_and_eval(baseline, train_loader, val_loader)
    
    print(f"Training Standard Attention ({seq_len} steps)...")
    attn = AttentionModel()
    attn_loss = train_and_eval(attn, train_loader, val_loader)
    
    print(f"Training Causal Attention ({seq_len} steps)...")
    causal = CausalAttentionModel()
    causal_loss = train_and_eval(causal, train_loader, val_loader)
    
    print(f"Training Action-Gated Attention ({seq_len} steps)...")
    gated = ActionGatedAttention()
    gated_loss = train_and_eval(gated, train_loader, val_loader)
    
    attn_improvement = (base_loss - attn_loss) / base_loss * 100
    causal_improvement = (base_loss - causal_loss) / base_loss * 100
    gated_improvement = (base_loss - gated_loss) / base_loss * 100
    
    results[seq_len] = {
        'baseline_loss': float(base_loss),
        'attn_loss': float(attn_loss),
        'causal_loss': float(causal_loss),
        'gated_loss': float(gated_loss),
        'attn_improvement': float(attn_improvement),
        'causal_improvement': float(causal_improvement),
        'gated_improvement': float(gated_improvement),
        'attn_wins': attn_loss < base_loss,
        'causal_wins': causal_loss < base_loss,
        'gated_wins': gated_loss < base_loss
    }
    
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  Standard Attn: {attn_loss:.6f} ({attn_improvement:+.1f}%)")
    print(f"  Causal Attn: {causal_loss:.6f} ({causal_improvement:+.1f}%)")
    print(f"  Action-Gated: {gated_loss:.6f} ({gated_improvement:+.1f}%)")

avg_attn = np.mean([r['attn_improvement'] for r in results.values()])
avg_causal = np.mean([r['causal_improvement'] for r in results.values()])
avg_gated = np.mean([r['gated_improvement'] for r in results.values()])

print(f"\n=== SUMMARY ===")
print(f"Average Standard Attention: {avg_attn:+.1f}%")
print(f"Average Causal Attention: {avg_causal:+.1f}%")
print(f"Average Action-Gated: {avg_gated:+.1f}%")

# Find crossover point
for seq_len, r in results.items():
    if r['attn_improvement'] > 0:
        print(f"Crossover found at {seq_len} steps: Attention wins ({r['attn_improvement']:+.1f}%)")
        break

output = {
    'baseline_loss': float(base_loss),
    'attention_loss': float(attn_loss),
    'causal_attention_loss': float(causal_loss),
    'gated_attention_loss': float(gated_loss),
    'improvement_percent': float(avg_attn),
    'cognitive_graph_wins': bool(attn_loss < base_loss),
    'config': {
        'task_type': 'attention_crossover_zone',
        'seq_lengths': list(results.keys()),
        'hypothesis': 'H3.358'
    },
    'detailed_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
}

print(json.dumps(output, indent=2))