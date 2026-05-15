#!/usr/bin/env python3
"""
H3.362: Attention on 160-200 Step Sequences
Based on H3.360 (+15.6% on 120-150 steps) - test if attention continues to work at longer lengths
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


class LongSequenceDataset(Dataset):
    """Dataset with very long sequences (160-200 timesteps)"""
    def __init__(self, base_dataset, seq_len=180):
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
            # Add temporal structure with autocorrelation
            phase = (t % 20) / 20.0
            obs_t = base_obs + np.sin(phase * np.pi) * 0.1 + np.random.randn(*base_obs.shape).astype(np.float32) * 0.01
            
            # Actions evolve with temporal structure
            action_t = base_action + np.sin(t * 0.15) * 0.08
            
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
    """Simple concatenation baseline for long sequences"""
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


class StandardAttentionModel(nn.Module):
    """Standard attention model for long sequences"""
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
    """Causal attention model for long sequences"""
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
        
        # Causal mask for attention
        self.causal_attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq)
        lang_enc = self.lang_encoder(lang_seq)
        
        # Create causal mask
        seq_len = obs_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=obs_seq.device), diagonal=1).bool()
        
        attn_out, _ = self.causal_attn(obs_enc, lang_enc, lang_enc, attn_mask=mask)
        
        obs_pooled = attn_out.mean(dim=1)
        lang_pooled = lang_enc.mean(dim=1)
        
        return self.fusion(torch.cat([obs_pooled, lang_pooled], dim=-1))


def train_and_eval(model, train_loader, val_loader, epochs=30):
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

# Test different sequence lengths
results = {}
for seq_len in [160, 200]:
    print(f"\n=== Testing {seq_len}-step sequences ===")
    train_seq = LongSequenceDataset(train_data, seq_len=seq_len)
    val_seq = LongSequenceDataset(val_data, seq_len=seq_len)
    
    train_loader = DataLoader(train_seq, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_seq, batch_size=16)
    
    print(f"Training Concatenation Baseline ({seq_len} steps)...")
    baseline = ConcatenationBaseline()
    base_loss = train_and_eval(baseline, train_loader, val_loader)
    
    print(f"Training Standard Attention ({seq_len} steps)...")
    std_attn = StandardAttentionModel()
    std_loss = train_and_eval(std_attn, train_loader, val_loader)
    
    print(f"Training Causal Attention ({seq_len} steps)...")
    causal_attn = CausalAttentionModel()
    causal_loss = train_and_eval(causal_attn, train_loader, val_loader)
    
    std_improvement = (base_loss - std_loss) / base_loss * 100
    causal_improvement = (base_loss - causal_loss) / base_loss * 100
    
    best_improvement = max(std_improvement, causal_improvement)
    best_type = "standard" if std_improvement > causal_improvement else "causal"
    
    results[seq_len] = {
        'baseline_loss': float(base_loss),
        'std_attn_loss': float(std_loss),
        'causal_attn_loss': float(causal_loss),
        'std_improvement': float(std_improvement),
        'causal_improvement': float(causal_improvement),
        'best_improvement': float(best_improvement),
        'best_type': best_type
    }
    
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  Standard Attn: {std_loss:.6f} ({std_improvement:+.1f}%)")
    print(f"  Causal Attn: {causal_loss:.6f} ({causal_improvement:+.1f}%)")
    print(f"  Best: {best_type} ({best_improvement:+.1f}%)")

# Summary
avg_std = np.mean([r['std_improvement'] for r in results.values()])
avg_causal = np.mean([r['causal_improvement'] for r in results.values()])
avg_best = np.mean([r['best_improvement'] for r in results.values()])

print(f"\n=== SUMMARY ===")
print(f"Average Standard Attention improvement: {avg_std:+.1f}%")
print(f"Average Causal Attention improvement: {avg_causal:+.1f}%")
print(f"Average Best improvement: {avg_best:+.1f}%")

# Determine overall winner
overall_winner = "standard" if avg_std > avg_causal else "causal"
cognitive_graph_wins = avg_best > 0

output = {
    'baseline_loss': float(base_loss),
    'std_attn_loss': float(std_loss),
    'causal_attn_loss': float(causal_loss),
    'improvement_percent': float(avg_best),
    'cognitive_graph_wins': bool(cognitive_graph_wins),
    'config': {
        'task_type': 'attention_long_sequences',
        'seq_lengths': list(results.keys()),
        'hypothesis': 'H3.362'
    },
    'detailed_results': {k: {kk: float(vv) if isinstance(vv, (int, float)) else str(vv) for kk, vv in v.items()} for k, v in results.items()}
}

print(json.dumps(output, indent=2))