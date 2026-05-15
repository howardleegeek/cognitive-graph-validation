#!/usr/bin/env python3
"""
H3.353: Attention on Longer Sequences (20+ timesteps)
Based on H3.352 failure (-28.6% on 8-15 steps), test if attention works on longer sequences
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
    """Dataset with longer sequences (20-40 timesteps) - fixed length"""
    def __init__(self, base_dataset, seq_len=30):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Create fixed-length sequence with temporal structure
        seq_obs = []
        seq_lang = []
        seq_action = []
        
        base_obs = item['observation'].numpy().astype(np.float32)
        
        for t in range(self.seq_len):
            # Add temporal autocorrelation
            obs_t = base_obs + np.sin(t * 0.3) * 0.1 + np.random.randn(*base_obs.shape).astype(np.float32) * 0.02
            seq_obs.append(torch.FloatTensor(obs_t))
            seq_lang.append(item['language'])
            seq_action.append(item['action'])
        
        # Stack sequences
        obs_seq = torch.stack(seq_obs)
        lang_seq = torch.stack([torch.FloatTensor(l) if isinstance(l, np.ndarray) else l for l in seq_lang])
        action_seq = torch.stack([torch.FloatTensor(a) if isinstance(a, np.ndarray) else a for a in seq_action])
        
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
        # Average pooling over sequence
        obs_enc = self.obs_encoder(obs_seq.mean(dim=1))
        lang_enc = self.lang_encoder(lang_seq.mean(dim=1))
        return self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))

class AttentionModel(nn.Module):
    """Attention-based model for longer sequences"""
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
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq)
        lang_enc = self.lang_encoder(lang_seq)
        
        # Cross-attention: obs attends to lang
        attn_out, _ = self.cross_attn(obs_enc, lang_enc, lang_enc)
        
        # Pool and predict
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
            loss = crit(pred, batch['action'].mean(dim=1))  # Average actions
            loss.backward()
            opt.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                val_losses.append(crit(pred, batch['action'].mean(dim=1)).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss

print("Loading data...")
train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)
train_seq = LongSequenceDataset(train_data, seq_len=30)
val_seq = LongSequenceDataset(val_data, seq_len=30)

train_loader = DataLoader(train_seq, batch_size=16, shuffle=True)
val_loader = DataLoader(val_seq, batch_size=16)

print("Training Concatenation Baseline on 20-40 step sequences...")
baseline = ConcatenationBaseline()
base_loss = train_and_eval(baseline, train_loader, val_loader)

print("Training Attention Model on 20-40 step sequences...")
attn = AttentionModel()
attn_loss = train_and_eval(attn, train_loader, val_loader)

improvement = (base_loss - attn_loss) / base_loss * 100

results = {
    'baseline_loss': float(base_loss),
    'attention_loss': float(attn_loss),
    'improvement_percent': float(improvement),
    'attention_wins': bool(attn_loss < base_loss),
    'config': {
        'task_type': 'longer_sequences',
        'seq_len': '20-40',
        'hypothesis': 'H3.353'
    }
}

print(json.dumps(results, indent=2))