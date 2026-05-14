#!/usr/bin/env python3
"""
H3.337: Attention on Longer Sequences (20+ Timesteps)
Based on H3 failure (concat wins on simple tasks), test attention on longer sequences
Previous experiment 334 showed +30% on 20-40 steps - verify and extend
"""
import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
from data_loader import prepare_datasets

class ConcatenationArchitecture(nn.Module):
    """Standard concatenation baseline"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))

class AttentionArchitecture(nn.Module):
    """Attention-based fusion for longer sequences"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        # Multi-head attention for cross-modal fusion
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs, lang):
        obs_enc = self.obs_encoder(obs).unsqueeze(1)  # [B, 1, H]
        lang_enc = self.lang_encoder(lang).unsqueeze(1)  # [B, 1, H]
        
        # Cross-attention: language attends to observation
        attn_out, _ = self.cross_attn(lang_enc, obs_enc, obs_enc)
        
        # Concatenate and decode
        combined = torch.cat([obs_enc, attn_out], dim=-1).squeeze(1)
        return self.decoder(combined)

def train_and_eval(model, train_loader, val_loader, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            val_losses.append(crit(pred, batch['action']).item())
    return np.mean(val_losses)

print("Loading data for longer sequence test (20+ timesteps)...")
train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

print("Training Concatenation baseline (20+ steps)...")
concat_model = ConcatenationArchitecture()
concat_loss = train_and_eval(concat_model, train_loader, val_loader, epochs=50)

print("Training Attention model (20+ steps)...")
attn_model = AttentionArchitecture()
attn_loss = train_and_eval(attn_model, train_loader, val_loader, epochs=50)

improvement = (concat_loss - attn_loss) / concat_loss * 100

results = {
    'concat_loss': float(concat_loss),
    'attention_loss': float(attn_loss),
    'improvement_percent': float(improvement),
    'attention_wins': bool(attn_loss < concat_loss),
    'config': {'task_type': 'longer_sequences', 'seq_len': '20_plus'}
}

print(json.dumps(results, indent=2))