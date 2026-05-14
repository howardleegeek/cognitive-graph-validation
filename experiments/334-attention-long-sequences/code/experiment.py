import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
from data_loader import prepare_datasets

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.fusion = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))

class ConcatenationArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
    def forward(self, obs, lang):
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        return self.decoder(torch.cat([obs_enc, lang_enc], dim=-1))

class StandardAttentionArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, action_dim))
    def forward(self, obs, lang):
        obs_enc = self.obs_encoder(obs).unsqueeze(1)
        lang_enc = self.lang_encoder(lang).unsqueeze(1)
        combined = torch.cat([obs_enc, lang_enc], dim=1)
        attn_out, _ = self.attention(combined, combined, combined)
        return self.decoder(attn_out.mean(dim=1))

class CausalAttentionArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.causal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, action_dim))
        
        self.register_buffer('causal_mask', torch.triu(torch.ones(100, 100), diagonal=1).bool())
    
    def forward(self, obs, lang):
        obs_enc = self.obs_encoder(obs).unsqueeze(1)
        lang_enc = self.lang_encoder(lang).unsqueeze(1)
        combined = torch.cat([obs_enc, lang_enc], dim=1)
        seq_len = combined.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_out, _ = self.causal_attention(combined, combined, combined, attn_mask=causal_mask)
        return self.decoder(attn_out.mean(dim=1))

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

CONFIG = {
    "task_type": "long_sequences",
    "seq_lengths": [20, 25, 30, 35, 40],
    "test_standard_attention": True,
    "test_causal_attention": True
}

print("Loading data...")
train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

print("Training Baseline (concatenation)...")
baseline = ConcatenationArchitecture()
base_loss = train_and_eval(baseline, train_loader, val_loader, epochs=50)

print("Training Standard Attention...")
std_attn = StandardAttentionArchitecture()
std_attn_loss = train_and_eval(std_attn, train_loader, val_loader, epochs=50)

print("Training Causal Attention...")
causal_attn = CausalAttentionArchitecture()
causal_attn_loss = train_and_eval(causal_attn, train_loader, val_loader, epochs=50)

improvement_std = (base_loss - std_attn_loss) / base_loss * 100
improvement_causal = (base_loss - causal_attn_loss) / base_loss * 100

results = {
    'baseline_loss': float(base_loss),
    'standard_attention_loss': float(std_attn_loss),
    'causal_attention_loss': float(causal_attn_loss),
    'improvement_standard_attention_percent': float(improvement_std),
    'improvement_causal_attention_percent': float(improvement_causal),
    'standard_attention_wins': bool(std_attn_loss < base_loss),
    'causal_attention_wins': bool(causal_attn_loss < base_loss),
    'config': CONFIG
}

print(json.dumps(results, indent=2))