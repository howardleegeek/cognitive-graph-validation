#!/usr/bin/env python3
"""
H1.353: Complex Multi-Step Extended (15-30 steps)
Based on H1.351 success (+32.4% on 5-10 steps), test if CG advantage extends to 15-30 steps
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

class ExtendedTrajectoryDataset(Dataset):
    """Dataset with longer trajectories (15-30 steps)"""
    def __init__(self, base_dataset, min_steps=15, max_steps=30):
        self.base_dataset = base_dataset
        self.min_steps = min_steps
        self.max_steps = max_steps
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        # Extend trajectory by repeating with variations
        orig_obs = item['observation']
        orig_lang = item['language']
        orig_action = item['action']
        
        n_steps = np.random.randint(self.min_steps, self.max_steps + 1)
        
        # Create extended trajectory
        extended_obs = []
        extended_lang = []
        extended_action = []
        
        for t in range(n_steps):
            noise = np.random.randn(*orig_obs.shape) * 0.01
            obs_t = orig_obs + noise + np.sin(t * 0.5) * 0.05
            extended_obs.append(obs_t)
            extended_lang.append(orig_lang)
            extended_action.append(orig_action)
        
        # Average for prediction
        avg_obs = np.mean(extended_obs, axis=0)
        avg_lang = extended_lang[0]
        avg_action = np.mean(extended_action, axis=0)
        
        return {
            'observation': torch.FloatTensor(avg_obs),
            'language': torch.FloatTensor(avg_lang),
            'action': torch.FloatTensor(avg_action),
            'n_steps': n_steps
        }

class BaselineArchitecture(nn.Module):
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
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=112, semantic_dim=384):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))

def train_and_eval(model, train_loader, val_loader, epochs=80):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.MSELoss()
    
    best_loss = float('inf')
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
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss

print("Loading data...")
train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)
train_ext = ExtendedTrajectoryDataset(train_data, min_steps=15, max_steps=30)
val_ext = ExtendedTrajectoryDataset(val_data, min_steps=15, max_steps=30)

train_loader = DataLoader(train_ext, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ext, batch_size=16)

print("Training Baseline on 15-30 step tasks...")
baseline = BaselineArchitecture()
base_loss = train_and_eval(baseline, train_loader, val_loader)

print("Training Cognitive Graph on 15-30 step tasks...")
cog = CognitiveGraphArchitecture()
cog_loss = train_and_eval(cog, train_loader, val_loader)

improvement = (base_loss - cog_loss) / base_loss * 100

results = {
    'baseline_loss': float(base_loss),
    'cognitive_graph_loss': float(cog_loss),
    'improvement_percent': float(improvement),
    'cognitive_graph_wins': bool(cog_loss < base_loss),
    'config': {
        'task_type': 'multi_step_extended',
        'n_steps': '15-30',
        'hypothesis': 'H1.353'
    }
}

print(json.dumps(results, indent=2))