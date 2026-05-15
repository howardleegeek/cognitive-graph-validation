#!/usr/bin/env python3
"""
H1.357: Cognitive Graph on 50-70 Step Sequences
Based on H1.355: CG +33.0% on 30-50 steps
Test if CG advantage continues at 50-70 steps
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
    """Dataset with long sequences (50-70 timesteps)"""
    def __init__(self, base_dataset, seq_len=60):
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
            phase = (t % 15) / 15.0
            obs_t = base_obs + np.sin(phase * np.pi) * 0.2 + np.random.randn(*base_obs.shape).astype(np.float32) * 0.02
            action_t = base_action + np.sin(t * 0.15) * 0.12
            
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


class CognitiveGraphModel(nn.Module):
    """Cognitive Graph model with unified representation"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=112, semantic_dim=384):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        outputs = []
        for t in range(obs_seq.size(1)):
            obs_t = obs_seq[:, t]
            lang_t = lang_seq[:, t]
            
            z_phys = self.obs_to_unified(obs_t)
            z_sem = self.lang_to_unified(lang_t)
            
            z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
            z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
            
            nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
            
            for layer in self.gnn_layers:
                msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
                nodes = nodes + layer(msgs)
            
            out = self.decoder(nodes.mean(dim=1))
            outputs.append(out)
        
        return torch.stack(outputs).mean(dim=0)


class CognitiveGraphWithAttention(nn.Module):
    """Cognitive Graph with attention for longer sequences"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=112, semantic_dim=384, n_heads=8):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        
        self.cross_attn = nn.MultiheadAttention(total_dim, n_heads, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        # Process all timesteps at once
        batch_size = obs_seq.size(0)
        seq_len = obs_seq.size(1)
        
        # Encode all timesteps
        obs_flat = obs_seq.view(-1, obs_seq.size(-1))
        lang_flat = lang_seq.view(-1, lang_seq.size(-1))
        
        z_phys = self.obs_to_unified(obs_flat)
        z_sem = self.lang_to_unified(lang_flat)
        
        z_phys = z_phys.view(batch_size, seq_len, -1)
        z_sem = z_sem.view(batch_size, seq_len, -1)
        
        # Stack nodes
        nodes = torch.cat([z_phys, z_sem], dim=-1)
        
        # GNN layers
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
            nodes = nodes + layer(msgs)
        
        # Attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode
        out = self.decoder(attn_out.mean(dim=1))
        
        return out


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
for seq_len in [50, 60, 70]:
    print(f"\n=== Testing {seq_len}-step sequences ===")
    train_seq = LongSequenceDataset(train_data, seq_len=seq_len)
    val_seq = LongSequenceDataset(val_data, seq_len=seq_len)
    
    train_loader = DataLoader(train_seq, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_seq, batch_size=16)
    
    print(f"Training Concatenation Baseline ({seq_len} steps)...")
    baseline = ConcatenationBaseline()
    base_loss = train_and_eval(baseline, train_loader, val_loader)
    
    print(f"Training Cognitive Graph ({seq_len} steps)...")
    cg = CognitiveGraphModel()
    cg_loss = train_and_eval(cg, train_loader, val_loader)
    
    print(f"Training CG + Attention ({seq_len} steps)...")
    cg_attn = CognitiveGraphWithAttention()
    cg_attn_loss = train_and_eval(cg_attn, train_loader, val_loader)
    
    cg_improvement = (base_loss - cg_loss) / base_loss * 100
    cg_attn_improvement = (base_loss - cg_attn_loss) / base_loss * 100
    
    results[seq_len] = {
        'baseline_loss': float(base_loss),
        'cg_loss': float(cg_loss),
        'cg_attn_loss': float(cg_attn_loss),
        'cg_improvement': float(cg_improvement),
        'cg_attn_improvement': float(cg_attn_improvement),
        'cg_wins': cg_loss < base_loss,
        'cg_attn_wins': cg_attn_loss < base_loss
    }
    
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  CG: {cg_loss:.6f} ({cg_improvement:+.1f}%)")
    print(f"  CG+Attn: {cg_attn_loss:.6f} ({cg_attn_improvement:+.1f}%)")

avg_cg = np.mean([r['cg_improvement'] for r in results.values()])
avg_cg_attn = np.mean([r['cg_attn_improvement'] for r in results.values()])

print(f"\n=== SUMMARY ===")
print(f"Average CG improvement: {avg_cg:+.1f}%")
print(f"Average CG+Attn improvement: {avg_cg_attn:+.1f}%")

best_cg = max(results.items(), key=lambda x: x[1]['cg_improvement'])
print(f"Best CG: {best_cg[0]} steps ({best_cg[1]['cg_improvement']:+.1f}%)")

output = {
    'baseline_loss': float(base_loss),
    'cognitive_graph_loss': float(cg_loss),
    'improvement_percent': float(avg_cg),
    'cognitive_graph_wins': bool(cg_loss < base_loss),
    'config': {
        'task_type': 'cg_long_sequences',
        'seq_lengths': list(results.keys()),
        'hypothesis': 'H1.357'
    },
    'detailed_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
}

print(json.dumps(output, indent=2))