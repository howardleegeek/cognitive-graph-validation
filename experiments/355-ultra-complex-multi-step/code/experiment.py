#!/usr/bin/env python3
"""
H1.355: Ultra-Complex Multi-Step Tasks (30-50 steps)
Based on H1.351 (+32.4% on 5-10 steps) and H1.353 (+26.4% on 15-30 steps)
Test if CG advantage continues at extreme complexity (30-50 steps)
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


class UltraComplexDataset(Dataset):
    """Dataset with ultra-complex multi-step tasks (30-50 timesteps)"""
    def __init__(self, base_dataset, seq_len=40):
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
            # Add complex temporal structure with multiple phases
            phase = (t % 10) / 10.0
            obs_t = base_obs + np.sin(phase * np.pi) * 0.15 + np.random.randn(*base_obs.shape).astype(np.float32) * 0.02
            
            # Actions evolve over time
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
    """Simple concatenation baseline for ultra-complex tasks"""
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
        # Process each timestep
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


class AttentionModel(nn.Module):
    """Attention-based model for ultra-long sequences"""
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

# Test different sequence lengths
results = {}
for seq_len in [30, 40, 50]:
    print(f"\n=== Testing {seq_len}-step sequences ===")
    train_seq = UltraComplexDataset(train_data, seq_len=seq_len)
    val_seq = UltraComplexDataset(val_data, seq_len=seq_len)
    
    train_loader = DataLoader(train_seq, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_seq, batch_size=16)
    
    print(f"Training Concatenation Baseline ({seq_len} steps)...")
    baseline = ConcatenationBaseline()
    base_loss = train_and_eval(baseline, train_loader, val_loader)
    
    print(f"Training Cognitive Graph ({seq_len} steps)...")
    cg = CognitiveGraphModel()
    cg_loss = train_and_eval(cg, train_loader, val_loader)
    
    print(f"Training Attention Model ({seq_len} steps)...")
    attn = AttentionModel()
    attn_loss = train_and_eval(attn, train_loader, val_loader)
    
    cg_improvement = (base_loss - cg_loss) / base_loss * 100
    attn_improvement = (base_loss - attn_loss) / base_loss * 100
    
    results[seq_len] = {
        'baseline_loss': float(base_loss),
        'cg_loss': float(cg_loss),
        'attn_loss': float(attn_loss),
        'cg_improvement': float(cg_improvement),
        'attn_improvement': float(attn_improvement),
        'cg_wins': cg_loss < base_loss,
        'attn_wins': attn_loss < base_loss
    }
    
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  CG: {cg_loss:.6f} ({cg_improvement:+.1f}%)")
    print(f"  Attn: {attn_loss:.6f} ({attn_improvement:+.1f}%)")

# Summary
avg_cg = np.mean([r['cg_improvement'] for r in results.values()])
avg_attn = np.mean([r['attn_improvement'] for r in results.values()])

print(f"\n=== SUMMARY ===")
print(f"Average CG improvement: {avg_cg:+.1f}%")
print(f"Average Attention improvement: {avg_attn:+.1f}%")

output = {
    'baseline_loss': float(base_loss),
    'cognitive_graph_loss': float(cg_loss),
    'attention_loss': float(attn_loss),
    'improvement_percent': float(avg_cg),
    'cognitive_graph_wins': bool(cg_loss < base_loss),
    'attention_wins': bool(attn_loss < base_loss),
    'config': {
        'task_type': 'ultra_complex_multi_step',
        'seq_lengths': list(results.keys()),
        'hypothesis': 'H1.355'
    },
    'detailed_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
}

print(json.dumps(output, indent=2))