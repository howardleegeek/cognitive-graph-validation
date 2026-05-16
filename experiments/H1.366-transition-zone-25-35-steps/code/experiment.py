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

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim))
        self.lang_to_unified = nn.Sequential(nn.Linear(lang_dim, 256), nn.ReLU(), nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim))
        self.gnn_layers = nn.ModuleList([nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)) for _ in range(3)])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(total_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
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

class CognitiveGraphWithAttention(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368, attention_dim=256):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim))
        self.lang_to_unified = nn.Sequential(nn.Linear(lang_dim, 256), nn.ReLU(), nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim))
        self.gnn_layers = nn.ModuleList([nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)) for _ in range(3)])
        self.temporal_attn = nn.MultiheadAttention(attention_dim, num_heads=8, batch_first=True)
        self.obs_proj = nn.Linear(physical_dim, attention_dim)
        self.lang_proj = nn.Linear(semantic_dim, attention_dim)
        self.decoder = nn.Sequential(nn.Linear(total_dim + attention_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        z_phys_proj = self.obs_proj(z_phys)
        z_sem_proj = self.lang_proj(z_sem)
        temporal_nodes = torch.stack([z_phys_proj, z_sem_proj], dim=1)
        attn_out, _ = self.temporal_attn(temporal_nodes, temporal_nodes, temporal_nodes)
        context = attn_out.mean(dim=1)
        return self.decoder(torch.cat([nodes.mean(dim=1), context], dim=-1))

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
    "n_train": 200,
    "n_val": 50,
    "complexity_range": "25-35_steps",
    "hypothesis": "H1.366: CG on transition zone (between sweet spot and failure)"
}

print("Loading data...")
train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

print("Testing transition zone (25-35 steps)...")
results_by_steps = {}

for n_steps in [25, 28, 30, 32, 35]:
    print(f"\n--- Testing {n_steps} steps ---")
    
    print("Training Baseline...")
    baseline = BaselineArchitecture()
    base_loss = train_and_eval(baseline, train_loader, val_loader, epochs=50)
    
    print("Training Cognitive Graph...")
    cog = CognitiveGraphArchitecture()
    cog_loss = train_and_eval(cog, train_loader, val_loader, epochs=50)
    
    print("Training CG + Attention...")
    cog_attn = CognitiveGraphWithAttention()
    cog_attn_loss = train_and_eval(cog_attn, train_loader, val_loader, epochs=50)
    
    improvement_cg = (base_loss - cog_loss) / base_loss * 100
    improvement_cg_attn = (base_loss - cog_attn_loss) / base_loss * 100
    
    results_by_steps[n_steps] = {
        "baseline": float(base_loss),
        "cognitive_graph": float(cog_loss),
        "cg_attention": float(cog_attn_loss),
        "cg_improvement": float(improvement_cg),
        "cg_attn_improvement": float(improvement_cg_attn),
        "cg_wins": bool(cog_loss < base_loss),
        "cg_attn_wins": bool(cog_attn_loss < base_loss)
    }
    
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  CG: {cog_loss:.6f} ({improvement_cg:+.1f}%)")
    print(f"  CG+Attn: {cog_attn_loss:.6f} ({improvement_cg_attn:+.1f}%)")

avg_cg = np.mean([r["cg_improvement"] for r in results_by_steps.values()])
avg_cg_attn = np.mean([r["cg_attn_improvement"] for r in results_by_steps.values()])

results = {
    "baseline_loss": float(np.mean([r["baseline"] for r in results_by_steps.values()])),
    "cognitive_graph_loss": float(np.mean([r["cognitive_graph"] for r in results_by_steps.values()])),
    "cg_attention_loss": float(np.mean([r["cg_attention"] for r in results_by_steps.values()])),
    "improvement_percent_cg": float(avg_cg),
    "improvement_percent_cg_attn": float(avg_cg_attn),
    "cognitive_graph_wins": bool(avg_cg > 0),
    "cg_attention_wins": bool(avg_cg_attn > 0),
    "results_by_steps": results_by_steps,
    "config": CONFIG
}

print("\n" + "="*50)
print("SUMMARY:")
print(f"  CG Average: {avg_cg:+.1f}%")
print(f"  CG+Attn Average: {avg_cg_attn:+.1f}%")
print("="*50)

print(json.dumps(results, indent=2))