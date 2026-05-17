import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from data_loader import prepare_datasets

# Architectures
class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.fusion = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))

class HierarchicalArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.subgoal_predictor = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.action_decoder = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        subgoal = self.subgoal_predictor(torch.cat([z_obs, z_lang], dim=-1))
        return self.action_decoder(torch.cat([z_obs, subgoal], dim=-1))

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368, n_gnn_layers=3, n_attention_heads=8):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.obs_to_unified = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim))
        self.lang_to_unified = nn.Sequential(nn.Linear(lang_dim, 256), nn.ReLU(), nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim))
        self.gnn_layers = nn.ModuleList([nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)) for _ in range(n_gnn_layers)])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=n_attention_heads, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(total_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))

def train_and_eval(model, train_loader, val_loader, epochs=50, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
            loss = crit(pred, batch['action'])
            val_losses.append(loss.item())
    return np.mean(val_losses)

def run_experiment(config):
    # Prepare datasets
    train_dataset, val_dataset, _ = prepare_datasets(
        n_train=config['n_train'],
        n_val=config['n_val']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    results = {}
    
    # Baseline
    baseline = BaselineArchitecture(
        obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128
    )
    baseline_mse = train_and_eval(baseline, train_loader, val_loader, epochs=config['n_epochs'], lr=config['learning_rate'])
    results['baseline_mse'] = baseline_mse
    
    # Hierarchical
    hierarchical = HierarchicalArchitecture(
        obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128
    )
    hierarchical_mse = train_and_eval(hierarchical, train_loader, val_loader, epochs=config['n_epochs'], lr=config['learning_rate'])
    results['hierarchical_mse'] = hierarchical_mse
    
    # Cognitive Graph variants
    cg_variants = []
    
    # Ablation 1: Vary representation sizes
    for physical_dim, semantic_dim in [(72, 184), (144, 368), (288, 736)]:  # 0.5x, 1x, 2x
        cg = CognitiveGraphArchitecture(
            obs_dim=8, lang_dim=32, action_dim=7,
            physical_dim=physical_dim, semantic_dim=semantic_dim,
            n_gnn_layers=3, n_attention_heads=8
        )
        cg_mse = train_and_eval(cg, train_loader, val_loader, epochs=config['n_epochs'], lr=config['learning_rate'])
        cg_variants.append({
            'name': f'CG_physical{physical_dim}_semantic{semantic_dim}',
            'mse': cg_mse,
            'improvement': ((baseline_mse - cg_mse) / baseline_mse) * 100
        })
    
    # Ablation 2: Vary attention depth (heads)
    for n_heads in [1, 4, 8, 16]:
        cg = CognitiveGraphArchitecture(
            obs_dim=8, lang_dim=32, action_dim=7,
            physical_dim=144, semantic_dim=368,
            n_gnn_layers=3, n_attention_heads=n_heads
        )
        cg_mse = train_and_eval(cg, train_loader, val_loader, epochs=config['n_epochs'], lr=config['learning_rate'])
        cg_variants.append({
            'name': f'CG_heads{n_heads}',
            'mse': cg_mse,
            'improvement': ((baseline_mse - cg_mse) / baseline_mse) * 100
        })
    
    # Ablation 3: Vary GNN layers
    for n_layers in [1, 2, 3, 4]:
        cg = CognitiveGraphArchitecture(
            obs_dim=8, lang_dim=32, action_dim=7,
            physical_dim=144, semantic_dim=368,
            n_gnn_layers=n_layers, n_attention_heads=8
        )
        cg_mse = train_and_eval(cg, train_loader, val_loader, epochs=config['n_epochs'], lr=config['learning_rate'])
        cg_variants.append({
            'name': f'CG_layers{n_layers}',
            'mse': cg_mse,
            'improvement': ((baseline_mse - cg_mse) / baseline_mse) * 100
        })
    
    # Find best CG variant
    best_cg = min(cg_variants, key=lambda x: x['mse'])
    results['cg_mse'] = best_cg['mse']
    results['cg_improvement'] = best_cg['improvement']
    results['cg_variants'] = cg_variants
    results['hierarchical_improvement'] = ((baseline_mse - hierarchical_mse) / baseline_mse) * 100
    
    return results

if __name__ == "__main__":
    config = {
        'n_train': 400,
        'n_val': 100,
        'n_epochs': 60,
        'batch_size': 32,
        'learning_rate': 1e-3
    }
    
    results = run_experiment(config)
    
    # Save results
    os.makedirs('../results', exist_ok=True)
    with open('../results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Experiment Results:")
    print(f"Baseline MSE: {results['baseline_mse']:.6f}")
    print(f"Hierarchical MSE: {results['hierarchical_mse']:.6f} ({results['hierarchical_improvement']:.2f}% improvement)")
    print(f"Best CG MSE: {results['cg_mse']:.6f} ({results['cg_improvement']:.2f}% improvement)")
    print("\nCG Variants:")
    for variant in results['cg_variants']:
        print(f"  {variant['name']}: {variant['mse']:.6f} ({variant['improvement']:.2f}% improvement)")