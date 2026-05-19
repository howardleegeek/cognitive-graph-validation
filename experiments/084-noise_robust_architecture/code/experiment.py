#!/usr/bin/env python3
"""
H1.465: Test architectural changes for noise robustness
Hypothesis: More robust GNN architectures (skip connections, batch norm, residual connections)
can improve CG's noise tolerance.

Based on H1.464 findings:
- Standard CG fails at 1% noise (-44% improvement, 1% win rate)
- Heavy noise augmentation (50%) partially restores advantage (6.94% improvement, 76% win rate)

This experiment tests:
1. CG with skip connections (residual GNN layers)
2. CG with batch normalization
3. CG with both skip connections + batch norm
4. CG with dropout for regularization
5. CG with layer normalization (stronger than current)
"""

import sys
import os
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from data_loader import prepare_datasets

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============== Architectures ==============

class BaselineArchitecture(nn.Module):
    """Simple concatenation baseline."""
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


class CognitiveGraphStandard(nn.Module):
    """Standard CG from H1.464 (for comparison)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
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
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        return self.decoder(nodes.mean(dim=1))


class CognitiveGraphSkipConnections(nn.Module):
    """CG with residual skip connections in GNN layers."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
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
        
        # GNN with skip connections
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
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
            # Residual connection: nodes = nodes + layer(nodes)
            nodes = nodes + layer(nodes)  # Skip connection here
        
        return self.decoder(nodes.mean(dim=1))


class CognitiveGraphBatchNorm(nn.Module):
    """CG with batch normalization instead of layer norm."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.BatchNorm1d(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.BatchNorm1d(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.BatchNorm1d(total_dim))
            for _ in range(3)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
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
            # Reshape for batch norm (needs 2D input)
            B, N, D = nodes.shape
            nodes_flat = nodes.view(B * N, D)
            nodes_flat = nodes_flat + layer(nodes_flat)
            nodes = nodes_flat.view(B, N, D)
        
        return self.decoder(nodes.mean(dim=1))


class CognitiveGraphSkipBatchNorm(nn.Module):
    """CG with both skip connections and batch normalization."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.BatchNorm1d(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.BatchNorm1d(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.BatchNorm1d(total_dim))
            for _ in range(3)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
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
            # Reshape for batch norm
            B, N, D = nodes.shape
            nodes_flat = nodes.view(B * N, D)
            nodes_flat = nodes_flat + layer(nodes_flat)  # Skip connection
            nodes = nodes_flat.view(B, N, D)
        
        return self.decoder(nodes.mean(dim=1))


class CognitiveGraphDropout(nn.Module):
    """CG with dropout for regularization."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368, dropout=0.3):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
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
        
        return self.decoder(nodes.mean(dim=1))


class CognitiveGraphPreNorm(nn.Module):
    """CG with pre-normalization (more stable gradients)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
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
        
        # Pre-norm GNN layers
        self.layer_norms = nn.ModuleList([nn.LayerNorm(total_dim) for _ in range(3)])
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU())
            for _ in range(3)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for ln, layer in zip(self.layer_norms, self.gnn_layers):
            # Pre-norm: normalize before transformation
            nodes = nodes + layer(ln(nodes))
        
        return self.decoder(nodes.mean(dim=1))


# ============== Training Functions ==============

def add_noise(data, noise_level):
    """Add Gaussian noise to observations."""
    noisy_data = {}
    for key, val in data.items():
        if key == 'observation':
            noise = torch.randn_like(val) * noise_level * val.std()
            noisy_data[key] = val + noise
        else:
            noisy_data[key] = val
    return noisy_data


def train_and_eval(model, train_loader, val_loader, epochs=50, noise_level=0.01):
    """Train model and evaluate with noisy inputs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Add noise during training
            noisy_batch = add_noise(batch, noise_level)
            
            pred = model(noisy_batch['observation'], noisy_batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Evaluate with noise
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            noisy_batch = add_noise(batch, noise_level)
            pred = model(noisy_batch['observation'], noisy_batch['language'])
            loss = criterion(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============== Main Experiment ==============

def main():
    print("=" * 60)
    print("H1.465: Architectural Changes for Noise Robustness")
    print("=" * 60)
    
    # Ensure results directory exists
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    train_data, val_data, _ = prepare_datasets(n_train=400, n_val=100, n_test=0)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Test at 1% noise (where standard CG fails)
    noise_level = 0.01
    
    results = {}
    
    # 1. Baseline
    print("\n[1/7] Training Baseline...")
    baseline = BaselineArchitecture()
    baseline_loss = train_and_eval(baseline, train_loader, val_loader, noise_level=noise_level)
    results['baseline'] = {
        'loss': baseline_loss,
        'params': count_parameters(baseline)
    }
    print(f"  Baseline loss: {baseline_loss:.6f}")
    
    # 2. Standard CG (reference)
    print("\n[2/7] Training Standard CG...")
    cg_standard = CognitiveGraphStandard()
    cg_standard_loss = train_and_eval(cg_standard, train_loader, val_loader, noise_level=noise_level)
    results['cg_standard'] = {
        'loss': cg_standard_loss,
        'params': count_parameters(cg_standard)
    }
    improvement_standard = (baseline_loss - cg_standard_loss) / baseline_loss * 100
    print(f"  Standard CG loss: {cg_standard_loss:.6f} ({improvement_standard:+.2f}%)")
    
    # 3. CG with Skip Connections
    print("\n[3/7] Training CG with Skip Connections...")
    cg_skip = CognitiveGraphSkipConnections()
    cg_skip_loss = train_and_eval(cg_skip, train_loader, val_loader, noise_level=noise_level)
    results['cg_skip'] = {
        'loss': cg_skip_loss,
        'params': count_parameters(cg_skip)
    }
    improvement_skip = (baseline_loss - cg_skip_loss) / baseline_loss * 100
    print(f"  Skip CG loss: {cg_skip_loss:.6f} ({improvement_skip:+.2f}%)")
    
    # 4. CG with Batch Norm
    print("\n[4/7] Training CG with Batch Norm...")
    cg_bn = CognitiveGraphBatchNorm()
    cg_bn_loss = train_and_eval(cg_bn, train_loader, val_loader, noise_level=noise_level)
    results['cg_bn'] = {
        'loss': cg_bn_loss,
        'params': count_parameters(cg_bn)
    }
    improvement_bn = (baseline_loss - cg_bn_loss) / baseline_loss * 100
    print(f"  BatchNorm CG loss: {cg_bn_loss:.6f} ({improvement_bn:+.2f}%)")
    
    # 5. CG with Skip + Batch Norm
    print("\n[5/7] Training CG with Skip + Batch Norm...")
    cg_skip_bn = CognitiveGraphSkipBatchNorm()
    cg_skip_bn_loss = train_and_eval(cg_skip_bn, train_loader, val_loader, noise_level=noise_level)
    results['cg_skip_bn'] = {
        'loss': cg_skip_bn_loss,
        'params': count_parameters(cg_skip_bn)
    }
    improvement_skip_bn = (baseline_loss - cg_skip_bn_loss) / baseline_loss * 100
    print(f"  Skip+BN CG loss: {cg_skip_bn_loss:.6f} ({improvement_skip_bn:+.2f}%)")
    
    # 6. CG with Dropout
    print("\n[6/7] Training CG with Dropout...")
    cg_dropout = CognitiveGraphDropout(dropout=0.3)
    cg_dropout_loss = train_and_eval(cg_dropout, train_loader, val_loader, noise_level=noise_level)
    results['cg_dropout'] = {
        'loss': cg_dropout_loss,
        'params': count_parameters(cg_dropout)
    }
    improvement_dropout = (baseline_loss - cg_dropout_loss) / baseline_loss * 100
    print(f"  Dropout CG loss: {cg_dropout_loss:.6f} ({improvement_dropout:+.2f}%)")
    
    # 7. CG with Pre-Norm
    print("\n[7/7] Training CG with Pre-Norm...")
    cg_prenorm = CognitiveGraphPreNorm()
    cg_prenorm_loss = train_and_eval(cg_prenorm, train_loader, val_loader, noise_level=noise_level)
    results['cg_prenorm'] = {
        'loss': cg_prenorm_loss,
        'params': count_parameters(cg_prenorm)
    }
    improvement_prenorm = (baseline_loss - cg_prenorm_loss) / baseline_loss * 100
    print(f"  PreNorm CG loss: {cg_prenorm_loss:.6f} ({improvement_prenorm:+.2f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Architecture':<25} {'Loss':<12} {'vs Baseline':<15} {'Params':<12}")
    print("-" * 60)
    
    for name, data in results.items():
        if name == 'baseline':
            print(f"{'Baseline':<25} {data['loss']:<12.6f} {'---':<15} {data['params']:<12,}")
        else:
            improvement = (baseline_loss - data['loss']) / baseline_loss * 100
            wins = "✓" if data['loss'] < baseline_loss else "✗"
            print(f"{name:<25} {data['loss']:<12.6f} {improvement:>+6.2f}% {wins:<6} {data['params']:<12,}")
    
    # Find best architecture
    best_name = min(results.keys(), key=lambda x: results[x]['loss'])
    best_improvement = (baseline_loss - results[best_name]['loss']) / baseline_loss * 100
    
    print("\n" + "=" * 60)
    print(f"BEST: {best_name} with {best_improvement:+.2f}% improvement")
    print("=" * 60)
    
    # Save results
    output = {
        'experiment': 'H1.465',
        'description': 'Architectural changes for noise robustness',
        'noise_level': noise_level,
        'results': {
            name: {
                'loss': float(data['loss']),
                'params': int(data['params']),
                'improvement_pct': float((baseline_loss - data['loss']) / baseline_loss * 100) if name != 'baseline' else 0.0,
                'cg_wins': bool(data['loss'] < baseline_loss) if name != 'baseline' else False
            }
            for name, data in results.items()
        },
        'best_architecture': best_name,
        'best_improvement_pct': float(best_improvement),
        'conclusion': f"{'SUPPORTED' if best_improvement > 0 else 'REFUTED'}: {best_name} achieves {best_improvement:+.2f}% improvement at {noise_level*100}% noise"
    }
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'metrics.json'}")
    
    return output


if __name__ == "__main__":
    main()