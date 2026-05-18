#!/usr/bin/env python3
"""
H1.398 - Controlled Data Ablation: Identify what structural properties enable CG advantage

Purpose: Systematically vary data generation properties to understand when/why
the CG architecture outperforms the baseline.

Hypothesis: CG advantage emerges when data has strong cross-modal coupling
(observations and language are non-trivially entangled in the target function).

Ablation dimensions:
1. Cross-modal coupling strength (0.0 to 1.0)
2. Non-linearity degree (linear to highly non-linear)
3. Interaction order (1st order to 3rd order cross-terms)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation with Controlled Properties
# ============================================================

def generate_ablated_data(
    n_samples=500,
    obs_dim=8,
    lang_dim=32,
    action_dim=7,
    coupling_strength=0.5,
    nonlinearity=0.5,
    interaction_order=1,
    seed=42
):
    """
    Generate data with precisely controlled properties.
    
    Args:
        coupling_strength: How much obs and lang interact (0=independent, 1=fully coupled)
        nonlinearity: Degree of non-linear transformation (0=linear, 1=highly non-linear)
        interaction_order: Order of cross-modal interactions (1=linear, 2=quadratic, 3=cubic)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    obs = np.random.randn(n_samples, obs_dim).astype(np.float32)
    lang = np.random.randn(n_samples, lang_dim).astype(np.float32)
    
    # Base linear components (always present)
    W_obs = np.random.randn(obs_dim, action_dim).astype(np.float32) * 0.5
    W_lang = np.random.randn(lang_dim, action_dim).astype(np.float32) * 0.5
    
    actions = obs @ W_obs + lang @ W_lang
    
    # Cross-modal coupling (the key property for CG)
    if coupling_strength > 0:
        if interaction_order >= 1:
            # 1st order: element-wise products of obs and lang projections
            obs_proj = obs @ np.random.randn(obs_dim, action_dim).astype(np.float32) * 0.3
            lang_proj = lang @ np.random.randn(lang_dim, action_dim).astype(np.float32) * 0.3
            actions += coupling_strength * (obs_proj * lang_proj)
        
        if interaction_order >= 2:
            # 2nd order: outer product interactions
            for i in range(min(3, action_dim)):
                obs_vec = obs[:, i % obs_dim]
                lang_vec = lang[:, i % lang_dim]
                interaction = np.outer(obs_vec, lang_vec).diagonal() if len(obs_vec) == len(lang_vec) else obs_vec * lang_vec[:len(obs_vec)]
                W_cross = np.random.randn(min(obs_dim, lang_dim), action_dim).astype(np.float32) * 0.2
                actions += coupling_strength * 0.5 * (obs[:, :min(obs_dim, lang_dim)] * lang[:, :min(obs_dim, lang_dim)]) @ W_cross
        
        if interaction_order >= 3:
            # 3rd order: triple interactions
            actions += coupling_strength * 0.1 * np.sin(obs @ W_obs) * np.cos(lang @ W_lang)
    
    # Non-linear transformations
    if nonlinearity > 0:
        actions = actions + nonlinearity * 0.3 * np.sin(actions * 2)
        actions = actions + nonlinearity * 0.2 * (obs ** 2) @ (W_obs * 0.5)
    
    # Noise
    noise_level = 0.02
    actions += np.random.randn(n_samples, action_dim).astype(np.float32) * noise_level
    
    return obs, lang, actions


class AblationDataset(Dataset):
    def __init__(self, obs, lang, actions):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.lang = torch.tensor(lang, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return {
            'observation': self.obs[idx],
            'language': self.lang[idx],
            'action': self.actions[idx]
        }


# ============================================================
# Architectures (same as H1.397)
# ============================================================

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 hidden_dim=256, n_heads=2, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=n_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_physical(obs)
        z_sem = self.lang_to_semantic(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                val_loss += criterion(pred, batch['action']).item()
                n_batches += 1
        
        val_loss /= n_batches
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


# ============================================================
# Ablation Sweep
# ============================================================

def run_ablation():
    """Run the controlled ablation across data property dimensions."""
    
    # Define ablation grid
    coupling_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    nonlinearities = [0.0, 0.5, 1.0]
    interaction_orders = [1, 2, 3]
    
    n_samples = 500
    train_ratio = 0.8
    
    results = {
        'experiment_id': 'H1.398',
        'description': 'Controlled data ablation: identify structural properties enabling CG advantage',
        'config': {
            'n_samples': n_samples,
            'epochs': 20,
            'lr': 1e-3,
            'hidden_dim': 256,
            'n_heads': 2,
            'coupling_strengths': coupling_strengths,
            'nonlinearities': nonlinearities,
            'interaction_orders': interaction_orders
        },
        'ablation_results': []
    }
    
    total_configs = len(coupling_strengths) * len(nonlinearities) * len(interaction_orders)
    config_num = 0
    
    print("=" * 70)
    print("H1.398 - Controlled Data Ablation")
    print(f"Testing {total_configs} data property configurations")
    print("=" * 70)
    
    for coupling in coupling_strengths:
        for nonlin in nonlinearities:
            for order in interaction_orders:
                config_num += 1
                seed = 42 + config_num
                
                print(f"\n[{config_num}/{total_configs}] coupling={coupling}, nonlin={nonlin}, order={order}")
                
                # Generate data
                obs, lang, actions = generate_ablated_data(
                    n_samples=n_samples,
                    coupling_strength=coupling,
                    nonlinearity=nonlin,
                    interaction_order=order,
                    seed=seed
                )
                
                # Split
                n_train = int(n_samples * train_ratio)
                indices = np.random.permutation(n_samples)
                train_idx, val_idx = indices[:n_train], indices[n_train:]
                
                train_dataset = AblationDataset(obs[train_idx], lang[train_idx], actions[train_idx])
                val_dataset = AblationDataset(obs[val_idx], lang[val_idx], actions[val_idx])
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Train both models
                baseline = BaselineArchitecture(hidden_dim=256)
                baseline_loss = train_model(baseline, train_loader, val_loader, epochs=20, lr=1e-3)
                
                cg = CognitiveGraphArchitecture(hidden_dim=256, n_heads=2)
                cg_loss = train_model(cg, train_loader, val_loader, epochs=20, lr=1e-3)
                
                improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
                cg_wins = cg_loss < baseline_loss
                
                print(f"  Baseline: {baseline_loss:.6f}, CG: {cg_loss:.6f}, Improvement: {improvement:+.1f}%, CG wins: {cg_wins}")
                
                results['ablation_results'].append({
                    'coupling_strength': coupling,
                    'nonlinearity': nonlin,
                    'interaction_order': order,
                    'baseline_loss': round(float(baseline_loss), 6),
                    'cg_loss': round(float(cg_loss), 6),
                    'improvement_percent': round(float(improvement), 2),
                    'cg_wins': bool(cg_wins)
                })
    
    # Analysis
    cg_wins_total = sum(1 for r in results['ablation_results'] if r['cg_wins'])
    improvements = [r['improvement_percent'] for r in results['ablation_results']]
    
    # Correlation of each property with improvement
    couplings = [r['coupling_strength'] for r in results['ablation_results']]
    nonlinearities_list = [r['nonlinearity'] for r in results['ablation_results']]
    orders = [r['interaction_order'] for r in results['ablation_results']]
    
    results['analysis'] = {
        'cg_wins': cg_wins_total,
        'total_configs': len(results['ablation_results']),
        'avg_improvement': round(float(np.mean(improvements)), 2),
        'correlation_coupling': round(float(np.corrcoef(couplings, improvements)[0, 1]), 3),
        'correlation_nonlinearity': round(float(np.corrcoef(nonlinearities_list, improvements)[0, 1]), 3),
        'correlation_order': round(float(np.corrcoef(orders, improvements)[0, 1]), 3),
        'best_config': max(results['ablation_results'], key=lambda x: x['improvement_percent']),
        'worst_config': min(results['ablation_results'], key=lambda x: x['improvement_percent'])
    }
    
    # Find threshold where CG starts winning
    winning_configs = [r for r in results['ablation_results'] if r['cg_wins']]
    if winning_configs:
        min_coupling_for_win = min(r['coupling_strength'] for r in winning_configs)
        results['analysis']['min_coupling_for_cg_win'] = min_coupling_for_win
    else:
        results['analysis']['min_coupling_for_cg_win'] = None
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"CG wins: {cg_wins_total}/{len(results['ablation_results'])}")
    print(f"Avg improvement: {results['analysis']['avg_improvement']:+.1f}%")
    print(f"Correlation (coupling → improvement): {results['analysis']['correlation_coupling']:.3f}")
    print(f"Correlation (nonlinearity → improvement): {results['analysis']['correlation_nonlinearity']:.3f}")
    print(f"Correlation (order → improvement): {results['analysis']['correlation_order']:.3f}")
    
    best = results['analysis']['best_config']
    print(f"\nBest config: coupling={best['coupling_strength']}, nonlin={best['nonlinearity']}, order={best['interaction_order']}")
    print(f"  Improvement: {best['improvement_percent']:+.1f}%")
    
    worst = results['analysis']['worst_config']
    print(f"Worst config: coupling={worst['coupling_strength']}, nonlin={worst['nonlinearity']}, order={worst['interaction_order']}")
    print(f"  Improvement: {worst['improvement_percent']:+.1f}%")
    
    if results['analysis']['min_coupling_for_cg_win'] is not None:
        print(f"\nCG starts winning at coupling strength >= {results['analysis']['min_coupling_for_cg_win']}")
    else:
        print(f"\nCG never wins in this ablation space")
    
    return results


if __name__ == '__main__':
    results = run_ablation()
    
    output_path = Path('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/085-data_ablation/results/metrics.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
