#!/usr/bin/env python3
"""
H1.401: Dimensionality Ratio Deep-Dive
Hypothesis: dim_ratio (physical_dim / total_dim) is the true moderator of CG advantage.
Previous H1.400 showed one config with 46.6% advantage at dim_ratio=0.7 - an outlier.
This experiment sweeps dim_ratio from 0.1 to 0.9 to find optimal split.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from data_loader import prepare_datasets

# Fixed seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Architectures
class BaselineArchitecture(nn.Module):
    """Separated architecture: JEPA-style encoder + LLM alignment"""
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
        # Late fusion - no gradient flow between modalities during encoding
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Unified cognitive graph architecture with tunable dimensionality ratio"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, dim_ratio=0.28):
        super().__init__()
        total_dim = 512
        physical_dim = int(total_dim * dim_ratio)
        semantic_dim = total_dim - physical_dim
        
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        # Early fusion: both modalities project to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers process unified representation
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=8, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Project to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Pad to create 2-node graph
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        # Stack as graph nodes
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode
        return self.decoder(attn_out.mean(dim=1))


def generate_synthetic_data(n_samples=500, seq_len=10, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate synthetic robotic manipulation data"""
    # Generate observations (endeffector pos, object pos, velocities)
    obs = np.random.randn(n_samples, seq_len, obs_dim).astype(np.float32)
    
    # Generate language embeddings (task instructions)
    lang = np.random.randn(n_samples, lang_dim).astype(np.float32)
    lang = lang / (np.linalg.norm(lang, axis=1, keepdims=True) + 1e-8)
    
    # Generate actions (joint velocities)
    # Actions depend on both observation and language
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)
    for i in range(n_samples):
        for t in range(seq_len):
            # Action is linear combination of obs and lang
            action = 0.3 * obs[i, t, :action_dim] + 0.5 * lang[i, :action_dim] + 0.2 * np.random.randn(action_dim)
            actions[i, t] = action
    
    # Flatten for training
    obs_flat = obs.reshape(n_samples * seq_len, obs_dim)
    lang_flat = np.repeat(lang, seq_len, axis=0)
    actions_flat = actions.reshape(n_samples * seq_len, action_dim)
    
    return obs_flat, lang_flat, actions_flat


def train_and_eval(model, train_obs, train_lang, train_act, val_obs, val_lang, val_act, epochs=30):
    """Train and evaluate model"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(
        torch.from_numpy(train_obs),
        torch.from_numpy(train_lang),
        torch.from_numpy(train_act)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        for obs, lang, act in train_loader:
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, act)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(val_obs), torch.from_numpy(val_lang))
        val_loss = criterion(val_pred, torch.from_numpy(val_act)).item()
    
    return val_loss


def run_dim_ratio_sweep():
    """Sweep dimensionality ratio to find optimal split"""
    print("=" * 60)
    print("H1.401: Dimensionality Ratio Deep-Dive")
    print("=" * 60)
    
    # Generate data
    n_samples = 500
    obs_dim, lang_dim, action_dim = 8, 32, 7
    
    obs, lang, actions = generate_synthetic_data(n_samples=n_samples)
    
    # Split train/val
    split = int(0.8 * len(obs))
    train_obs, val_obs = obs[:split], obs[split:]
    train_lang, val_lang = lang[:split], lang[split:]
    train_act, val_act = actions[:split], actions[split:]
    
    # Sweep dim_ratio from 0.1 to 0.9
    dim_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    print(f"\nSweeping dim_ratio from 0.1 to 0.9...")
    print("-" * 60)
    
    for dim_ratio in dim_ratios:
        # Train baseline
        torch.manual_seed(42)
        np.random.seed(42)
        
        baseline = BaselineArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim)
        baseline_loss = train_and_eval(baseline, train_obs, train_lang, train_act, val_obs, val_lang, val_act)
        
        # Train CG with this dim_ratio
        torch.manual_seed(42)
        np.random.seed(42)
        
        cg = CognitiveGraphArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, dim_ratio=dim_ratio)
        cg_loss = train_and_eval(cg, train_obs, train_lang, train_act, val_obs, val_lang, val_act)
        
        # Calculate improvement
        improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
        cg_wins = cg_loss < baseline_loss
        
        physical_dim = int(512 * dim_ratio)
        semantic_dim = 512 - physical_dim
        
        results.append({
            'dim_ratio': dim_ratio,
            'physical_dim': physical_dim,
            'semantic_dim': semantic_dim,
            'baseline_loss': baseline_loss,
            'cg_loss': cg_loss,
            'improvement_percent': improvement,
            'cg_wins': cg_wins
        })
        
        print(f"dim_ratio={dim_ratio:.1f} (phys={physical_dim}, sem={semantic_dim}): "
              f"baseline={baseline_loss:.6f}, CG={cg_loss:.6f}, "
              f"improvement={improvement:+.1f}%, CG wins={cg_wins}")
    
    # Find optimal
    best = max(results, key=lambda x: x['improvement_percent'])
    worst = min(results, key=lambda x: x['improvement_percent'])
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Best dim_ratio: {best['dim_ratio']} ({best['physical_dim']} phys / {best['semantic_dim']} sem)")
    print(f"  -> {best['improvement_percent']:+.1f}% improvement")
    print(f"Worst dim_ratio: {worst['dim_ratio']} ({worst['physical_dim']} phys / {worst['semantic_dim']} sem)")
    print(f"  -> {worst['improvement_percent']:+.1f}% improvement")
    
    # Calculate correlation
    dim_ratios_list = [r['dim_ratio'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    correlation = np.corrcoef(dim_ratios_list, improvements)[0, 1]
    print(f"\nCorrelation (dim_ratio vs improvement): r = {correlation:.3f}")
    
    # Save results
    output = {
        'experiment_id': 'H1.401',
        'hypothesis': 'dim_ratio is the true moderator of CG advantage',
        'dim_ratios_tested': dim_ratios,
        'results': results,
        'best_dim_ratio': best['dim_ratio'],
        'best_improvement': best['improvement_percent'],
        'correlation': correlation,
        'conclusion': 'SUPPORTED' if abs(correlation) > 0.5 else 'INCONCLUSIVE'
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nConclusion: {output['conclusion']}")
    print("Results saved to results.json")
    
    return output


if __name__ == '__main__':
    run_dim_ratio_sweep()
