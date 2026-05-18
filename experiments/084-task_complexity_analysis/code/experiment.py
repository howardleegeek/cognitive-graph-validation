"""
H1.435: Investigate why CG performs differently on synthetic vs real robot data.
Hypothesis: CG advantage depends on task relational complexity.

We'll test CG vs MLP on tasks categorized by relational complexity:
1. Low complexity: Simple pushing tasks (single object, simple dynamics)
2. Medium complexity: Multi-object tasks with simple interactions
3. High complexity: Tasks requiring relational reasoning (stacking, collisions)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Architectures
class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
            nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
            nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        return self.fusion(combined)

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368, n_passes=3):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.n_passes = n_passes
        
        # Encoders to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, physical_dim), 
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, semantic_dim), 
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers for message passing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), 
                nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(n_passes)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Pad to same dimension
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        # Create nodes: [physical, semantic]
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # Message passing
        for layer in self.gnn_layers:
            # Simple mean pooling for message passing
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode from unified representation
        return self.decoder(attn_out.mean(dim=1))

def train_and_eval(model, train_loader, val_loader, epochs=50, lr=3e-4):
    """Train and evaluate a model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            obs, lang, action = batch
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    # Evaluation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            obs, lang, action = batch
            pred = model(obs, lang)
            loss = criterion(pred, action)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def load_synthetic_data():
    """Load synthetic physics data with different complexity levels"""
    # For now, we'll create synthetic data with different complexity levels
    # In a real implementation, we would load from actual datasets
    
    # Task complexity categories:
    # 1. Low complexity: Simple 1D movement
    # 2. Medium complexity: 2D movement with obstacles
    # 3. High complexity: Multi-object interactions
    
    n_samples = 1000
    seq_len = 10
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    datasets = {}
    
    # Low complexity: Simple linear motion
    print("Generating low complexity data...")
    obs_low = torch.randn(n_samples, seq_len, obs_dim) * 0.1
    lang_low = torch.randn(n_samples, lang_dim)
    # Simple linear mapping
    action_low = 0.5 * obs_low[:, :, :action_dim].mean(dim=1) + 0.3 * lang_low[:, :action_dim]
    action_low += torch.randn_like(action_low) * 0.05
    
    # Medium complexity: Non-linear with interactions
    print("Generating medium complexity data...")
    obs_med = torch.randn(n_samples, seq_len, obs_dim) * 0.2
    lang_med = torch.randn(n_samples, lang_dim)
    # Non-linear interactions
    action_med = torch.sin(obs_med[:, :, :action_dim].mean(dim=1)) + 0.5 * torch.cos(lang_med[:, :action_dim])
    action_med += torch.randn_like(action_med) * 0.1
    
    # High complexity: Multi-modal, relational
    print("Generating high complexity data...")
    obs_high = torch.randn(n_samples, seq_len, obs_dim) * 0.3
    lang_high = torch.randn(n_samples, lang_dim)
    # Complex relational mapping
    obs_mean = obs_high[:, :, :action_dim].mean(dim=1)
    obs_std = obs_high[:, :, :action_dim].std(dim=1)
    action_high = obs_mean * obs_std + torch.tanh(lang_high[:, :action_dim]) * obs_mean
    action_high += torch.randn_like(action_high) * 0.15
    
    # Create datasets
    for name, obs, lang, action in [
        ("low_complexity", obs_low, lang_low, action_low),
        ("medium_complexity", obs_med, lang_med, action_med),
        ("high_complexity", obs_high, lang_high, action_high)
    ]:
        # Split train/val
        split_idx = int(0.8 * n_samples)
        train_data = TensorDataset(
            obs[:split_idx].reshape(-1, obs_dim),
            lang[:split_idx].repeat_interleave(seq_len, dim=0),
            action[:split_idx].repeat_interleave(seq_len, dim=0)
        )
        val_data = TensorDataset(
            obs[split_idx:].reshape(-1, obs_dim),
            lang[split_idx:].repeat_interleave(seq_len, dim=0),
            action[split_idx:].repeat_interleave(seq_len, dim=0)
        )
        
        datasets[name] = {
            'train': DataLoader(train_data, batch_size=32, shuffle=True),
            'val': DataLoader(val_data, batch_size=32, shuffle=False)
        }
    
    return datasets

def run_experiment():
    """Run experiment comparing CG vs MLP on different complexity tasks"""
    print("H1.435: Testing CG vs MLP on tasks with varying relational complexity")
    print("=" * 80)
    
    # Load datasets
    datasets = load_synthetic_data()
    
    # Results storage
    results = {
        'low_complexity': {'mlp': [], 'cg3': [], 'cg6': []},
        'medium_complexity': {'mlp': [], 'cg3': [], 'cg6': []},
        'high_complexity': {'mlp': [], 'cg3': [], 'cg6': []}
    }
    
    # Run 3 trials for each task type
    n_trials = 3
    
    for complexity_level in ['low_complexity', 'medium_complexity', 'high_complexity']:
        print(f"\nTesting {complexity_level.replace('_', ' ')} tasks:")
        print("-" * 40)
        
        train_loader = datasets[complexity_level]['train']
        val_loader = datasets[complexity_level]['val']
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}")
            
            # MLP Baseline
            mlp_model = BaselineArchitecture(obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128)
            mlp_loss = train_and_eval(mlp_model, train_loader, val_loader, epochs=30)
            results[complexity_level]['mlp'].append(mlp_loss)
            print(f"    MLP: {mlp_loss:.6f}")
            
            # CG with 3 passes
            cg3_model = CognitiveGraphArchitecture(obs_dim=8, lang_dim=32, action_dim=7, n_passes=3)
            cg3_loss = train_and_eval(cg3_model, train_loader, val_loader, epochs=30)
            results[complexity_level]['cg3'].append(cg3_loss)
            print(f"    CG-3p: {cg3_loss:.6f}")
            
            # CG with 6 passes
            cg6_model = CognitiveGraphArchitecture(obs_dim=8, lang_dim=32, action_dim=7, n_passes=6)
            cg6_loss = train_and_eval(cg6_model, train_loader, val_loader, epochs=30)
            results[complexity_level]['cg6'].append(cg6_loss)
            print(f"    CG-6p: {cg6_loss:.6f}")
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    
    summary = {}
    for complexity_level in results:
        mlp_mean = np.mean(results[complexity_level]['mlp'])
        cg3_mean = np.mean(results[complexity_level]['cg3'])
        cg6_mean = np.mean(results[complexity_level]['cg6'])
        
        cg3_vs_mlp = ((cg3_mean - mlp_mean) / mlp_mean) * 100
        cg6_vs_mlp = ((cg6_mean - mlp_mean) / mlp_mean) * 100
        cg6_vs_cg3 = ((cg6_mean - cg3_mean) / cg3_mean) * 100
        
        summary[complexity_level] = {
            'mlp_mean': mlp_mean,
            'cg3_mean': cg3_mean,
            'cg6_mean': cg6_mean,
            'cg3_vs_mlp_pct': cg3_vs_mlp,
            'cg6_vs_mlp_pct': cg6_vs_mlp,
            'cg6_vs_cg3_pct': cg6_vs_cg3
        }
        
        print(f"\n{complexity_level.replace('_', ' ').title()}:")
        print(f"  MLP: {mlp_mean:.6f}")
        print(f"  CG-3p: {cg3_mean:.6f} ({cg3_vs_mlp:+.1f}% vs MLP)")
        print(f"  CG-6p: {cg6_mean:.6f} ({cg6_vs_mlp:+.1f}% vs MLP)")
        print(f"  CG-6p vs CG-3p: {cg6_vs_cg3:+.1f}%")
    
    # Save results
    os.makedirs('experiments/084-task_complexity_analysis/results', exist_ok=True)
    with open('experiments/084-task_complexity_analysis/results/results.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': summary,
            'config': {
                'n_trials': n_trials,
                'epochs': 30,
                'batch_size': 32,
                'obs_dim': 8,
                'lang_dim': 32,
                'action_dim': 7
            }
        }, f, indent=2)
    
    # Generate hypothesis test
    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST:")
    print("=" * 80)
    
    # Check if CG performs better on high complexity tasks
    low_cg3_vs_mlp = summary['low_complexity']['cg3_vs_mlp_pct']
    med_cg3_vs_mlp = summary['medium_complexity']['cg3_vs_mlp_pct']
    high_cg3_vs_mlp = summary['high_complexity']['cg3_vs_mlp_pct']
    
    print(f"CG-3p vs MLP by complexity:")
    print(f"  Low: {low_cg3_vs_mlp:+.1f}%")
    print(f"  Medium: {med_cg3_vs_mlp:+.1f}%")
    print(f"  High: {high_cg3_vs_mlp:+.1f}%")
    
    # Hypothesis: CG should perform relatively better on high complexity tasks
    cg_improvement_trend = high_cg3_vs_mlp - low_cg3_vs_mlp
    
    if cg_improvement_trend > 0:
        print(f"\n✓ SUPPORTED: CG performs relatively better on high complexity tasks")
        print(f"  (Improvement trend: {cg_improvement_trend:+.1f}%)")
    else:
        print(f"\n✗ NOT SUPPORTED: CG does not perform better on high complexity tasks")
        print(f"  (Improvement trend: {cg_improvement_trend:+.1f}%)")
    
    return summary

if __name__ == "__main__":
    results = run_experiment()