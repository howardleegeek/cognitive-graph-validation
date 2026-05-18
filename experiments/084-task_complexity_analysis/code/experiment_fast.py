"""
H1.435: Investigate why CG performs differently on synthetic vs real robot data.
Hypothesis: CG advantage depends on task relational complexity.

Simplified version for faster execution.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Simplified Architectures
class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.lang_encoder = nn.Linear(lang_dim, latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        return self.fusion(combined)

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, n_passes=3):
        super().__init__()
        self.n_passes = n_passes
        latent_dim = 64
        
        # Encoders
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.lang_encoder = nn.Linear(lang_dim, latent_dim)
        
        # Simple GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim*2, latent_dim*2), 
                nn.ReLU()
            ) for _ in range(n_passes)
        ])
        
        # Decoder
        self.decoder = nn.Linear(latent_dim*2, action_dim)
    
    def forward(self, obs, lang):
        # Encode
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        
        # Create nodes: [obs, lang]
        nodes = torch.stack([obs_enc, lang_enc], dim=1)
        
        # Simple message passing
        for layer in self.gnn_layers:
            # Mean pooling for messages
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Decode from combined representation
        combined = nodes.mean(dim=1)
        return self.decoder(combined)

def train_and_eval_fast(model, train_loader, val_loader, epochs=10, lr=1e-3):
    """Fast training and evaluation"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Fast training
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            obs, lang, action = batch
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
    
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

def generate_synthetic_data_complexity(n_samples=200, seq_len=5, complexity='low'):
    """Generate synthetic data with different complexity levels"""
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    # Generate base data
    obs = torch.randn(n_samples, seq_len, obs_dim)
    lang = torch.randn(n_samples, lang_dim)
    
    # Different complexity levels
    if complexity == 'low':
        # Simple linear mapping
        action = 0.5 * obs[:, :, :action_dim].mean(dim=1) + 0.3 * lang[:, :action_dim]
        noise = torch.randn_like(action) * 0.05
        
    elif complexity == 'medium':
        # Non-linear with interactions
        obs_mean = obs[:, :, :action_dim].mean(dim=1)
        action = torch.sin(obs_mean) + 0.5 * torch.cos(lang[:, :action_dim])
        noise = torch.randn_like(action) * 0.1
        
    else:  # high
        # Complex relational mapping
        obs_mean = obs[:, :, :action_dim].mean(dim=1)
        obs_std = obs[:, :, :action_dim].std(dim=1)
        action = obs_mean * obs_std + torch.tanh(lang[:, :action_dim]) * obs_mean
        noise = torch.randn_like(action) * 0.15
    
    action = action + noise
    
    # Flatten for training
    obs_flat = obs.reshape(-1, obs_dim)
    lang_flat = lang.repeat_interleave(seq_len, dim=0)
    action_flat = action.repeat_interleave(seq_len, dim=0)
    
    # Split
    split_idx = int(0.8 * len(obs_flat))
    
    return {
        'train': (obs_flat[:split_idx], lang_flat[:split_idx], action_flat[:split_idx]),
        'val': (obs_flat[split_idx:], lang_flat[split_idx:], action_flat[split_idx:])
    }

def run_fast_experiment():
    """Run fast experiment"""
    print("H1.435: Testing CG vs MLP on tasks with varying relational complexity (FAST)")
    print("=" * 80)
    
    # Results storage
    results = {}
    
    # Test each complexity level
    complexities = ['low', 'medium', 'high']
    n_trials = 2  # Reduced for speed
    
    for complexity in complexities:
        print(f"\nTesting {complexity} complexity tasks:")
        print("-" * 40)
        
        results[complexity] = {'mlp': [], 'cg3': [], 'cg6': []}
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}")
            
            # Generate data
            data = generate_synthetic_data_complexity(complexity=complexity)
            
            # Create data loaders
            train_data = TensorDataset(*data['train'])
            val_data = TensorDataset(*data['val'])
            
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
            
            # MLP Baseline
            mlp_model = BaselineArchitecture(latent_dim=32)
            mlp_loss = train_and_eval_fast(mlp_model, train_loader, val_loader, epochs=5)
            results[complexity]['mlp'].append(mlp_loss)
            print(f"    MLP: {mlp_loss:.6f}")
            
            # CG with 3 passes
            cg3_model = CognitiveGraphArchitecture(n_passes=3)
            cg3_loss = train_and_eval_fast(cg3_model, train_loader, val_loader, epochs=5)
            results[complexity]['cg3'].append(cg3_loss)
            print(f"    CG-3p: {cg3_loss:.6f}")
            
            # CG with 6 passes
            cg6_model = CognitiveGraphArchitecture(n_passes=6)
            cg6_loss = train_and_eval_fast(cg6_model, train_loader, val_loader, epochs=5)
            results[complexity]['cg6'].append(cg6_loss)
            print(f"    CG-6p: {cg6_loss:.6f}")
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    
    summary = {}
    for complexity in complexities:
        mlp_mean = np.mean(results[complexity]['mlp'])
        cg3_mean = np.mean(results[complexity]['cg3'])
        cg6_mean = np.mean(results[complexity]['cg6'])
        
        cg3_vs_mlp = ((cg3_mean - mlp_mean) / mlp_mean) * 100
        cg6_vs_mlp = ((cg6_mean - mlp_mean) / mlp_mean) * 100
        
        summary[complexity] = {
            'mlp_mean': float(mlp_mean),
            'cg3_mean': float(cg3_mean),
            'cg6_mean': float(cg6_mean),
            'cg3_vs_mlp_pct': float(cg3_vs_mlp),
            'cg6_vs_mlp_pct': float(cg6_vs_mlp)
        }
        
        print(f"\n{complexity.title()} Complexity:")
        print(f"  MLP: {mlp_mean:.6f}")
        print(f"  CG-3p: {cg3_mean:.6f} ({cg3_vs_mlp:+.1f}% vs MLP)")
        print(f"  CG-6p: {cg6_mean:.6f} ({cg6_vs_mlp:+.1f}% vs MLP)")
    
    # Save results
    os.makedirs('experiments/084-task_complexity_analysis/results', exist_ok=True)
    with open('experiments/084-task_complexity_analysis/results/results_fast.json', 'w') as f:
        json.dump({
            'results': {k: {kk: [float(vv) for vv in vv] for kk, vv in v.items()} for k, v in results.items()},
            'summary': summary,
            'config': {
                'n_trials': n_trials,
                'epochs': 5,
                'batch_size': 16,
                'obs_dim': 8,
                'lang_dim': 32,
                'action_dim': 7
            }
        }, f, indent=2)
    
    # Hypothesis test
    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST:")
    print("=" * 80)
    
    # Check if CG performs better on high complexity tasks
    low_cg3 = summary['low']['cg3_vs_mlp_pct']
    med_cg3 = summary['medium']['cg3_vs_mlp_pct']
    high_cg3 = summary['high']['cg3_vs_mlp_pct']
    
    print(f"CG-3p vs MLP by complexity:")
    print(f"  Low: {low_cg3:+.1f}%")
    print(f"  Medium: {med_cg3:+.1f}%")
    print(f"  High: {high_cg3:+.1f}%")
    
    # Calculate improvement trend
    improvement_trend = high_cg3 - low_cg3
    
    if improvement_trend > 0:
        print(f"\n✓ SUPPORTED: CG performs relatively better on high complexity tasks")
        print(f"  (Improvement trend: {improvement_trend:+.1f}%)")
    else:
        print(f"\n✗ NOT SUPPORTED: CG does not perform better on high complexity tasks")
        print(f"  (Improvement trend: {improvement_trend:+.1f}%)")
    
    return summary

if __name__ == "__main__":
    results = run_fast_experiment()