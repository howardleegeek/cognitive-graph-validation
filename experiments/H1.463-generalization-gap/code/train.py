#!/usr/bin/env python3
"""
H1.463: Investigating Generalization Gap - Why CG Advantage Collapses on Real Data

Hypothesis: The 81.31% CG improvement in H1.461 was due to synthetic data having 
cleaner graph structure and less noise. Adding noise/perturbations to H1.461's 
synthetic data should cause similar performance collapse as seen in real robot data (H1.462).

Test approach:
1. Re-run H1.461's winning config (CG no-attention) on clean synthetic data (baseline)
2. Add increasing levels of noise to the synthetic data
3. Track when CG advantage disappears
4. Compare noise threshold to real robot data characteristics

Prediction: If noise causes CG collapse, then CG advantage is data-quality dependent,
not architecture-dependent. This would explain H1.462's failure on real robot data.
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


class SyntheticManipulationDataset(Dataset):
    """Synthetic manipulation dataset - clean version from H1.461."""
    
    def __init__(self, n_samples=200, seq_len=10, n_concepts=4, noise_level=0.0):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_concepts = n_concepts
        self.noise_level = noise_level
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        
        # Generate observation sequence (physical state)
        obs = torch.randn(self.seq_len, 8)  # 8-dim physical state
        
        # Generate language embedding (semantic)
        lang = torch.randn(32)  # 32-dim language embedding
        
        # Generate action target
        action = torch.randn(7)  # 7-DOF action
        
        # Add noise if specified
        if self.noise_level > 0:
            obs = obs + torch.randn_like(obs) * self.noise_level
            lang = lang + torch.randn_like(lang) * self.noise_level
            action = action + torch.randn_like(action) * self.noise_level
        
        return {
            'observation': obs,
            'language': lang,
            'action': action
        }


class BaselineConcat(nn.Module):
    """Simple concatenation baseline."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_last = obs[:, -1, :]
        return self.net(torch.cat([obs_last, lang], dim=-1))


class SimplifiedGNN(nn.Module):
    """Simplified GNN layer from H1.461."""
    
    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = max(hidden_dim // n_heads, 1)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.out_proj(out)


class CognitiveGraphNoAttention(nn.Module):
    """CG without attention - the H1.461 winner."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256, n_gnn_layers=3):
        super().__init__()
        
        # Project to common hidden space
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            SimplifiedGNN(hidden_dim, n_heads=4) for _ in range(n_gnn_layers)
        ])
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs, lang):
        batch_size = obs.shape[0]
        
        # Project to hidden space
        obs_hidden = self.obs_proj(obs)  # (B, seq_len, hidden)
        lang_hidden = self.lang_proj(lang).unsqueeze(1).expand(-1, obs_hidden.shape[1], -1)
        
        # Stack as graph: [obs nodes, lang nodes]
        graph_nodes = torch.cat([obs_hidden, lang_hidden], dim=1)
        
        # Apply GNN layers
        for gnn in self.gnn_layers:
            graph_nodes = graph_nodes + gnn(graph_nodes)
        
        # Get last obs node
        obs_last = graph_nodes[:, -1, :]
        
        # Aggregate lang nodes
        lang_agg = graph_nodes[:, obs.shape[1]:, :].mean(dim=1)
        
        return self.out_proj(torch.cat([obs_last, lang_agg], dim=-1))


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            obs = batch['observation']
            lang = batch['language']
            action = batch['action']
            
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                lang = batch['language']
                action = batch['action']
                pred = model(obs, lang)
                val_loss += criterion(pred, action).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_loss


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(noise_level):
    """Run experiment with given noise level."""
    # Create datasets
    train_dataset = SyntheticManipulationDataset(n_samples=200, seq_len=10, noise_level=noise_level)
    val_dataset = SyntheticManipulationDataset(n_samples=50, seq_len=10, noise_level=noise_level)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Baseline
    baseline = BaselineConcat(obs_dim=8, lang_dim=32, action_dim=7, hidden=256)
    baseline_loss = train_model(baseline, train_loader, val_loader)
    baseline_params = count_parameters(baseline)
    
    # CG no attention
    cg = CognitiveGraphNoAttention(obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256, n_gnn_layers=3)
    cg_loss = train_model(cg, train_loader, val_loader)
    cg_params = count_parameters(cg)
    
    # Calculate improvement
    if baseline_loss > 0:
        improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
    else:
        improvement = 0
    
    return {
        'noise_level': noise_level,
        'baseline_loss': baseline_loss,
        'baseline_params': baseline_params,
        'cg_loss': cg_loss,
        'cg_params': cg_params,
        'improvement_pct': improvement,
        'cg_wins': cg_loss < baseline_loss
    }


def main():
    print("=" * 60)
    print("H1.463: Investigating Generalization Gap")
    print("=" * 60)
    print("\nHypothesis: CG advantage is data-quality dependent.")
    print("Testing: Adding noise to synthetic data should cause CG collapse.\n")
    
    # Test different noise levels
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = []
    
    for noise in noise_levels:
        print(f"Testing noise level: {noise}")
        result = run_experiment(noise)
        results.append(result)
        print(f"  Baseline: {result['baseline_loss']:.6f}, CG: {result['cg_loss']:.6f}")
        print(f"  Improvement: {result['improvement_pct']:.2f}%, CG wins: {result['cg_wins']}")
    
    # Find noise threshold where CG advantage disappears
    cg_wins_at = [r['noise_level'] for r in results if r['cg_wins']]
    threshold = max(cg_wins_at) if cg_wins_at else 0.0
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Noise':<10} {'Baseline':<15} {'CG':<15} {'Improvement':<15} {'CG Wins'}")
    print("-" * 60)
    for r in results:
        print(f"{r['noise_level']:<10.2f} {r['baseline_loss']:<15.6f} {r['cg_loss']:<15.6f} {r['improvement_pct']:<15.2f} {r['cg_wins']}")
    
    print(f"\nNoise threshold where CG advantage disappears: {threshold}")
    
    # Determine conclusion
    if threshold < 0.1:
        conclusion = "CONFIRMED: CG advantage is highly noise-sensitive. Real robot data (with inherent noise) explains H1.462 collapse."
        cg_advantage_data_dependent = True
    else:
        conclusion = "PARTIAL: Noise has some effect but doesn't fully explain H1.462 collapse. Other factors may be involved."
        cg_advantage_data_dependent = False
    
    print(f"\nConclusion: {conclusion}")
    
    # Save results
    output = {
        'experiment': 'H1.463',
        'hypothesis': 'CG advantage is data-quality dependent (noise-sensitive)',
        'results': results,
        'noise_threshold': threshold,
        'conclusion': conclusion,
        'cg_advantage_data_dependent': cg_advantage_data_dependent
    }
    
    with open('results/metrics.json', 'w') as f:
        # Convert tensors to floats for JSON
        json_output = {
            'experiment': output['experiment'],
            'hypothesis': output['hypothesis'],
            'noise_threshold': output['noise_threshold'],
            'conclusion': output['conclusion'],
            'cg_advantage_data_dependent': output['cg_advantage_data_dependent'],
            'results': [
                {
                    'noise_level': r['noise_level'],
                    'baseline_loss': float(r['baseline_loss']),
                    'baseline_params': r['baseline_params'],
                    'cg_loss': float(r['cg_loss']),
                    'cg_params': r['cg_params'],
                    'improvement_pct': float(r['improvement_pct']),
                    'cg_wins': r['cg_wins']
                }
                for r in results
            ]
        }
        json.dump(json_output, f, indent=2)
    
    print("\nResults saved to results/metrics.json")
    return output


if __name__ == '__main__':
    main()
