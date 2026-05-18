#!/usr/bin/env python3
"""
H1.403 - Training Dynamics Investigation (Fast Version)
Test if CG needs more epochs or different learning rates.

Hypothesis: CG's cross-modal attention and GNN processing require more 
training epochs to converge compared to the simpler baseline concatenation.

Test Plan:
1. Train both models for 30, 50, 100 epochs
2. Test learning rates: 1e-3, 5e-3
3. Use smaller hidden dim for speed
4. Use best dim_ratio from H1.402 (0.1) and coupling=0.0
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_synthetic_data(n_samples=300, seq_len=10, obs_dim=8, lang_dim=32, coupling_strength=0.0):
    """Generate synthetic data with minimal coupling (best case for CG from H1.402)."""
    observations = np.random.randn(n_samples, seq_len, obs_dim) * 0.1
    language = np.random.randn(n_samples, lang_dim) * 0.1
    coupling_matrix = np.random.randn(obs_dim, lang_dim) * coupling_strength
    
    for i in range(n_samples):
        for t in range(seq_len):
            lang_influence = np.dot(coupling_matrix, language[i])
            observations[i, t] += lang_influence * 0.1
    
    lang_to_obs_proj = np.random.randn(lang_dim, obs_dim) * 0.1
    actions = np.zeros((n_samples, seq_len, obs_dim))
    for i in range(n_samples):
        for t in range(seq_len):
            lang_projected = np.dot(language[i], lang_to_obs_proj)
            actions[i, t] = (
                0.3 * observations[i, t] + 
                0.5 * lang_projected + 
                np.random.randn(obs_dim) * 0.05
            )
    
    return {
        'observations': torch.FloatTensor(observations),
        'language': torch.FloatTensor(language),
        'actions': torch.FloatTensor(actions)
    }

class BaselineModel(nn.Module):
    """Baseline: separate encoders for obs and lang, concatenated"""
    def __init__(self, obs_dim=8, lang_dim=32, hidden_dim=128, action_dim=8):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim // 2)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim // 2)
        self.decoder = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        obs_flat = obs.reshape(-1, obs.shape[-1])
        obs_encoded = F.relu(self.obs_encoder(obs_flat))
        obs_encoded = obs_encoded.reshape(batch_size, seq_len, -1)
        lang_encoded = F.relu(self.lang_encoder(lang)).unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        return self.decoder(combined)

class CognitiveGraphModel(nn.Module):
    """Cognitive Graph: unified representation space"""
    def __init__(self, obs_dim=8, lang_dim=32, total_dim=256, action_dim=8, dim_ratio=0.1):
        super().__init__()
        self.physical_dim = int(total_dim * dim_ratio)
        self.semantic_dim = total_dim - self.physical_dim
        
        self.obs_proj = nn.Linear(obs_dim, self.physical_dim)
        self.lang_proj = nn.Linear(lang_dim, self.semantic_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=2,
            batch_first=True
        )
        
        self.gnn = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, total_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Linear(total_dim, action_dim)
        
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        obs_flat = obs.reshape(-1, obs.shape[-1])
        obs_proj_flat = self.obs_proj(obs_flat)
        obs_proj = obs_proj_flat.reshape(batch_size, seq_len, -1)
        lang_proj = self.lang_proj(lang).unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat([obs_proj, lang_proj], dim=-1)
        attended, _ = self.attention(combined, combined, combined)
        processed = self.gnn(attended)
        return self.decoder(processed)

def train_model_with_history(model, data, epochs=100, lr=1e-3):
    """Train a model and return loss history"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    obs = data['observations']
    lang = data['language']
    actions = data['actions']
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_actions = model(obs, lang)
        loss = criterion(pred_actions, actions)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return losses

def run_single_config(epochs, lr, dim_ratio=0.1, coupling=0.0, seed=42):
    """Run a single configuration and return final losses"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data = generate_synthetic_data(
        n_samples=300,
        seq_len=10,
        obs_dim=8,
        lang_dim=32,
        coupling_strength=coupling
    )
    
    baseline = BaselineModel()
    baseline_losses = train_model_with_history(baseline, data, epochs=epochs, lr=lr)
    
    cg = CognitiveGraphModel(dim_ratio=dim_ratio)
    cg_losses = train_model_with_history(cg, data, epochs=epochs, lr=lr)
    
    return {
        'baseline_final_loss': baseline_losses[-1],
        'cg_final_loss': cg_losses[-1],
        'baseline_losses': baseline_losses,
        'cg_losses': cg_losses,
        'improvement': (baseline_losses[-1] - cg_losses[-1]) / baseline_losses[-1] * 100,
        'cg_wins': cg_losses[-1] < baseline_losses[-1]
    }

def main():
    print("=" * 80)
    print("H1.403 - Training Dynamics Investigation (Fast Version)")
    print("Testing if CG needs more epochs or different learning rates")
    print("=" * 80)
    
    # Test configurations (reduced for speed)
    epoch_values = [30, 50, 100]
    lr_values = [1e-3, 5e-3]
    
    # Use best dim_ratio from H1.402 (0.1) and coupling=0.0
    dim_ratio = 0.1
    coupling = 0.0
    
    results = []
    
    for epochs in epoch_values:
        for lr in lr_values:
            print(f"\nTesting epochs={epochs}, lr={lr}")
            result = run_single_config(epochs=epochs, lr=lr, dim_ratio=dim_ratio, coupling=coupling)
            result['epochs'] = epochs
            result['lr'] = lr
            results.append(result)
            
            print(f"  Baseline loss: {result['baseline_final_loss']:.6f}")
            print(f"  CG loss: {result['cg_final_loss']:.6f}")
            print(f"  Improvement: {result['improvement']:+.2f}%")
            print(f"  CG wins: {result['cg_wins']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    best_cg_result = min(results, key=lambda r: r['cg_final_loss'])
    best_improvement_result = max(results, key=lambda r: r['improvement'])
    
    print(f"\nBest CG loss: {best_cg_result['cg_final_loss']:.6f} (epochs={best_cg_result['epochs']}, lr={best_cg_result['lr']})")
    print(f"Best improvement: {best_improvement_result['improvement']:+.2f}% (epochs={best_improvement_result['epochs']}, lr={best_improvement_result['lr']})")
    
    cg_wins = sum(1 for r in results if r['cg_wins'])
    print(f"\nCG wins in {cg_wins}/{len(results)} configurations")
    
    # Analyze by epochs
    print("\n--- By Epochs ---")
    for epochs in epoch_values:
        epoch_results = [r for r in results if r['epochs'] == epochs]
        avg_improvement = np.mean([r['improvement'] for r in epoch_results])
        wins = sum(1 for r in epoch_results if r['cg_wins'])
        print(f"  epochs={epochs}: avg_improvement={avg_improvement:+.2f}%, wins={wins}/{len(epoch_results)}")
    
    # Analyze by learning rate
    print("\n--- By Learning Rate ---")
    for lr in lr_values:
        lr_results = [r for r in results if r['lr'] == lr]
        avg_improvement = np.mean([r['improvement'] for r in lr_results])
        wins = sum(1 for r in lr_results if r['cg_wins'])
        print(f"  lr={lr}: avg_improvement={avg_improvement:+.2f}%, wins={wins}/{len(lr_results)}")
    
    # Save results
    output = {
        'experiment': 'H1.403',
        'description': 'Training dynamics investigation',
        'dim_ratio': dim_ratio,
        'coupling': coupling,
        'results': [{
            'epochs': r['epochs'],
            'lr': r['lr'],
            'baseline_loss': r['baseline_final_loss'],
            'cg_loss': r['cg_final_loss'],
            'improvement': r['improvement'],
            'cg_wins': r['cg_wins']
        } for r in results],
        'summary': {
            'total_configs': len(results),
            'cg_wins': cg_wins,
            'best_cg_loss': best_cg_result['cg_final_loss'],
            'best_improvement': best_improvement_result['improvement'],
            'best_epochs': best_improvement_result['epochs'],
            'best_lr': best_improvement_result['lr']
        }
    }
    
    with open('experiments/H1.403-results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to experiments/H1.403-results.json")
    
    # Plot loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot by epochs (with lr=1e-3)
    ax1 = axes[0]
    for epochs in epoch_values:
        result = next(r for r in results if r['epochs'] == epochs and r['lr'] == 1e-3)
        ax1.plot(result['baseline_losses'], label=f'Baseline (e={epochs})', linestyle='--')
        ax1.plot(result['cg_losses'], label=f'CG (e={epochs})', linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves by Epochs (lr=1e-3)')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Bar chart of final losses
    ax2 = axes[1]
    x = np.arange(len(epoch_values))
    width = 0.35
    baseline_losses = [next(r for r in results if r['epochs'] == e and r['lr'] == 1e-3)['baseline_final_loss'] for e in epoch_values]
    cg_losses = [next(r for r in results if r['epochs'] == e and r['lr'] == 1e-3)['cg_final_loss'] for e in epoch_values]
    ax2.bar(x - width/2, baseline_losses, width, label='Baseline')
    ax2.bar(x + width/2, cg_losses, width, label='Cognitive Graph')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Final Loss')
    ax2.set_title('Final Loss by Epochs (lr=1e-3)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(epoch_values)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('experiments/H1.403-loss-curves.png', dpi=150)
    print("Loss curves saved to experiments/H1.403-loss-curves.png")
    
    return output

if __name__ == '__main__':
    main()