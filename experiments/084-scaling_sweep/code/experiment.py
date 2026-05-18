#!/usr/bin/env python3
"""
H1.397 - Scaling Sweep: Test Config A (256-dim, 2-heads) across full complexity range

Purpose: Verify scaling behavior of the optimized CG architecture and identify
optimal model size for each complexity level.

Config A (from H1.396): 256 hidden dim, 2 attention heads, 20 epochs, lr=1e-3

Complexity levels: [20, 60, 100, 150, 170, 200, 300, 400, 500, 600]
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation - Complexity-controlled synthetic tasks
# ============================================================

def generate_complexity_data(complexity, n_samples=500, obs_dim=8, lang_dim=32, action_dim=7):
    """
    Generate synthetic data with controlled complexity.
    
    Complexity controls:
    - Non-linearity of the mapping
    - Number of interaction terms
    - Noise level
    
    Higher complexity = harder to learn, more benefit from structured representations.
    """
    np.random.seed(42 + complexity)
    torch.manual_seed(42 + complexity)
    
    # Generate observations and language embeddings
    obs = np.random.randn(n_samples, obs_dim).astype(np.float32)
    lang = np.random.randn(n_samples, lang_dim).astype(np.float32)
    
    # Generate actions based on complexity-dependent function
    # Low complexity: mostly linear
    # High complexity: highly non-linear with cross-modal interactions
    
    # Base linear component
    W_obs = np.random.randn(obs_dim, action_dim).astype(np.float32) * 0.5
    W_lang = np.random.randn(lang_dim, action_dim).astype(np.float32) * 0.5
    
    actions = obs @ W_obs + lang @ W_lang
    
    # Add non-linear interactions scaled by complexity
    complexity_factor = complexity / 100.0
    
    # Cross-modal interaction terms (this is where CG should excel)
    if complexity > 50:
        # Create interaction features
        obs_expanded = np.tile(obs[:, :, np.newaxis], (1, 1, lang_dim)).reshape(n_samples, obs_dim * lang_dim)
        W_interaction = np.random.randn(obs_dim * lang_dim, action_dim).astype(np.float32) * 0.1 * complexity_factor
        actions = actions + obs_expanded @ W_interaction
    
    # Non-linear transformations
    if complexity > 100:
        actions = actions + np.sin(actions) * 0.3 * complexity_factor
        actions = actions + (obs ** 2) @ (W_obs * 0.2 * complexity_factor)
    
    # Higher-order interactions for very complex tasks
    if complexity > 300:
        lang_squared = lang ** 2
        W_high = np.random.randn(lang_dim, action_dim).astype(np.float32) * 0.05 * (complexity / 300.0)
        actions = actions + lang_squared @ W_high
    
    # Add noise (inversely proportional to complexity to maintain signal)
    noise_level = 0.01 * (1 + complexity_factor * 0.5)
    actions = actions + np.random.randn(n_samples, action_dim).astype(np.float32) * noise_level
    
    return obs, lang, actions


class ComplexityDataset(Dataset):
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
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Separated architecture: encode obs and lang separately, then concatenate."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """
    Config A (from H1.396): 256 hidden dim, 2 attention heads
    
    Unified representation with cross-modal attention.
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 hidden_dim=256, n_heads=2, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Map observations to physical subspace
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        
        # Map language to semantic subspace
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers for message passing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention with 2 heads (Config A)
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=n_heads, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode into unified space
        z_phys = self.obs_to_physical(obs)  # [batch, physical_dim]
        z_sem = self.lang_to_semantic(lang)  # [batch, semantic_dim]
        
        # Create node representations (pad to same dimension)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))  # [batch, total_dim]
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)  # [batch, total_dim]
        
        # Stack as graph nodes
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [batch, 2, total_dim]
        
        # Message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode from mean-pooled nodes
        return self.decoder(attn_out.mean(dim=1))


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    """Train model and return final validation loss."""
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
        
        # Validation
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


def run_complexity_sweep():
    """Run the full scaling sweep across all complexity levels."""
    
    complexity_levels = [20, 60, 100, 150, 170, 200, 300, 400, 500, 600]
    n_samples = 500
    train_ratio = 0.8
    
    results = {
        'experiment_id': 'H1.397',
        'description': 'Scaling sweep: Config A (256-dim, 2-heads) across full complexity range',
        'config': {
            'hidden_dim': 256,
            'attention_heads': 2,
            'epochs': 20,
            'learning_rate': 1e-3,
            'n_samples': n_samples,
            'complexity_levels': complexity_levels
        },
        'per_complexity': []
    }
    
    print("=" * 60)
    print("H1.397 - Scaling Sweep")
    print("Config A: 256-dim, 2-heads, 20 epochs, lr=1e-3")
    print("=" * 60)
    
    for complexity in complexity_levels:
        print(f"\n--- Complexity = {complexity} ---")
        
        # Generate data
        obs, lang, actions = generate_complexity_data(complexity, n_samples)
        
        # Split into train/val
        n_train = int(n_samples * train_ratio)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        train_dataset = ComplexityDataset(obs[train_idx], lang[train_idx], actions[train_idx])
        val_dataset = ComplexityDataset(obs[val_idx], lang[val_idx], actions[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train baseline
        baseline = BaselineArchitecture(hidden_dim=256)
        baseline_loss = train_model(baseline, train_loader, val_loader, epochs=20, lr=1e-3)
        
        # Train CG (Config A)
        cg = CognitiveGraphArchitecture(hidden_dim=256, n_heads=2)
        cg_loss = train_model(cg, train_loader, val_loader, epochs=20, lr=1e-3)
        
        # Calculate improvement
        improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
        cg_wins = cg_loss < baseline_loss
        
        print(f"  Baseline loss: {baseline_loss:.6f}")
        print(f"  CG loss:       {cg_loss:.6f}")
        print(f"  Improvement:   {improvement:+.1f}%")
        print(f"  CG wins:       {cg_wins}")
        
        results['per_complexity'].append({
            'complexity': int(complexity),
            'baseline_loss': round(float(baseline_loss), 6),
            'cg_loss': round(float(cg_loss), 6),
            'improvement_percent': round(float(improvement), 2),
            'cg_wins': bool(cg_wins)
        })
    
    # Summary statistics
    improvements = [r['improvement_percent'] for r in results['per_complexity']]
    cg_wins_count = sum(1 for r in results['per_complexity'] if r['cg_wins'])
    
    results['summary'] = {
        'avg_improvement': round(float(np.mean(improvements)), 2),
        'std_improvement': round(float(np.std(improvements)), 2),
        'max_improvement': round(float(max(improvements)), 2),
        'min_improvement': round(float(min(improvements)), 2),
        'cg_wins': int(cg_wins_count),
        'cg_wins_total': len(complexity_levels),
        'best_complexity': int(results['per_complexity'][int(np.argmax(improvements))]['complexity']),
        'worst_complexity': int(results['per_complexity'][int(np.argmin(improvements))]['complexity'])
    }
    
    # Correlation analysis
    complexities = [r['complexity'] for r in results['per_complexity']]
    corr = float(np.corrcoef(complexities, improvements)[0, 1])
    results['summary']['complexity_improvement_correlation'] = round(corr, 3)
    
    # Quadratic fit analysis
    if len(complexities) > 2:
        coeffs = np.polyfit(complexities, improvements, 2)
        peak_complexity = float(-coeffs[1] / (2 * coeffs[2])) if coeffs[2] != 0 else float('inf')
        results['summary']['quadratic_fit'] = {
            'coefficients': [round(float(c), 6) for c in coeffs],
            'peak_complexity': round(peak_complexity, 1),
            'is_inverted_u': bool(coeffs[2] < 0)
        }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Avg improvement: {results['summary']['avg_improvement']:+.1f}%")
    print(f"CG wins: {results['summary']['cg_wins']}/{results['summary']['cg_wins_total']}")
    print(f"Best at complexity: {results['summary']['best_complexity']}")
    print(f"Correlation (complexity vs improvement): {results['summary']['complexity_improvement_correlation']:.3f}")
    
    if 'quadratic_fit' in results['summary']:
        qf = results['summary']['quadratic_fit']
        print(f"Quadratic fit peak: {qf['peak_complexity']}")
        print(f"Inverted-U pattern: {qf['is_inverted_u']}")
    
    return results


if __name__ == '__main__':
    results = run_complexity_sweep()
    
    # Save results
    output_path = Path('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-scaling_sweep/results/metrics.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
