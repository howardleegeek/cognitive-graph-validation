#!/usr/bin/env python3
"""
H1.369: Autocorrelation Threshold for CG Effectiveness
Tests the hypothesis that there exists a critical autocorrelation threshold
above which CG significantly outperforms baseline.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============== Architectures ==============

class BaselineArchitecture(nn.Module):
    """Standard separated encoding with late fusion."""
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
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Unified cognitive graph with early fusion."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
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
        
        # GNN layers for graph processing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
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
        
        # Create graph nodes (pad to same dimension)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


# ============== Dataset with Controlled Autocorrelation ==============

class AutocorrelationDataset(Dataset):
    """Dataset with controlled temporal autocorrelation."""
    
    def __init__(self, n_samples=500, seq_len=30, autocorr=0.5, obs_dim=8, lang_dim=32, action_dim=7):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.autocorr = autocorr
        self.obs_dim = obs_dim
        self.lang_dim = lang_dim
        self.action_dim = action_dim
        
        self.data = self._generate_data()
    
    def _generate_autocorr_sequence(self, length, dim, rho):
        """Generate sequence with specified autocorrelation."""
        # AR(1) process: x_t = rho * x_{t-1} + epsilon
        eps = np.random.randn(length, dim) * np.sqrt(1 - rho**2)
        x = np.zeros((length, dim))
        x[0] = np.random.randn(dim)
        for t in range(1, length):
            x[t] = rho * x[t-1] + eps[t]
        return x
    
    def _generate_data(self):
        """Generate dataset with controlled autocorrelation."""
        data = []
        for _ in range(self.n_samples):
            # Generate observations with autocorrelation
            obs_seq = self._generate_autocorr_sequence(
                self.seq_len, self.obs_dim, self.autocorr
            )
            
            # Generate language embedding (static per sequence)
            lang = np.random.randn(self.lang_dim).astype(np.float32)
            
            # Generate actions with autocorrelation (related to observations)
            action_seq = self._generate_autocorr_sequence(
                self.seq_len, self.action_dim, self.autocorr
            )
            
            # Add relationship between obs and action
            for t in range(self.seq_len):
                action_seq[t] += 0.3 * obs_seq[t, :self.action_dim]
            
            data.append({
                'observations': obs_seq.astype(np.float32),
                'language': lang,
                'actions': action_seq.astype(np.float32)
            })
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Return middle timestep for single-step prediction
        t = self.seq_len // 2
        return {
            'observation': torch.from_numpy(item['observations'][t]),
            'language': torch.from_numpy(item['language']),
            'action': torch.from_numpy(item['actions'][t])
        }


# ============== Training Functions ==============

def train_model(model, train_loader, val_loader, epochs=50, lr=3e-4):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_single_autocorr_test(autocorr, n_runs=3):
    """Run test for a single autocorrelation value."""
    results = []
    
    for run in range(n_runs):
        # Create datasets
        train_data = AutocorrelationDataset(n_samples=400, seq_len=30, autocorr=autocorr)
        val_data = AutocorrelationDataset(n_samples=100, seq_len=30, autocorr=autocorr)
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Train baseline
        baseline = BaselineArchitecture()
        baseline_loss = train_model(baseline, train_loader, val_loader)
        
        # Train cognitive graph
        cg = CognitiveGraphArchitecture()
        cg_loss = train_model(cg, train_loader, val_loader)
        
        improvement = (baseline_loss - cg_loss) / baseline_loss * 100
        results.append({
            'baseline_loss': baseline_loss,
            'cg_loss': cg_loss,
            'improvement': improvement
        })
    
    # Average results
    avg_baseline = np.mean([r['baseline_loss'] for r in results])
    avg_cg = np.mean([r['cg_loss'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    std_improvement = np.std([r['improvement'] for r in results])
    
    return {
        'autocorr': autocorr,
        'baseline_loss': avg_baseline,
        'cg_loss': avg_cg,
        'improvement_percent': avg_improvement,
        'improvement_std': std_improvement,
        'n_runs': n_runs
    }


def main():
    """Run the full autocorrelation threshold experiment."""
    print("=" * 60)
    print("H1.369: Autocorrelation Threshold for CG Effectiveness")
    print("=" * 60)
    
    # Test autocorrelation values
    autocorr_values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    results = []
    for rho in autocorr_values:
        print(f"\n[Testing ρ = {rho}]")
        result = run_single_autocorr_test(rho, n_runs=3)
        results.append(result)
        print(f"  Baseline MSE: {result['baseline_loss']:.6f}")
        print(f"  CG MSE: {result['cg_loss']:.6f}")
        print(f"  Improvement: {result['improvement_percent']:+.1f}% ± {result['improvement_std']:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'ρ':>6} | {'Baseline':>10} | {'CG':>10} | {'Improvement':>12}")
    print("-" * 50)
    for r in results:
        print(f"{r['autocorr']:>6.2f} | {r['baseline_loss']:>10.6f} | {r['cg_loss']:>10.6f} | {r['improvement_percent']:>+10.1f}%")
    
    # Find threshold (first ρ where improvement > 15%)
    threshold = None
    for r in results:
        if r['improvement_percent'] > 15:
            threshold = r['autocorr']
            break
    
    # Save results
    output = {
        'experiment': 'H1.369',
        'hypothesis': 'Autocorrelation threshold for CG effectiveness',
        'results': results,
        'threshold_found': threshold,
        'conclusion': 'SUPPORTED' if threshold and 0.4 <= threshold <= 0.6 else 'REFUTED'
    }
    
    output_path = Path(__file__).parent / 'results' / 'metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[Results saved to {output_path}]")
    print(f"\nConclusion: {output['conclusion']}")
    if threshold:
        print(f"Critical threshold ρ* ≈ {threshold}")
    
    return output


if __name__ == '__main__':
    main()