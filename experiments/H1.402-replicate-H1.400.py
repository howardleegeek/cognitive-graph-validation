#!/usr/bin/env python3
"""
H1.402 - Replicate H1.400's data generation to verify claims
Investigate discrepancy between H1.400 (CG wins 100% across 96 configs) 
and H1.401 (CG loses across all dim_ratios)
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import LIBERODataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_synthetic_data(n_samples=500, seq_len=10, obs_dim=8, lang_dim=32, coupling_strength=0.7):
    """
    Generate synthetic data similar to H1.400's data generation.
    Based on findings: H1.400 used linear combination with coupling.
    """
    # Observations: random noise
    observations = np.random.randn(n_samples, seq_len, obs_dim) * 0.1
    
    # Language: random embeddings
    language = np.random.randn(n_samples, lang_dim) * 0.1
    
    # Actions: linear combination of obs and lang with coupling
    # H1.400 used: actions = 0.3*obs + 0.5*lang + noise
    # But with coupling_strength controlling cross-modal interaction
    
    # Create coupling: language influences observation dynamics
    coupling_matrix = np.random.randn(obs_dim, lang_dim) * coupling_strength
    
    # Generate coupled observations
    for i in range(n_samples):
        for t in range(seq_len):
            # Add language influence to observations
            lang_influence = np.dot(coupling_matrix, language[i])
            observations[i, t] += lang_influence * 0.1
    
    # Actions: combination of current obs and language
    # Need to match dimensions: obs_dim = 8, lang_dim = 32
    # We'll project language to obs_dim space
    lang_to_obs_proj = np.random.randn(lang_dim, obs_dim) * 0.1
    
    actions = np.zeros((n_samples, seq_len, obs_dim))
    for i in range(n_samples):
        for t in range(seq_len):
            # Project language to obs_dim space
            lang_projected = np.dot(language[i], lang_to_obs_proj)
            
            # Linear combination as in H1.400
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
    def __init__(self, obs_dim=8, lang_dim=32, hidden_dim=256, action_dim=8):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim // 2)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim // 2)
        self.decoder = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang):
        # obs shape: (batch, seq_len, obs_dim)
        # lang shape: (batch, lang_dim)
        
        # Encode observations
        batch_size, seq_len, _ = obs.shape
        obs_flat = obs.reshape(-1, obs.shape[-1])  # (batch*seq_len, obs_dim)
        obs_encoded = self.obs_encoder(obs_flat)
        obs_encoded = obs_encoded.reshape(batch_size, seq_len, -1)  # (batch, seq_len, hidden_dim//2)
        
        # Encode language and repeat for sequence length
        lang_encoded = self.lang_encoder(lang)  # (batch, hidden_dim//2)
        lang_encoded = lang_encoded.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim//2)
        
        # Concatenate and decode
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        return self.decoder(combined)

class CognitiveGraphModel(nn.Module):
    """Cognitive Graph: unified representation space"""
    def __init__(self, obs_dim=8, lang_dim=32, total_dim=512, action_dim=8, dim_ratio=0.5):
        super().__init__()
        self.physical_dim = int(total_dim * dim_ratio)
        self.semantic_dim = total_dim - self.physical_dim
        
        # Project both modalities into unified space
        self.obs_proj = nn.Linear(obs_dim, self.physical_dim)
        self.lang_proj = nn.Linear(lang_dim, self.semantic_dim)
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=4,
            batch_first=True
        )
        
        # GNN-like processing
        self.gnn = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, total_dim),
            nn.ReLU()
        )
        
        # Decoder to action space
        self.decoder = nn.Linear(total_dim, action_dim)
        
    def forward(self, obs, lang):
        # obs shape: (batch, seq_len, obs_dim)
        # lang shape: (batch, lang_dim)
        
        # Project to unified space
        batch_size, seq_len, _ = obs.shape
        obs_flat = obs.reshape(-1, obs.shape[-1])
        obs_proj_flat = self.obs_proj(obs_flat)
        obs_proj = obs_proj_flat.reshape(batch_size, seq_len, -1)
        
        lang_proj = self.lang_proj(lang).unsqueeze(1).repeat(1, seq_len, 1)
        
        # Combine
        combined = torch.cat([obs_proj, lang_proj], dim=-1)
        
        # Cross-modal attention
        attended, _ = self.attention(combined, combined, combined)
        
        # GNN processing
        processed = self.gnn(attended)
        
        # Decode to actions
        return self.decoder(processed)

def train_model(model, data, epochs=30, lr=1e-3):
    """Train a model on the data"""
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
        
    return losses[-1]  # Final loss

def run_experiment(dim_ratio=0.5, coupling_strength=0.7, epochs=30):
    """Run one experiment with given parameters"""
    # Generate data
    data = generate_synthetic_data(
        n_samples=500,
        seq_len=10,
        obs_dim=8,
        lang_dim=32,
        coupling_strength=coupling_strength
    )
    
    # Train baseline
    baseline = BaselineModel()
    baseline_loss = train_model(baseline, data, epochs=epochs)
    
    # Train cognitive graph
    cg = CognitiveGraphModel(dim_ratio=dim_ratio)
    cg_loss = train_model(cg, data, epochs=epochs)
    
    # Calculate improvement
    improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
    
    return {
        'dim_ratio': dim_ratio,
        'coupling_strength': coupling_strength,
        'baseline_loss': baseline_loss,
        'cg_loss': cg_loss,
        'improvement_percent': improvement,
        'cg_wins': improvement > 0
    }

def main():
    """Main experiment: Replicate H1.400's conditions"""
    print("H1.402 - Replicating H1.400's data generation")
    print("=" * 60)
    
    # Test multiple coupling strengths to understand H1.400's claim
    coupling_strengths = [0.0, 0.3, 0.5, 0.7, 0.9]
    dim_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = []
    
    for coupling in coupling_strengths:
        print(f"\nTesting coupling_strength={coupling}")
        print("-" * 40)
        
        for dim_ratio in dim_ratios:
            result = run_experiment(
                dim_ratio=dim_ratio,
                coupling_strength=coupling,
                epochs=30
            )
            results.append(result)
            
            win_str = "✓" if result['cg_wins'] else "✗"
            print(f"  dim_ratio={dim_ratio:.1f}: "
                  f"baseline={result['baseline_loss']:.6f}, "
                  f"CG={result['cg_loss']:.6f}, "
                  f"improvement={result['improvement_percent']:+.2f}% {win_str}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Count wins
    total_runs = len(results)
    cg_wins = sum(1 for r in results if r['cg_wins'])
    win_rate = (cg_wins / total_runs) * 100
    
    print(f"Total configurations tested: {total_runs}")
    print(f"CG wins: {cg_wins}/{total_runs} ({win_rate:.1f}%)")
    
    # Group by coupling strength
    for coupling in coupling_strengths:
        coupling_results = [r for r in results if r['coupling_strength'] == coupling]
        coupling_wins = sum(1 for r in coupling_results if r['cg_wins'])
        coupling_rate = (coupling_wins / len(coupling_results)) * 100 if coupling_results else 0
        
        avg_improvement = np.mean([r['improvement_percent'] for r in coupling_results])
        
        print(f"\nCoupling={coupling}:")
        print(f"  Win rate: {coupling_wins}/{len(coupling_results)} ({coupling_rate:.1f}%)")
        print(f"  Avg improvement: {avg_improvement:+.2f}%")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "experiments" / "H1.402"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    summary = {
        'experiment_id': 'H1.402',
        'description': 'Replicate H1.400 data generation to investigate discrepancy with H1.401',
        'total_configs': total_runs,
        'cg_win_rate': f"{win_rate:.1f}%",
        'avg_improvement': f"{np.mean([r['improvement_percent'] for r in results]):+.2f}%",
        'key_finding': f"CG win rate: {cg_wins}/{total_runs} ({win_rate:.1f}%) across coupling strengths {coupling_strengths}"
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    # Determine conclusion
    if win_rate > 90:  # Close to H1.400's 100% claim
        conclusion = "SUPPORTED_H1_400"
        print("\nCONCLUSION: H1.400's claim appears valid with proper data generation")
    elif win_rate > 50:
        conclusion = "PARTIALLY_SUPPORTED"
        print("\nCONCLUSION: CG has advantage but not 100% win rate")
    else:
        conclusion = "REFUTED_H1_400"
        print("\nCONCLUSION: H1.400's 100% win rate claim cannot be replicated")
    
    return conclusion, results

if __name__ == "__main__":
    conclusion, results = main()