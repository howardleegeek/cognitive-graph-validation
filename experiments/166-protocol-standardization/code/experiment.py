#!/usr/bin/env python3
"""
H1.395 - Protocol Standardization Experiment
=============================================
Goal: Resolve discrepancy between H1.393 (CG wins at medium complexity) 
and H1.394 (CG loses everywhere) by using identical data generation 
and training parameters.

Approach: Run both experiment configurations with:
- Same random seed (42)
- Same data generation parameters
- Same training epochs (20)
- Same train/val split
- Compare results directly
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List
import pickle

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============ ARCHITECTURES ============

class BaselineArchitecture(nn.Module):
    """Separated architecture: JEPA + LLM alignment"""
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
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Unified cognitive graph architecture"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
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
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), 
                nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create unified graph nodes
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN processing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


# ============ DATA GENERATION ============

OBS_DIM = 8
LANG_DIM = 32
ACTION_DIM = 7

class SyntheticLiberoDataset(Dataset):
    """Synthetic LIBERO-style dataset with controlled complexity"""
    
    def __init__(self, n_samples=500, complexity=100, seq_len=10, seed=42):
        self.n_samples = n_samples
        self.complexity = complexity
        self.seq_len = seq_len
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data based on complexity parameter
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate data with complexity-controlled patterns"""
        data = []
        
        # Complexity affects:
        # 1. Number of relevant features (capped at obs_dim)
        # 2. Noise level
        # 3. Non-linearity of mapping
        
        n_relevant = min(OBS_DIM, max(2, int(self.complexity / 50)))
        noise_level = 0.1 + (self.complexity / 1000)  # 0.1 to 0.7
        
        for i in range(self.n_samples):
            # Observation: 8-dim state
            obs = np.random.randn(OBS_DIM).astype(np.float32)
            
            # Make first n_relevant features correlated with action
            action_base = np.random.randn(ACTION_DIM).astype(np.float32)
            for j in range(n_relevant):
                action_base[j % ACTION_DIM] += obs[j] * (1.0 + self.complexity/200)
            
            # Add noise
            action = action_base + np.random.randn(ACTION_DIM).astype(np.float32) * noise_level
            
            # Language: 32-dim embedding
            lang = np.random.randn(LANG_DIM).astype(np.float32)
            # Make language partially predictive (fixed: use same n_relevant capped at 32)
            n_lang_relevant = min(n_relevant, LANG_DIM)
            lang[:n_lang_relevant] = lang[:n_lang_relevant] + obs[:n_lang_relevant] * 0.5
            
            data.append({
                'observation': obs,
                'action': action,
                'language': lang
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'observation': torch.tensor(self.data[idx]['observation']),
            'action': torch.tensor(self.data[idx]['action']),
            'language': torch.tensor(self.data[idx]['language'])
        }


def create_dataloaders(complexity, n_train=400, n_val=100, batch_size=32, seed=42):
    """Create train/val dataloaders with same seed"""
    train_dataset = SyntheticLiberoDataset(n_samples=n_train, complexity=complexity, seed=seed)
    val_dataset = SyntheticLiberoDataset(n_samples=n_val, complexity=complexity, seed=seed+1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# ============ TRAINING ============

def train_model(model, train_loader, val_loader, epochs=20, lr=3e-4):
    """Train model and return final validation loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                loss = criterion(pred, batch['action'])
                val_loss += loss.item()
        
        best_val_loss = min(best_val_loss, val_loss)
    
    return best_val_loss


# ============ EXPERIMENT ============

def run_experiment(complexity, seed=42, epochs=20):
    """Run single experiment: baseline vs CG"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader, val_loader = create_dataloaders(
        complexity=complexity, 
        n_train=400, 
        n_val=100, 
        batch_size=32,
        seed=seed
    )
    
    # Baseline
    baseline = BaselineArchitecture()
    baseline_loss = train_model(baseline, train_loader, val_loader, epochs=epochs)
    
    # Cognitive Graph
    cg = CognitiveGraphArchitecture()
    cg_loss = train_model(cg, train_loader, val_loader, epochs=epochs)
    
    # Calculate improvement
    improvement = (baseline_loss - cg_loss) / baseline_loss * 100
    
    return {
        'complexity': complexity,
        'baseline_loss': baseline_loss,
        'cg_loss': cg_loss,
        'improvement_percent': improvement,
        'cg_wins': improvement > 0
    }


def run_h1_393_style():
    """Run H1.393 style experiment (7 configs from simple to very_complex)"""
    print("\n" + "="*60)
    print("Running H1.393 style experiment")
    print("="*60)
    
    # Same configurations as H1.393
    configs = [
        ('simple', 20),
        ('simple2', 60),
        ('medium', 100),
        ('threshold', 150),
        ('crossover', 170),
        ('complex', 300),
        ('very_complex', 550)
    ]
    
    results = []
    for name, complexity in configs:
        result = run_experiment(complexity=complexity, seed=42, epochs=20)
        result['name'] = name
        results.append(result)
        print(f"  {name}: complexity={complexity}, improvement={result['improvement_percent']:.1f}%, CG wins={result['cg_wins']}")
    
    # Calculate correlation
    complexities = [r['complexity'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    correlation = np.corrcoef(complexities, improvements)[0, 1]
    
    avg_improvement = np.mean(improvements)
    cg_wins = sum(1 for r in results if r['cg_wins'])
    
    return {
        'style': 'H1.393',
        'results': results,
        'correlation': correlation,
        'avg_improvement': avg_improvement,
        'cg_wins': cg_wins,
        'total': len(results)
    }


def run_h1_394_style():
    """Run H1.394 style experiment (8 complexity levels)"""
    print("\n" + "="*60)
    print("Running H1.394 style experiment")
    print("="*60)
    
    # Same configurations as H1.394
    complexity_levels = [50, 100, 150, 200, 300, 400, 500, 600]
    
    results = []
    for complexity in complexity_levels:
        result = run_experiment(complexity=complexity, seed=42, epochs=20)
        results.append(result)
        print(f"  complexity={complexity}: improvement={result['improvement_percent']:.1f}%, CG wins={result['cg_wins']}")
    
    # Calculate correlation
    complexities = [r['complexity'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    correlation = np.corrcoef(complexities, improvements)[0, 1]
    
    avg_improvement = np.mean(improvements)
    cg_wins = sum(1 for r in results if r['cg_wins'])
    
    return {
        'style': 'H1.394',
        'results': results,
        'correlation': correlation,
        'avg_improvement': avg_improvement,
        'cg_wins': cg_wins,
        'total': len(results)
    }


def run_unified_style():
    """Run unified experiment with both styles using same data"""
    print("\n" + "="*60)
    print("Running UNIFIED style experiment (same data for both)")
    print("="*60)
    
    # Generate one set of complexities
    all_complexities = sorted(set([20, 60, 100, 150, 170, 200, 300, 400, 500, 600]))
    
    results = []
    for complexity in all_complexities:
        result = run_experiment(complexity=complexity, seed=42, epochs=20)
        results.append(result)
        print(f"  complexity={complexity}: improvement={result['improvement_percent']:.1f}%, CG wins={result['cg_wins']}")
    
    # Calculate correlation
    complexities = [r['complexity'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    correlation = np.corrcoef(complexities, improvements)[0, 1]
    
    avg_improvement = np.mean(improvements)
    cg_wins = sum(1 for r in results if r['cg_wins'])
    
    return {
        'style': 'UNIFIED',
        'results': results,
        'correlation': correlation,
        'avg_improvement': avg_improvement,
        'cg_wins': cg_wins,
        'total': len(results)
    }


# ============ MAIN ============

if __name__ == "__main__":
    print("H1.395 - Protocol Standardization Experiment")
    print("="*60)
    print("Resolving H1.393 vs H1.394 discrepancy")
    print("Using identical seeds, data generation, and training parameters\n")
    
    # Run all three styles
    h1_393_results = run_h1_393_style()
    h1_394_results = run_h1_394_style()
    unified_results = run_unified_style()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nH1.393 style:")
    print(f"  Correlation: {h1_393_results['correlation']:.3f}")
    print(f"  Avg improvement: {h1_393_results['avg_improvement']:.1f}%")
    print(f"  CG wins: {h1_393_results['cg_wins']}/{h1_393_results['total']}")
    
    print(f"\nH1.394 style:")
    print(f"  Correlation: {h1_394_results['correlation']:.3f}")
    print(f"  Avg improvement: {h1_394_results['avg_improvement']:.1f}%")
    print(f"  CG wins: {h1_394_results['cg_wins']}/{h1_394_results['total']}")
    
    print(f"\nUNIFIED style:")
    print(f"  Correlation: {unified_results['correlation']:.3f}")
    print(f"  Avg improvement: {unified_results['avg_improvement']:.1f}%")
    print(f"  CG wins: {unified_results['cg_wins']}/{unified_results['total']}")
    
    # Save results
    output = {
        'h1_393_style': h1_393_results,
        'h1_394_style': h1_394_results,
        'unified_style': unified_results,
        'conclusion': 'DISCREPANCY_RESOLVED' if abs(h1_393_results['correlation'] - h1_394_results['correlation']) < 0.3 else 'DISCREPANCY_PERSISTS'
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nConclusion: {output['conclusion']}")
    print("Results saved to results.json")
