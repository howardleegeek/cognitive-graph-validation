#!/usr/bin/env python3
"""
H1.470.1.1.1: Even finer dimension sweep around 832

Hypothesis: Optimal representation dimension is ~832 for CG on multi-step tasks.
Based on H1.470.1.1 results: 832 shows best multi-step performance (+41.49% improvement, -1.76% gap).
768 only achieves +31.06% multi-step (3rd best). High variance at 768 suggests instability.

Prediction: Even finer sweep around 832 [800, 816, 832, 848, 864] will show:
1. Peak multi-step improvement around 832
2. Single-step improvement may continue increasing with dimension (overfitting)
3. Improvement gap (single-to-multi) minimized around 832
4. Performance drops significantly away from 832

Test: Compare CG with dimensions [800, 816, 832, 848, 864] on single-step vs 3-step tasks.
15 epochs, 800 train / 200 test samples per dimension, 3 runs per dimension.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Results storage
results = {
    "experiment_id": "H1.470.1.1.1",
    "description": "Even finer dimension sweep around 832 [800, 816, 832, 848, 864]",
    "hypothesis": "Optimal representation dimension is ~832 for CG on multi-step tasks",
    "dimensions_tested": [800, 816, 832, 848, 864],
    "timestamp": datetime.now().isoformat(),
    "results": {}
}

# ============================================================
# Model Definitions
# ============================================================

class BaselineArchitecture(nn.Module):
    """Standard baseline with late fusion."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_feat = self.obs_encoder(obs)
        lang_feat = self.lang_encoder(lang)
        combined = torch.cat([obs_feat, lang_feat], dim=-1)
        return self.fusion(combined)


class CognitiveGraphVariableDim(nn.Module):
    """Cognitive Graph with variable representation dimension."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 total_dim=832, dropout=0.4):
        super().__init__()
        # Fixed ratio: 144/512 = ~28% physical, 368/512 = ~72% semantic
        # Maintain same ratio for all dimensions
        physical_dim = int(total_dim * 0.28)  # ~28% for physical
        semantic_dim = total_dim - physical_dim  # ~72% for semantic
        
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = total_dim
        
        # Encoders to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim), nn.Dropout(dropout)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=total_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)  # [batch, physical_dim]
        z_sem = self.lang_to_unified(lang)  # [batch, semantic_dim]
        
        # Create nodes: physical and semantic as separate nodes
        # Pad to total_dim for uniform processing
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))  # [batch, total_dim]
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)  # [batch, total_dim]
        
        # Stack into graph nodes
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [batch, 2, total_dim]
        
        # GNN message passing
        for layer in self.gnn_layers:
            # Simple mean aggregation
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode from graph representation
        graph_repr = attn_out.mean(dim=1)  # [batch, total_dim]
        return self.decoder(graph_repr)


# ============================================================
# Data Generation
# ============================================================

def generate_single_step_data(num_samples=1000, obs_dim=8, lang_dim=32, action_dim=7, noise_std=0.1):
    """Generate simple single-step task data."""
    # Observations: robot state
    observations = np.random.randn(num_samples, obs_dim) * 0.5
    
    # Language: one-hot encoded instructions
    language = np.zeros((num_samples, lang_dim))
    for i in range(num_samples):
        lang_idx = np.random.randint(0, lang_dim)
        language[i, lang_idx] = 1.0
    
    # Simple actions: linear mapping
    actions = np.zeros((num_samples, action_dim))
    for i in range(num_samples):
        lang_weight = np.where(language[i] == 1)[0][0] / lang_dim
        # Simple linear mapping
        actions[i] = observations[i, :action_dim] * 0.7 + lang_weight * 0.3
        actions[i] = np.tanh(actions[i]) + np.random.randn(action_dim) * noise_std
    
    return observations, language, actions


def generate_multi_step_data(num_samples=1000, obs_dim=8, lang_dim=32, action_dim=7, noise_std=0.15):
    """Generate more complex multi-step task data."""
    # Observations: robot state with more complex structure
    observations = np.random.randn(num_samples, obs_dim) * 0.5
    
    # Language: one-hot encoded instructions
    language = np.zeros((num_samples, lang_dim))
    for i in range(num_samples):
        lang_idx = np.random.randint(0, lang_dim)
        language[i, lang_idx] = 1.0
    
    # Complex actions: nonlinear mapping with interactions
    actions = np.zeros((num_samples, action_dim))
    for i in range(num_samples):
        lang_weight = np.where(language[i] == 1)[0][0] / lang_dim
        
        # More complex nonlinear mapping for multi-step tasks
        # Include interactions between observation dimensions
        obs_scaled = observations[i, :action_dim] * 0.5
        
        # Quadratic terms for complexity
        quadratic = 0.1 * obs_scaled**2
        # Cross terms for interaction
        cross_terms = 0.05 * np.ones(action_dim)
        for j in range(action_dim-1):
            cross_terms[j] *= obs_scaled[j] * obs_scaled[j+1]
        
        actions[i] = obs_scaled * 0.6 + quadratic + cross_terms + lang_weight * 0.4
        actions[i] = np.tanh(actions[i]) + np.random.randn(action_dim) * noise_std
    
    return observations, language, actions


def prepare_datasets(single_step=True):
    """Prepare single-step or multi-step datasets."""
    if single_step:
        obs, lang, act = generate_single_step_data(num_samples=1000)
    else:
        obs, lang, act = generate_multi_step_data(num_samples=1000)
    
    # Split into train/test
    n_train = 800
    train_data = {
        'observation': torch.FloatTensor(obs[:n_train]),
        'language': torch.FloatTensor(lang[:n_train]),
        'action': torch.FloatTensor(act[:n_train])
    }
    test_data = {
        'observation': torch.FloatTensor(obs[n_train:]),
        'language': torch.FloatTensor(lang[n_train:]),
        'action': torch.FloatTensor(act[n_train:])
    }
    
    # Create dataloaders
    train_dataset = TensorDataset(
        train_data['observation'], train_data['language'], train_data['action']
    )
    test_dataset = TensorDataset(
        test_data['observation'], test_data['language'], test_data['action']
    )
    
    return DataLoader(train_dataset, batch_size=32, shuffle=True), \
           DataLoader(test_dataset, batch_size=32, shuffle=False)


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=15):
    """Train a model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for obs, lang, action in train_loader:
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, lang, action in val_loader:
                pred = model(obs, lang)
                loss = criterion(pred, action)
                val_loss += loss.item()
    
    # Final validation loss
    model.eval()
    final_val_loss = 0.0
    with torch.no_grad():
        for obs, lang, action in val_loader:
            pred = model(obs, lang)
            loss = criterion(pred, action)
            final_val_loss += loss.item()
    
    return final_val_loss / len(val_loader)


def run_experiment_for_dimension(dim, num_runs=3):
    """Run experiment for a specific dimension."""
    print(f"\n=== Testing dimension {dim} ===")
    
    single_step_losses = []
    multi_step_losses = []
    
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}")
        
        # Single-step task
        train_loader_single, val_loader_single = prepare_datasets(single_step=True)
        baseline_single = BaselineArchitecture()
        cg_single = CognitiveGraphVariableDim(total_dim=dim)
        
        baseline_loss_single = train_model(baseline_single, train_loader_single, val_loader_single)
        cg_loss_single = train_model(cg_single, train_loader_single, val_loader_single)
        
        # Multi-step task
        train_loader_multi, val_loader_multi = prepare_datasets(single_step=False)
        baseline_multi = BaselineArchitecture()
        cg_multi = CognitiveGraphVariableDim(total_dim=dim)
        
        baseline_loss_multi = train_model(baseline_multi, train_loader_multi, val_loader_multi)
        cg_loss_multi = train_model(cg_multi, train_loader_multi, val_loader_multi)
        
        single_step_losses.append((baseline_loss_single, cg_loss_single))
        multi_step_losses.append((baseline_loss_multi, cg_loss_multi))
    
    # Calculate statistics
    baseline_single_mean = np.mean([x[0] for x in single_step_losses])
    cg_single_mean = np.mean([x[1] for x in single_step_losses])
    baseline_multi_mean = np.mean([x[0] for x in multi_step_losses])
    cg_multi_mean = np.mean([x[1] for x in multi_step_losses])
    
    single_step_improvement = ((baseline_single_mean - cg_single_mean) / baseline_single_mean) * 100
    multi_step_improvement = ((baseline_multi_mean - cg_multi_mean) / baseline_multi_mean) * 100
    improvement_gap = single_step_improvement - multi_step_improvement
    
    baseline_s2m_change = ((baseline_single_mean - baseline_multi_mean) / baseline_single_mean) * 100
    cg_s2m_change = ((cg_single_mean - cg_multi_mean) / cg_single_mean) * 100
    
    return {
        "dimension": dim,
        "single_step": {
            "baseline_loss": float(baseline_single_mean),
            "cg_loss": float(cg_single_mean),
            "improvement_percent": float(single_step_improvement),
            "std": float(np.std([x[1] for x in single_step_losses]))
        },
        "multi_step": {
            "baseline_loss": float(baseline_multi_mean),
            "cg_loss": float(cg_multi_mean),
            "improvement_percent": float(multi_step_improvement),
            "std": float(np.std([x[1] for x in multi_step_losses]))
        },
        "improvement_gap": float(improvement_gap),
        "baseline_s2m_change": float(baseline_s2m_change),
        "cg_s2m_change": float(cg_s2m_change)
    }


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 80)
    print("H1.470.1.1.1: Even finer dimension sweep around 832 [800, 816, 832, 848, 864]")
    print("Hypothesis: Optimal representation dimension is ~832 for CG on multi-step tasks")
    print("=" * 80)
    
    # Test all dimensions
    dimensions = [800, 816, 832, 848, 864]
    
    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"Testing dimension: {dim}")
        print(f"{'='*60}")
        
        result = run_experiment_for_dimension(dim, num_runs=3)
        results["results"][str(dim)] = result
        
        # Print summary
        print(f"\n  Dimension {dim} Results:")
        print(f"  Single-step: CG {result['single_step']['improvement_percent']:.2f}% better")
        print(f"  Multi-step:  CG {result['multi_step']['improvement_percent']:.2f}% better")
        print(f"  Improvement gap: {result['improvement_gap']:.2f}%")
        print(f"  Baseline s2m change: {result['baseline_s2m_change']:.2f}%")
        print(f"  CG s2m change: {result['cg_s2m_change']:.2f}%")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"Results saved to: {output_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dimension':<10} {'Single-step':<15} {'Multi-step':<15} {'Gap':<10} {'Base s2m':<10} {'CG s2m':<10}")
    print(f"{'-'*80}")
    
    for dim in dimensions:
        r = results["results"][str(dim)]
        print(f"{dim:<10} {r['single_step']['improvement_percent']:>6.2f}% ±{r['single_step']['std']:.2f}  "
              f"{r['multi_step']['improvement_percent']:>6.2f}% ±{r['multi_step']['std']:.2f}  "
              f"{r['improvement_gap']:>6.2f}%  "
              f"{r['baseline_s2m_change']:>6.2f}%  "
              f"{r['cg_s2m_change']:>6.2f}%")
    
    # Find best dimension
    best_dim = max(dimensions, key=lambda d: results["results"][str(d)]["multi_step"]["improvement_percent"])
    best_result = results["results"][str(best_dim)]
    
    print(f"\n{'='*80}")
    print(f"BEST DIMENSION: {best_dim}")
    print(f"  Multi-step improvement: {best_result['multi_step']['improvement_percent']:.2f}%")
    print(f"  Single-step improvement: {best_result['single_step']['improvement_percent']:.2f}%")
    print(f"  Improvement gap: {best_result['improvement_gap']:.2f}%")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    main()