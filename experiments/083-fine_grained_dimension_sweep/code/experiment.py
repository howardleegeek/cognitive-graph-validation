#!/usr/bin/env python3
"""
H1.470.1.1: Fine-grained dimension sweep around 768

Hypothesis: There exists an optimal representation dimension (~768) for CG on multi-step tasks.
Below this dimension: representation is too constrained.
Above this dimension: overfits single-step tasks, diminishing returns on multi-step.

Prediction: Fine-grained sweep around 768 [640, 704, 768, 832, 896] will show:
1. Peak multi-step improvement around 768
2. Single-step improvement may continue increasing with dimension (overfitting)
3. Improvement gap (single-to-multi) minimized around optimal dimension

Test: Compare CG with dimensions [640, 704, 768, 832, 896] on single-step vs 3-step tasks.
15 epochs, 800 train / 200 test samples per dimension.
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
    "experiment_id": "H1.470.1.1",
    "description": "Fine-grained dimension sweep around 768 [640, 704, 768, 832, 896]",
    "hypothesis": "Optimal representation dimension (~768) exists for CG on multi-step tasks",
    "dimensions_tested": [640, 704, 768, 832, 896],
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
                 total_dim=768, dropout=0.4):
        super().__init__()
        # Fixed ratio: 144/512 = ~28% physical, 368/512 = ~72% semantic
        # Maintain same ratio for all dimensions
        physical_dim = int(total_dim * 0.28)  # ~28% for physical
        semantic_dim = total_dim - physical_dim  # ~72% for semantic
        
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = total_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross attention between physical and semantic
        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        
        # Decoder to action
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        physical = self.obs_to_physical(obs).unsqueeze(1)  # [batch, 1, physical_dim]
        semantic = self.lang_to_semantic(lang).unsqueeze(1)  # [batch, 1, semantic_dim]
        
        # Concatenate physical and semantic
        unified = torch.cat([physical, semantic], dim=-1)  # [batch, 1, total_dim]
        
        # GNN processing (simplified as MLP since we don't have explicit graph)
        for layer in self.gnn_layers:
            unified = layer(unified)
        
        # Cross attention (self-attention on unified representation)
        attn_out, _ = self.cross_attn(unified, unified, unified)
        unified = unified + attn_out  # residual
        
        # Decode to action
        action = self.decoder(unified.squeeze(1))
        return action


# ============================================================
# Data Generation
# ============================================================

def generate_single_step_data(n_samples=1000, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate simple single-step prediction data."""
    # Observations: random normalized values
    obs = torch.randn(n_samples, obs_dim) * 0.5 + 0.5
    # Language: one-hot encoded instructions (simplified)
    lang = torch.zeros(n_samples, lang_dim)
    for i in range(n_samples):
        lang[i, i % lang_dim] = 1.0
    
    # Actions: simple linear combination with noise
    W = torch.randn(obs_dim + lang_dim, action_dim) * 0.1
    combined = torch.cat([obs, lang], dim=-1)
    actions = combined @ W + torch.randn(n_samples, action_dim) * 0.05
    
    return obs, lang, actions

def generate_multi_step_data(n_samples=1000, n_steps=3, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate multi-step prediction data with temporal dependencies."""
    obs_seq = []
    action_seq = []
    
    # Language: one-hot encoded instructions (simplified)
    lang = torch.zeros(n_samples, lang_dim)
    for i in range(n_samples):
        lang[i, i % lang_dim] = 1.0
    
    for step in range(n_steps):
        # Observations: depend on previous actions
        if step == 0:
            obs = torch.randn(n_samples, obs_dim) * 0.5 + 0.5
        else:
            # Next observation depends on previous action (pad if needed)
            prev_action = action_seq[-1]
            # Pad or truncate action to match obs_dim
            if prev_action.shape[1] < obs_dim:
                # Pad with zeros
                padding = torch.zeros(n_samples, obs_dim - prev_action.shape[1])
                prev_action_padded = torch.cat([prev_action, padding], dim=-1)
            else:
                # Truncate
                prev_action_padded = prev_action[:, :obs_dim]
            
            obs = obs_seq[-1] + prev_action_padded * 0.3 + torch.randn(n_samples, obs_dim) * 0.1
        
        # Actions: depend on current obs, lang, and previous actions
        if step == 0:
            context = torch.cat([obs, lang], dim=-1)
        else:
            context = torch.cat([obs, lang, action_seq[-1]], dim=-1)
        
        W = torch.randn(context.shape[-1], action_dim) * 0.1
        actions = context @ W + torch.randn(n_samples, action_dim) * 0.05
        
        obs_seq.append(obs)
        action_seq.append(actions)
    
    # For multi-step prediction, we predict the final action
    return obs_seq[0], lang, action_seq[-1]  # initial obs, lang, final action

# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=15, lr=0.001):
    """Train a model and return final validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for obs, lang, actions in train_loader:
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, lang, actions in val_loader:
                pred = model(obs, lang)
                loss = criterion(pred, actions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_loss

def run_experiment_for_dimension(dimension, n_runs=3):
    """Run experiment for a specific dimension across single and multi-step tasks."""
    print(f"\n=== Testing dimension {dimension} ===")
    
    single_step_losses = []
    multi_step_losses = []
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}")
        
        # Generate data
        obs_train, lang_train, actions_train = generate_single_step_data(n_samples=800)
        obs_val, lang_val, actions_val = generate_single_step_data(n_samples=200)
        
        train_dataset = TensorDataset(obs_train, lang_train, actions_train)
        val_dataset = TensorDataset(obs_val, lang_val, actions_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train baseline
        baseline = BaselineArchitecture(hidden_dim=256)
        baseline_loss = train_model(baseline, train_loader, val_loader, epochs=15)
        
        # Train CG
        cg = CognitiveGraphVariableDim(total_dim=dimension)
        cg_loss = train_model(cg, train_loader, val_loader, epochs=15)
        
        single_step_losses.append({
            "baseline": baseline_loss,
            "cg": cg_loss,
            "improvement": ((baseline_loss - cg_loss) / baseline_loss) * 100
        })
        
        # Multi-step data
        obs_train_m, lang_train_m, actions_train_m = generate_multi_step_data(n_samples=800, n_steps=3)
        obs_val_m, lang_val_m, actions_val_m = generate_multi_step_data(n_samples=200, n_steps=3)
        
        train_dataset_m = TensorDataset(obs_train_m, lang_train_m, actions_train_m)
        val_dataset_m = TensorDataset(obs_val_m, lang_val_m, actions_val_m)
        
        train_loader_m = DataLoader(train_dataset_m, batch_size=32, shuffle=True)
        val_loader_m = DataLoader(val_dataset_m, batch_size=32, shuffle=False)
        
        # Train baseline on multi-step
        baseline_m = BaselineArchitecture(hidden_dim=256)
        baseline_loss_m = train_model(baseline_m, train_loader_m, val_loader_m, epochs=15)
        
        # Train CG on multi-step
        cg_m = CognitiveGraphVariableDim(total_dim=dimension)
        cg_loss_m = train_model(cg_m, train_loader_m, val_loader_m, epochs=15)
        
        multi_step_losses.append({
            "baseline": baseline_loss_m,
            "cg": cg_loss_m,
            "improvement": ((baseline_loss_m - cg_loss_m) / baseline_loss_m) * 100
        })
    
    # Average results
    avg_single = {
        "baseline": np.mean([r["baseline"] for r in single_step_losses]),
        "cg": np.mean([r["cg"] for r in single_step_losses]),
        "improvement": np.mean([r["improvement"] for r in single_step_losses]),
        "std": np.std([r["improvement"] for r in single_step_losses])
    }
    
    avg_multi = {
        "baseline": np.mean([r["baseline"] for r in multi_step_losses]),
        "cg": np.mean([r["cg"] for r in multi_step_losses]),
        "improvement": np.mean([r["improvement"] for r in multi_step_losses]),
        "std": np.std([r["improvement"] for r in multi_step_losses])
    }
    
    improvement_gap = avg_single["improvement"] - avg_multi["improvement"]
    
    return {
        "dimension": dimension,
        "single_step": avg_single,
        "multi_step": avg_multi,
        "improvement_gap": improvement_gap,
        "single_to_multi_baseline_change": ((avg_multi["baseline"] - avg_single["baseline"]) / avg_single["baseline"]) * 100,
        "single_to_multi_cg_change": ((avg_multi["cg"] - avg_single["cg"]) / avg_single["cg"]) * 100
    }

# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 80)
    print("H1.470.1.1: Fine-grained dimension sweep around 768")
    print("Dimensions: [640, 704, 768, 832, 896]")
    print("=" * 80)
    
    for dim in results["dimensions_tested"]:
        print(f"\n{'='*60}")
        print(f"Testing dimension: {dim}")
        print(f"{'='*60}")
        
        result = run_experiment_for_dimension(dim, n_runs=3)
        results["results"][str(dim)] = result
        
        print(f"  Single-step: CG {result['single_step']['improvement']:.2f}% improvement")
        print(f"  Multi-step:  CG {result['multi_step']['improvement']:.2f}% improvement")
        print(f"  Improvement gap: {result['improvement_gap']:.2f}%")
        print(f"  Baseline s2m change: {result['single_to_multi_baseline_change']:.2f}%")
        print(f"  CG s2m change: {result['single_to_multi_cg_change']:.2f}%")
    
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
    print("\nSummary Table:")
    print("=" * 120)
    print(f"{'Dimension':>10} | {'Single CG %':>12} | {'Multi CG %':>12} | {'Gap':>8} | {'Base s2m %':>12} | {'CG s2m %':>12}")
    print("-" * 120)
    
    for dim in results["dimensions_tested"]:
        r = results["results"][str(dim)]
        print(f"{dim:>10} | {r['single_step']['improvement']:>12.2f} | {r['multi_step']['improvement']:>12.2f} | "
              f"{r['improvement_gap']:>8.2f} | {r['single_to_multi_baseline_change']:>12.2f} | "
              f"{r['single_to_multi_cg_change']:>12.2f}")
    
    # Find best dimension for multi-step
    best_dim = max(results["dimensions_tested"], 
                   key=lambda d: results["results"][str(d)]["multi_step"]["improvement"])
    best_multi_improvement = results["results"][str(best_dim)]["multi_step"]["improvement"]
    best_gap = results["results"][str(best_dim)]["improvement_gap"]
    
    print(f"\nBest dimension for multi-step: {best_dim} ({best_multi_improvement:.2f}% improvement)")
    print(f"Improvement gap at {best_dim}: {best_gap:.2f}%")
    
    # Check hypothesis
    if best_dim == 768:
        print("\n✓ HYPOTHESIS SUPPORTED: 768 is optimal dimension for multi-step tasks")
    else:
        print(f"\n✗ HYPOTHESIS REFUTED: Optimal dimension is {best_dim}, not 768")

if __name__ == "__main__":
    main()