#!/usr/bin/env python3
"""
H1.388: Investigate discrepancy between H1.386 (+25% CG win) and H1.387 (CG loses)

Key differences:
- H1.386: n_train=400, n_epochs=60, single object count
- H1.387: n_train=300, n_epochs=50, multiple object counts (2-8)

Hypothesis: The discrepancy is due to:
1. Training set size (300 vs 400)
2. Number of epochs (50 vs 60)
3. Object count complexity effect

This experiment systematically tests these factors.
"""

import sys
import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")
from data_loader import LIBERODataset


class CognitiveGraphModel(nn.Module):
    """Cognitive Graph with unified representation."""
    
    def __init__(self, phys_dim=144, sem_dim=368, n_heads=4, n_layers=1):
        super().__init__()
        self.phys_dim = phys_dim
        self.sem_dim = sem_dim
        self.total_dim = phys_dim + sem_dim
        
        # Unified encoder
        self.phys_encoder = nn.Linear(7, phys_dim)  # 7 = action dim
        self.sem_encoder = nn.Linear(512, sem_dim)  # Language embedding
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.total_dim, num_heads=n_heads, batch_first=True)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Linear(self.total_dim, self.total_dim) for _ in range(n_layers)
        ])
        
        # Decoder
        self.decoder = nn.Linear(self.total_dim, 7)
        
    def forward(self, actions, lang_emb, return_graph=False):
        # Encode - ensure lang_emb is 2D
        if lang_emb.dim() == 1:
            lang_emb = lang_emb.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            
        phys = self.phys_encoder(actions)
        sem = self.sem_encoder(lang_emb)
        x = torch.cat([phys, sem], dim=-1)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x + attn_out.squeeze(0)
        
        # GNN processing
        for layer in self.gnn_layers:
            x = F.relu(layer(x))
        
        # Decode
        pred = self.decoder(x)
        return pred


class BaselineModel(nn.Module):
    """Baseline: separated JEPA + LLM alignment."""
    
    def __init__(self, phys_dim=144, sem_dim=368):
        super().__init__()
        self.phys_encoder = nn.Linear(7, phys_dim)
        self.sem_encoder = nn.Linear(512, sem_dim)
        self.decoder = nn.Linear(phys_dim + sem_dim, 7)
        
    def forward(self, actions, lang_emb):
        # Ensure 2D
        if lang_emb.dim() == 1:
            lang_emb = lang_emb.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            
        phys = self.phys_encoder(actions)
        sem = self.sem_encoder(lang_emb)
        x = torch.cat([phys, sem], dim=-1)
        return self.decoder(x)


class HierarchicalModel(nn.Module):
    """Hierarchical: subgoals + low-level control."""
    
    def __init__(self, phys_dim=144, sem_dim=368, n_subgoals=3):
        super().__init__()
        self.n_subgoals = n_subgoals
        self.phys_encoder = nn.Linear(7, phys_dim)
        self.sem_encoder = nn.Linear(512, sem_dim)
        self.subgoal_predictor = nn.Linear(phys_dim + sem_dim, n_subgoals * 7)
        self.low_level = nn.Linear(phys_dim + sem_dim + n_subgoals * 7, 7)
        
    def forward(self, actions, lang_emb):
        # Ensure 2D
        if lang_emb.dim() == 1:
            lang_emb = lang_emb.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            
        phys = self.phys_encoder(actions)
        sem = self.sem_encoder(lang_emb)
        x = torch.cat([phys, sem], dim=-1)
        
        subgoals = self.subgoal_predictor(x).view(-1, self.n_subgoals, 7)
        x_expanded = torch.cat([x.unsqueeze(1).expand(-1, self.n_subgoals, -1), subgoals], dim=-1)
        x_flat = x_expanded.reshape(x.size(0), -1)
        
        return self.low_level(x_flat)


def train_model(model, train_data, val_data, n_epochs=60, lr=1e-3, device='cpu'):
    """Train model and return best validation loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for actions, lang_emb, targets in train_data:
            actions = actions.to(device)
            lang_emb = lang_emb.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            pred = model(actions, lang_emb)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for actions, lang_emb, targets in val_data:
                actions = actions.to(device)
                lang_emb = lang_emb.to(device)
                targets = targets.to(device)
                pred = model(actions, lang_emb)
                val_loss += criterion(pred, targets).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss


def generate_data(n_samples, seq_len, n_objects, seed=42):
    """Generate synthetic manipulation data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data = []
    for _ in range(n_samples):
        # Generate trajectory
        actions = np.random.randn(seq_len, 7).astype(np.float32) * 0.5
        # Add some structure: move toward objects
        for i in range(1, seq_len):
            actions[i, :3] += actions[i-1, :3] * 0.1  # Momentum
        
        # Language embedding (simulated) - 2D [seq_len, 512]
        lang_emb = np.random.randn(seq_len, 512).astype(np.float32) * 0.1
        # Object-related features
        lang_emb[:, :n_objects * 10] += np.random.randn(seq_len, n_objects * 10).astype(np.float32) * 0.5
        
        # Target: next action
        targets = actions[1:].copy()
        actions = actions[:-1]
        lang_emb = lang_emb[:-1]
        
        data.append((actions, lang_emb, targets))
    
    return data


def run_experiment(n_train, n_val, n_epochs, n_objects, seq_len=10, rep_size='standard'):
    """Run single experiment configuration."""
    
    # Set representation dimensions based on rep_size
    if rep_size == 'small':
        phys_dim, sem_dim = 72, 184
    elif rep_size == 'standard':
        phys_dim, sem_dim = 144, 368
    else:  # large
        phys_dim, sem_dim = 288, 736
    
    # Generate data
    train_data = generate_data(n_train, seq_len, n_objects, seed=42)
    val_data = generate_data(n_val, seq_len, n_objects, seed=43)
    
    # Convert to tensors
    train_tensors = [(torch.from_numpy(a), torch.from_numpy(l), torch.from_numpy(t)) for a, l, t in train_data]
    val_tensors = [(torch.from_numpy(a), torch.from_numpy(l), torch.from_numpy(t)) for a, l, t in val_data]
    
    # Train models
    print(f"  Training Baseline (n_train={n_train}, epochs={n_epochs}, objects={n_objects})...")
    baseline = BaselineModel(phys_dim, sem_dim)
    baseline, baseline_loss = train_model(baseline, train_tensors, val_tensors, n_epochs)
    
    print(f"  Training CG (n_train={n_train}, epochs={n_epochs}, objects={n_objects})...")
    cg = CognitiveGraphModel(phys_dim, sem_dim, n_heads=1, n_layers=1)
    cg, cg_loss = train_model(cg, train_tensors, val_tensors, n_epochs)
    
    print(f"  Training Hierarchical (n_train={n_train}, epochs={n_epochs}, objects={n_objects})...")
    hier = HierarchicalModel(phys_dim, sem_dim)
    hier, hier_loss = train_model(hier, train_tensors, val_tensors, n_epochs)
    
    # Calculate improvement
    baseline_mse = baseline_loss
    cg_mse = cg_loss
    hier_mse = hier_loss
    
    cg_improvement = (baseline_mse - cg_mse) / baseline_mse * 100
    hier_improvement = (baseline_mse - hier_mse) / baseline_mse * 100
    
    return {
        'baseline_mse': baseline_mse,
        'cg_mse': cg_mse,
        'hier_mse': hier_mse,
        'cg_improvement': cg_improvement,
        'hier_improvement': hier_improvement,
    }


def main():
    print("=" * 60)
    print("H1.388: Discrepancy Investigation")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Replicate H1.386 conditions (n_train=400, n_epochs=60)
    print("\n[1] H1.386 conditions: n_train=400, n_epochs=60, 2 objects")
    r1 = run_experiment(400, 100, 60, n_objects=2, rep_size='small')
    results['h1.386_conditions'] = r1
    print(f"  Baseline MSE: {r1['baseline_mse']:.6f}")
    print(f"  CG MSE: {r1['cg_mse']:.6f} ({r1['cg_improvement']:+.2f}%)")
    print(f"  Hier MSE: {r1['hier_mse']:.6f} ({r1['hier_improvement']:+.2f}%)")
    
    # Test 2: H1.387 conditions (n_train=300, n_epochs=50)
    print("\n[2] H1.387 conditions: n_train=300, n_epochs=50, 2 objects")
    r2 = run_experiment(300, 75, 50, n_objects=2, rep_size='large')
    results['h1.387_conditions'] = r2
    print(f"  Baseline MSE: {r2['baseline_mse']:.6f}")
    print(f"  CG MSE: {r2['cg_mse']:.6f} ({r2['cg_improvement']:+.2f}%)")
    print(f"  Hier MSE: {r2['hier_mse']:.6f} ({r2['hier_improvement']:+.2f}%)")
    
    # Test 3: H1.386 with large representation
    print("\n[3] H1.386 conditions + large rep: n_train=400, n_epochs=60, large rep")
    r3 = run_experiment(400, 100, 60, n_objects=2, rep_size='large')
    results['h1.386_large_rep'] = r3
    print(f"  Baseline MSE: {r3['baseline_mse']:.6f}")
    print(f"  CG MSE: {r3['cg_mse']:.6f} ({r3['cg_improvement']:+.2f}%)")
    print(f"  Hier MSE: {r3['hier_mse']:.6f} ({r3['hier_improvement']:+.2f}%)")
    
    # Test 4: H1.387 with small representation
    print("\n[4] H1.387 conditions + small rep: n_train=300, n_epochs=50, small rep")
    r4 = run_experiment(300, 75, 50, n_objects=2, rep_size='small')
    results['h1.387_small_rep'] = r4
    print(f"  Baseline MSE: {r4['baseline_mse']:.6f}")
    print(f"  CG MSE: {r4['cg_mse']:.6f} ({r4['cg_improvement']:+.2f}%)")
    print(f"  Hier MSE: {r4['hier_mse']:.6f} ({r4['hier_improvement']:+.2f}%)")
    
    # Test 5: More objects (4) with H1.386 conditions
    print("\n[5] More objects: n_train=400, n_epochs=60, 4 objects")
    r5 = run_experiment(400, 100, 60, n_objects=4, rep_size='small')
    results['4_objects'] = r5
    print(f"  Baseline MSE: {r5['baseline_mse']:.6f}")
    print(f"  CG MSE: {r5['cg_mse']:.6f} ({r5['cg_improvement']:+.2f}%)")
    print(f"  Hier MSE: {r5['hier_mse']:.6f} ({r5['hier_improvement']:+.2f}%)")
    
    # Test 6: More objects (4) with H1.387 conditions
    print("\n[6] More objects: n_train=300, n_epochs=50, 4 objects")
    r6 = run_experiment(300, 75, 50, n_objects=4, rep_size='large')
    results['4_objects_h1.387'] = r6
    print(f"  Baseline MSE: {r6['baseline_mse']:.6f}")
    print(f"  CG MSE: {r6['cg_mse']:.6f} ({r6['cg_improvement']:+.2f}%)")
    print(f"  Hier MSE: {r6['hier_mse']:.6f} ({r6['hier_improvement']:+.2f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: CG Improvement (%)")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: {r['cg_improvement']:+.2f}%")
    
    # Save results
    output_file = "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/158-h1.388-discrepancy-investigation/results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Determine conclusion
    cg_wins = sum(1 for r in results.values() if r['cg_improvement'] > 0)
    print(f"\nCG wins: {cg_wins}/{len(results)} configurations")


if __name__ == "__main__":
    main()
