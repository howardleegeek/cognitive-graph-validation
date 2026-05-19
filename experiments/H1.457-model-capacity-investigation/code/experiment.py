#!/usr/bin/env python3
"""
H1.457: Investigate whether model capacity and data complexity explain H1.453 discrepancy.

Hypothesis: The original H1.453 may have used different model capacity or data complexity
that enabled the massive +82.81% improvement. We test:
1. Model capacity: hidden dims [128, 256, 512, 1024]
2. GNN depth: layers [1, 2, 3, 5, 8]
3. Attention heads: [1, 2, 4, 8]
4. Data complexity: simple vs complex patterns

If CG advantage emerges at higher capacity or complexity, this would explain the discrepancy.
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
results = {}

# ============================================================
# Model Definitions
# ============================================================

class BaselineMLP(nn.Module):
    """Standard MLP baseline with late fusion."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, hidden_dim=256):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang_emb):
        obs_feat = F.relu(self.obs_proj(obs))
        lang_feat = F.relu(self.lang_proj(lang_emb))
        combined = torch.cat([obs_feat, lang_feat], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CognitiveGraphVariableCapacity(nn.Module):
    """Cognitive Graph with variable capacity."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, 
                 hidden_dim=256, n_layers=3, n_heads=4, physical_dim=144, semantic_dim=368):
        super().__init__()
        self.n_layers = n_layers
        total_dim = physical_dim + semantic_dim
        
        # Encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(n_layers)
        ])
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=n_heads, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, obs, lang_emb):
        # Encode to unified space
        z_phys = self.obs_encoder(obs)
        z_sem = self.lang_encoder(lang_emb)
        
        # Create nodes (pad physical to match total_dim)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [B, 2, total_dim]
        
        # GNN message passing
        for layer in self.gnn_layers:
            # Mean aggregation
            msg = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msg)
        
        # Cross attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out
        
        # Decode
        return self.decoder(nodes.mean(dim=1))


# ============================================================
# Data Generation
# ============================================================

def generate_complex_data(n_demos=500, n_steps_per_goal=3, complexity='medium', seed=42):
    """Generate synthetic data with varying complexity."""
    np.random.seed(seed)
    
    # Language embeddings (simulated)
    lang_dim = 384
    lang_embeddings = np.random.randn(n_demos, lang_dim).astype(np.float32)
    
    # Observation dimension
    obs_dim = 8
    
    # Generate observations and actions based on complexity
    if complexity == 'simple':
        # Simple linear relationship
        obs = np.random.randn(n_demos, obs_dim).astype(np.float32)
        action = 0.5 * obs[:, :7] + 0.1 * np.random.randn(n_demos, 7)
    elif complexity == 'medium':
        # Medium: non-linear with language influence
        obs = np.random.randn(n_demos, obs_dim).astype(np.float32)
        lang_influence = np.random.randn(n_demos, 7).astype(np.float32) * 0.3
        action = np.tanh(obs[:, :7] * 0.5) + lang_influence
    else:  # complex
        # Complex: multi-step dependencies, language-grounded
        obs = np.random.randn(n_demos, obs_dim).astype(np.float32)
        # Language affects action in non-linear way
        lang_proj = np.random.randn(lang_dim, 7).astype(np.float32) * 0.1
        lang_effect = np.tanh(lang_embeddings @ lang_proj)
        # Multi-step dependency
        action = np.sin(obs[:, :7] * 2) * 0.5 + lang_effect * 0.5
        action += np.random.randn(n_demos, 7).astype(np.float32) * 0.1
    
    return {
        'observations': obs,
        'actions': action,
        'language_embeddings': lang_embeddings
    }


def prepare_dataloaders(data, batch_size=32, train_ratio=0.8):
    """Prepare train/val dataloaders."""
    n = len(data['observations'])
    n_train = int(n * train_ratio)
    
    # Shuffle
    idx = np.random.permutation(n)
    obs = data['observations'][idx]
    actions = data['actions'][idx]
    lang = data['language_embeddings'][idx]
    
    # Split
    train_dataset = TensorDataset(
        torch.tensor(obs[:n_train]),
        torch.tensor(lang[:n_train]),
        torch.tensor(actions[:n_train])
    )
    val_dataset = TensorDataset(
        torch.tensor(obs[n_train:]),
        torch.tensor(lang[n_train:]),
        torch.tensor(actions[n_train:])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=3e-4, verbose=False):
    """Train model and return best validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for obs, lang, action in train_loader:
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, lang, action in val_loader:
                pred = model(obs, lang)
                val_loss += criterion(pred, action).item()
        
        val_loss /= len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
    
    return best_val_loss


def run_capacity_experiment(hidden_dims, n_layers_list, n_heads_list, complexities):
    """Run experiments varying model capacity and data complexity."""
    results = {
        'capacity_experiments': {},
        'complexity_experiments': {},
        'summary': {}
    }
    
    # Test 1: Hidden dimension sweep
    print("\n=== Testing Hidden Dimensions ===")
    for hidden_dim in hidden_dims:
        key = f"hidden_{hidden_dim}"
        print(f"\nHidden dim: {hidden_dim}")
        
        data = generate_complex_data(n_demos=500, complexity='medium', seed=SEED)
        train_loader, val_loader = prepare_dataloaders(data)
        
        # Baseline
        baseline = BaselineMLP(hidden_dim=hidden_dim)
        baseline_loss = train_model(baseline, train_loader, val_loader)
        
        # Cognitive Graph
        cg = CognitiveGraphVariableCapacity(hidden_dim=hidden_dim)
        cg_loss = train_model(cg, train_loader, val_loader)
        
        improvement = (baseline_loss - cg_loss) / baseline_loss * 100
        
        results['capacity_experiments'][key] = {
            'baseline_loss': baseline_loss,
            'cg_loss': cg_loss,
            'improvement_pct': improvement,
            'cg_wins': cg_loss < baseline_loss
        }
        print(f"  Baseline: {baseline_loss:.6f}, CG: {cg_loss:.6f}, Improvement: {improvement:+.2f}%")
    
    # Test 2: GNN depth sweep
    print("\n=== Testing GNN Depth ===")
    for n_layers in n_layers_list:
        key = f"layers_{n_layers}"
        print(f"\nLayers: {n_layers}")
        
        data = generate_complex_data(n_demos=500, complexity='medium', seed=SEED)
        train_loader, val_loader = prepare_dataloaders(data)
        
        # Baseline (same for all)
        baseline = BaselineMLP(hidden_dim=256)
        baseline_loss = train_model(baseline, train_loader, val_loader)
        
        # Cognitive Graph with variable depth
        cg = CognitiveGraphVariableCapacity(hidden_dim=256, n_layers=n_layers)
        cg_loss = train_model(cg, train_loader, val_loader)
        
        improvement = (baseline_loss - cg_loss) / baseline_loss * 100
        
        results['capacity_experiments'][key] = {
            'baseline_loss': baseline_loss,
            'cg_loss': cg_loss,
            'improvement_pct': improvement,
            'cg_wins': cg_loss < baseline_loss
        }
        print(f"  Baseline: {baseline_loss:.6f}, CG: {cg_loss:.6f}, Improvement: {improvement:+.2f}%")
    
    # Test 3: Attention heads sweep
    print("\n=== Testing Attention Heads ===")
    for n_heads in n_heads_list:
        key = f"heads_{n_heads}"
        print(f"\nHeads: {n_heads}")
        
        data = generate_complex_data(n_demos=500, complexity='medium', seed=SEED)
        train_loader, val_loader = prepare_dataloaders(data)
        
        baseline = BaselineMLP(hidden_dim=256)
        baseline_loss = train_model(baseline, train_loader, val_loader)
        
        cg = CognitiveGraphVariableCapacity(hidden_dim=256, n_heads=n_heads)
        cg_loss = train_model(cg, train_loader, val_loader)
        
        improvement = (baseline_loss - cg_loss) / baseline_loss * 100
        
        results['capacity_experiments'][key] = {
            'baseline_loss': baseline_loss,
            'cg_loss': cg_loss,
            'improvement_pct': improvement,
            'cg_wins': cg_loss < baseline_loss
        }
        print(f"  Baseline: {baseline_loss:.6f}, CG: {cg_loss:.6f}, Improvement: {improvement:+.2f}%")
    
    # Test 4: Data complexity
    print("\n=== Testing Data Complexity ===")
    for complexity in complexities:
        key = f"complexity_{complexity}"
        print(f"\nComplexity: {complexity}")
        
        data = generate_complex_data(n_demos=500, complexity=complexity, seed=SEED)
        train_loader, val_loader = prepare_dataloaders(data)
        
        baseline = BaselineMLP(hidden_dim=256)
        baseline_loss = train_model(baseline, train_loader, val_loader)
        
        cg = CognitiveGraphVariableCapacity(hidden_dim=256)
        cg_loss = train_model(cg, train_loader, val_loader)
        
        improvement = (baseline_loss - cg_loss) / baseline_loss * 100
        
        results['complexity_experiments'][key] = {
            'baseline_loss': baseline_loss,
            'cg_loss': cg_loss,
            'improvement_pct': improvement,
            'cg_wins': cg_loss < baseline_loss
        }
        print(f"  Baseline: {baseline_loss:.6f}, CG: {cg_loss:.6f}, Improvement: {improvement:+.2f}%")
    
    # Summary
    all_improvements = [v['improvement_pct'] for v in results['capacity_experiments'].values()]
    all_improvements += [v['improvement_pct'] for v in results['complexity_experiments'].values()]
    
    results['summary'] = {
        'avg_improvement': np.mean(all_improvements),
        'max_improvement': np.max(all_improvements),
        'min_improvement': np.min(all_improvements),
        'cg_wins_count': sum(1 for v in results['capacity_experiments'].values() if v['cg_wins']) +
                        sum(1 for v in results['complexity_experiments'].values() if v['cg_wins']),
        'total_experiments': len(all_improvements)
    }
    
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("H1.457: Model Capacity and Data Complexity Investigation")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Run experiments
    results = run_capacity_experiment(
        hidden_dims=[128, 256, 512, 1024],
        n_layers_list=[1, 2, 3, 5, 8],
        n_heads_list=[1, 2, 4, 8],
        complexities=['simple', 'medium', 'complex']
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average improvement: {results['summary']['avg_improvement']:+.2f}%")
    print(f"Max improvement: {results['summary']['max_improvement']:+.2f}%")
    print(f"Min improvement: {results['summary']['min_improvement']:+.2f}%")
    print(f"CG wins: {results['summary']['cg_wins_count']}/{results['summary']['total_experiments']}")
    
    # Key finding
    if results['summary']['max_improvement'] > 10:
        conclusion = "Model capacity/data complexity CAN explain H1.453 discrepancy"
    elif results['summary']['max_improvement'] > 5:
        conclusion = "Partial explanation: some configurations show moderate gains"
    else:
        conclusion = "Model capacity/data complexity do NOT explain H1.453 discrepancy"
    
    print(f"\nConclusion: {conclusion}")
    
    # Save results
    results['conclusion'] = conclusion
    results['timestamp'] = datetime.now().isoformat()
    
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Completed: {datetime.now().isoformat()}")