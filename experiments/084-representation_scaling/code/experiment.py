#!/usr/bin/env python3
"""
H1.387: Representation Scaling Hypothesis

Hypothesis: The optimal representation size scales with task complexity.
Smaller representations work better on simple tasks because they prevent
overfitting, but larger representations will be needed for more complex tasks.

Prediction: On tasks with more objects/concepts, the optimal representation
size will increase. Specifically:
- 2 objects: 72+184 optimal (smaller is better)
- 4 objects: 144+368 optimal (standard size)
- 6+ objects: 288+736 optimal (larger is better)

Test: Vary number of objects and representation sizes, measure performance.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class SimpleBaseline(nn.Module):
    """Baseline: separate processing of physical and semantic."""
    def __init__(self, physical_dim=72, semantic_dim=184, hidden_dim=256):
        super().__init__()
        self.physical_encoder = nn.Sequential(
            nn.Linear(physical_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim + semantic_dim)
        )
    
    def forward(self, physical, semantic):
        p_enc = self.physical_encoder(physical)
        s_enc = self.semantic_encoder(semantic)
        combined = torch.cat([p_enc, s_enc], dim=-1)
        return self.decoder(combined)


class CognitiveGraphLayer(nn.Module):
    """Single GNN layer with cross-modal attention."""
    def __init__(self, physical_dim, semantic_dim, hidden_dim, n_heads=1):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        # Node projections
        self.phys_proj = nn.Linear(physical_dim, hidden_dim)
        self.sem_proj = nn.Linear(semantic_dim, hidden_dim)
        
        # Cross-modal attention (single head by default based on H1.386)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, physical, semantic):
        # Project to common space
        p_nodes = self.phys_proj(physical)  # [B, hidden]
        s_nodes = self.sem_proj(semantic)   # [B, hidden]
        
        # Stack for attention
        nodes = torch.stack([p_nodes, s_nodes], dim=1)  # [B, 2, hidden]
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Split back
        p_out, s_out = attn_out[:, 0], attn_out[:, 1]
        
        # Combine
        combined = torch.cat([p_out, s_out], dim=-1)
        return self.out_proj(combined)


class CognitiveGraph(nn.Module):
    """Cognitive Graph with configurable representation size."""
    def __init__(self, physical_dim=72, semantic_dim=184, hidden_dim=256, n_gnn_layers=1, n_heads=1):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        # Input projections
        self.phys_input = nn.Linear(physical_dim, hidden_dim)
        self.sem_input = nn.Linear(semantic_dim, hidden_dim)
        
        # GNN layers (single layer by default based on H1.386)
        self.gnn_layers = nn.ModuleList([
            CognitiveGraphLayer(physical_dim, semantic_dim, hidden_dim, n_heads)
            for _ in range(n_gnn_layers)
        ])
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim + semantic_dim)
        )
    
    def forward(self, physical, semantic):
        # Initial projection
        h = self.gnn_layers[0](physical, semantic) if self.gnn_layers else torch.cat([
            self.phys_input(physical), self.sem_input(semantic)
        ], dim=-1)
        
        # Apply additional GNN layers
        for gnn in self.gnn_layers[1:]:
            h = gnn(physical, semantic) + h  # Residual
        
        return self.decoder(h)


def generate_synthetic_data(n_samples, n_objects, seq_length=12, physical_dim=72, semantic_dim=184):
    """Generate synthetic data with varying number of objects."""
    np.random.seed(42 + n_objects)  # Different seed for different object counts
    
    data = []
    for _ in range(n_samples):
        # Physical state: object positions, velocities
        # More objects = more complex physical state
        obj_features = []
        for obj_id in range(n_objects):
            # Position (3D), velocity (3D), rotation (4D quaternion)
            pos = np.random.randn(3) * 0.5
            vel = np.random.randn(3) * 0.1
            rot = np.random.randn(4)
            rot = rot / np.linalg.norm(rot)  # Normalize quaternion
            obj_features.extend([*pos, *vel, *rot])
        
        # Pad or truncate to physical_dim
        physical = np.array(obj_features[:physical_dim])
        if len(physical) < physical_dim:
            physical = np.pad(physical, (0, physical_dim - len(physical)))
        
        # Semantic state: object descriptions, relationships
        # More objects = more complex semantic state
        sem_features = []
        for obj_id in range(n_objects):
            # Object type (one-hot, 10 types)
            obj_type = np.zeros(10)
            obj_type[np.random.randint(10)] = 1
            # Object attributes (color, size, shape)
            attrs = np.random.randn(5)
            sem_features.extend([*obj_type, *attrs])
        
        # Add relationships between objects
        for i in range(min(n_objects, 4)):
            for j in range(min(n_objects, 4)):
                if i != j:
                    rel = np.random.randn(3)  # relationship vector
                    sem_features.extend(rel)
        
        # Pad or truncate to semantic_dim
        semantic = np.array(sem_features[:semantic_dim])
        if len(semantic) < semantic_dim:
            semantic = np.pad(semantic, (0, semantic_dim - len(semantic)))
        
        # Target: next state prediction
        target_physical = physical + np.random.randn(physical_dim) * 0.1
        target_semantic = semantic + np.random.randn(semantic_dim) * 0.05
        
        data.append({
            'physical': physical.astype(np.float32),
            'semantic': semantic.astype(np.float32),
            'target_physical': target_physical.astype(np.float32),
            'target_semantic': target_semantic.astype(np.float32)
        })
    
    return data


def train_and_evaluate(model_class, model_kwargs, train_data, val_data, n_epochs=50, lr=1e-3):
    """Train model and return validation loss."""
    device = torch.device('cpu')
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Prepare tensors
    train_phys = torch.tensor([d['physical'] for d in train_data])
    train_sem = torch.tensor([d['semantic'] for d in train_data])
    train_target = torch.tensor([
        np.concatenate([d['target_physical'], d['target_semantic']])
        for d in train_data
    ])
    
    val_phys = torch.tensor([d['physical'] for d in val_data])
    val_sem = torch.tensor([d['semantic'] for d in val_data])
    val_target = torch.tensor([
        np.concatenate([d['target_physical'], d['target_semantic']])
        for d in val_data
    ])
    
    # Training loop
    model.train()
    batch_size = 32
    n_samples = len(train_data)
    
    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            phys_batch = train_phys[batch_idx]
            sem_batch = train_sem[batch_idx]
            target_batch = train_target[batch_idx]
            
            optimizer.zero_grad()
            pred = model(phys_batch, sem_batch)
            loss = F.mse_loss(pred, target_batch)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        pred = model(val_phys, val_sem)
        val_loss = F.mse_loss(pred, val_target).item()
    
    return val_loss


def run_experiment():
    """Run representation scaling experiment."""
    results = {
        'hypothesis': 'H1.387',
        'description': 'Test if optimal representation size scales with task complexity (number of objects)',
        'prediction': 'Larger representations become optimal as number of objects increases',
        'config': {
            'n_epochs': 50,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'n_train': 300,
            'n_val': 75
        },
        'results': {}
    }
    
    # Test configurations
    object_counts = [2, 4, 6, 8]
    rep_sizes = [
        ('small', 72, 184),
        ('standard', 144, 368),
        ('large', 288, 736)
    ]
    
    for n_objects in object_counts:
        print(f"\n[Objects: {n_objects}]")
        results['results'][n_objects] = {}
        
        # Generate data for this object count
        data = generate_synthetic_data(
            results['config']['n_train'] + results['config']['n_val'],
            n_objects
        )
        train_data = data[:results['config']['n_train']]
        val_data = data[results['config']['n_train']:]
        
        best_loss = float('inf')
        best_size = None
        
        for size_name, phys_dim, sem_dim in rep_sizes:
            # Regenerate data with correct dimensions
            data = generate_synthetic_data(
                results['config']['n_train'] + results['config']['n_val'],
                n_objects,
                physical_dim=phys_dim,
                semantic_dim=sem_dim
            )
            train_data = data[:results['config']['n_train']]
            val_data = data[results['config']['n_train']:]
            
            # Train baseline
            baseline_loss = train_and_evaluate(
                SimpleBaseline,
                {'physical_dim': phys_dim, 'semantic_dim': sem_dim},
                train_data, val_data,
                n_epochs=results['config']['n_epochs'],
                lr=results['config']['learning_rate']
            )
            
            # Train CG (optimal config from H1.386: 1 GNN layer, 1 head)
            cg_loss = train_and_evaluate(
                CognitiveGraph,
                {'physical_dim': phys_dim, 'semantic_dim': sem_dim, 'n_gnn_layers': 1, 'n_heads': 1},
                train_data, val_data,
                n_epochs=results['config']['n_epochs'],
                lr=results['config']['learning_rate']
            )
            
            improvement = (baseline_loss - cg_loss) / baseline_loss * 100
            
            print(f"  {size_name} ({phys_dim}+{sem_dim}): Baseline={baseline_loss:.6f}, CG={cg_loss:.6f}, Improvement={improvement:+.2f}%")
            
            results['results'][n_objects][size_name] = {
                'physical_dim': phys_dim,
                'semantic_dim': sem_dim,
                'baseline_loss': baseline_loss,
                'cg_loss': cg_loss,
                'improvement_percent': improvement
            }
            
            if cg_loss < best_loss:
                best_loss = cg_loss
                best_size = size_name
        
        results['results'][n_objects]['optimal_size'] = best_size
        print(f"  -> Optimal size for {n_objects} objects: {best_size}")
    
    # Analyze results
    print("\n[Analysis]")
    optimal_sizes = [results['results'][n]['optimal_size'] for n in object_counts]
    print(f"Optimal sizes by object count: {dict(zip(object_counts, optimal_sizes))}")
    
    # Check if optimal size increases with object count
    size_order = {'small': 0, 'standard': 1, 'large': 2}
    size_values = [size_order[s] for s in optimal_sizes]
    
    # Calculate trend
    correlation = np.corrcoef(object_counts, size_values)[0, 1]
    results['trend_correlation'] = correlation
    
    if correlation > 0.3:
        conclusion = "SUPPORTED - Optimal representation size increases with task complexity"
    elif correlation < -0.3:
        conclusion = "REFUTED - Optimal representation size decreases with task complexity"
    else:
        conclusion = "INCONCLUSIVE - No clear trend between representation size and task complexity"
    
    results['conclusion'] = conclusion
    print(f"Trend correlation: {correlation:.3f}")
    print(f"Conclusion: {conclusion}")
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'metrics.json'}")