#!/usr/bin/env python3
"""
H1.428 - Hybrid Architecture Experiment
Combines Per-Object CG (for perception) with 2-Node CG (for action prediction)
to get best of both worlds.

Hypothesis: A hybrid architecture that uses Per-Object CG for encoding 
world state and 2-Node CG for action prediction will outperform both 
individual architectures.
"""

import sys
import os
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")
from data_loader import LIBERODataset


class PerObjectCG(nn.Module):
    """Per-Object Cognitive Graph - encodes each object separately"""
    def __init__(self, obs_dim, lang_dim, action_dim, hidden_dim, n_objects=5):
        super().__init__()
        self.n_objects = n_objects
        self.hidden_dim = hidden_dim
        
        # Project each object's observation to hidden dimension
        self.obj_projectors = nn.ModuleList([
            nn.Linear(obs_dim, hidden_dim) for _ in range(n_objects)
        ])
        
        # Language encoder
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        
        # Per-object graph layers (process each object after projection)
        self.obj_graph = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_objects)
        ])
        
        # Fusion
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang):
        # obs: (batch, n_objects, obs_dim)
        # lang: (batch, lang_dim)
        batch_size = obs.shape[0]
        
        # Encode each object
        obj_embeds = []
        for i in range(self.n_objects):
            # Project to hidden dim, then process
            obj_h = self.obj_projectors[i](obs[:, i])
            obj_h = self.obj_graph[i](obj_h)
            obj_embeds.append(obj_h)
        
        obj_embeds = torch.stack(obj_embeds, dim=1)  # (batch, n_objects, hidden)
        
        # Language conditioning
        lang_h = self.lang_encoder(lang).unsqueeze(1).expand(-1, self.n_objects, -1)
        
        # Fuse object + language
        fused = torch.cat([obj_embeds, lang_h], dim=-1)
        fused = self.fusion(fused)
        
        # Aggregate to single representation
        pooled = fused.mean(dim=1)  # (batch, hidden)
        
        # Predict action
        action = self.action_head(pooled)
        
        return action


class TwoNodeCG(nn.Module):
    """2-Node Cognitive Graph - physical + semantic nodes"""
    def __init__(self, obs_dim, lang_dim, action_dim, hidden_dim, n_objects=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_objects = n_objects
        
        # Project aggregated observation to hidden
        self.physical_encoder = nn.Linear(obs_dim * n_objects, hidden_dim)
        
        # Semantic node (language)
        self.semantic_encoder = nn.Linear(lang_dim, hidden_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang):
        # obs: (batch, n_objects, obs_dim)
        # lang: (batch, lang_dim)
        
        # Flatten and encode physical
        obs_flat = obs.view(obs.size(0), -1)  # (batch, n_objects * obs_dim)
        phys_h = self.physical_encoder(obs_flat)
        
        # Encode semantic
        sem_h = self.semantic_encoder(lang)
        
        # Cross-attention
        phys_h = phys_h.unsqueeze(1)  # (batch, 1, hidden)
        sem_h = sem_h.unsqueeze(1)    # (batch, 1, hidden)
        
        attn_out, _ = self.cross_attn(phys_h, sem_h, sem_h)
        attn_out = attn_out.squeeze(1)
        
        # Combine
        combined = torch.cat([phys_h.squeeze(1), sem_h.squeeze(1)], dim=-1)
        output = self.output_proj(combined)
        
        # Predict action
        action = self.action_head(output)
        
        return action


class HybridCG(nn.Module):
    """
    Hybrid Architecture: Per-Object CG for perception + 2-Node CG for action prediction
    
    Design:
    - Per-Object branch: encodes detailed object-level features
    - 2-Node branch: models physical-semantic dynamics
    - Combined: fuses both representations for action prediction
    """
    def __init__(self, obs_dim, lang_dim, action_dim, hidden_dim, n_objects=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Per-Object branch for perception
        self.per_object = PerObjectCG(obs_dim, lang_dim, action_dim, hidden_dim, n_objects)
        
        # 2-Node branch for dynamics
        self.two_node = TwoNodeCG(obs_dim, lang_dim, action_dim, hidden_dim, n_objects)
        
        # Fusion layer - combines both representations
        self.fusion = nn.Sequential(
            nn.Linear(action_dim * 2, action_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim, action_dim)
        )
        
    def forward(self, obs, lang):
        # obs: (batch, n_objects, obs_dim)
        
        # Get both representations (each returns action predictions)
        per_obj_action = self.per_object(obs, lang)
        two_node_action = self.two_node(obs, lang)
        
        # Combine actions
        combined = torch.cat([per_obj_action, two_node_action], dim=-1)
        fused = self.fusion(combined)
        
        return fused


class BaselineModel(nn.Module):
    """Simple MLP baseline"""
    def __init__(self, obs_dim, lang_dim, action_dim, hidden_dim, n_objects=5):
        super().__init__()
        # Flatten observation
        total_obs = obs_dim * n_objects
        
        self.net = nn.Sequential(
            nn.Linear(total_obs + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs, lang):
        # Flatten obs
        obs_flat = obs.view(obs.size(0), -1)
        combined = torch.cat([obs_flat, lang], dim=-1)
        return self.net(combined)


def generate_task_data(n_demos, n_objects=5, task_type="spatial_relations", seed=42):
    """Generate synthetic task data with task-specific patterns"""
    np.random.seed(seed)
    data = []
    
    for i in range(n_demos):
        # Generate observations (object positions + features)
        # Different task types have different observation patterns
        if task_type == "spatial_relations":
            # Spatial relations: objects have spatial correlations
            base_pos = np.random.randn(n_objects, 3) * 0.5
            # Add spatial structure
            for j in range(1, n_objects):
                base_pos[j] = base_pos[j-1] + np.random.randn(3) * 0.3
            obs = np.concatenate([base_pos, np.random.randn(n_objects, 92)], axis=-1)
        else:  # multi_stage
            # Multi-stage: sequential patterns in observations
            base = np.random.randn(n_objects, 95)
            # Add sequential/phase information
            phase = (i % 3) / 3.0
            obs = base + np.concatenate([np.zeros((n_objects, 3)) + phase, np.zeros((n_objects, 92))], axis=-1)
        
        obs = obs.astype(np.float32)
        
        # Generate language instruction - different patterns per task
        if task_type == "spatial_relations":
            # Spatial: focus on spatial dimensions
            lang = np.random.randn(32).astype(np.float32)
            lang[:16] *= 2.0  # Emphasize first half
        else:  # multi_stage
            # Multi-stage: different pattern
            lang = np.random.randn(32).astype(np.float32)
            lang[16:] *= 2.0  # Emphasize second half
        
        # Generate actions - task-specific patterns
        if task_type == "spatial_relations":
            # Spatial: actions depend on object positions
            action = np.random.randn(7).astype(np.float32)
            action[:3] += obs[:1, :3].mean(axis=0)  # Position-based
        else:  # multi_stage
            # Multi-stage: sequential actions
            action = np.random.randn(7).astype(np.float32)
            action[3:] += (i % 3) * 0.5  # Phase-dependent
        
        data.append({
            'observations': obs,
            'language': lang,
            'action': action
        })
    
    return data


def train_model(model, train_data, val_data, epochs=30, lr=1e-3, batch_size=32):
    """Train a model and return validation MSE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        # Shuffle data
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_obs = torch.stack([torch.tensor(train_data[j]['observations']) for j in batch_idx])
            batch_lang = torch.stack([torch.tensor(train_data[j]['language']) for j in batch_idx])
            batch_action = torch.stack([torch.tensor(train_data[j]['action']) for j in batch_idx])
            
            optimizer.zero_grad()
            pred = model(batch_obs, batch_lang)
            loss = criterion(pred, batch_action)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for j in range(len(val_data)):
                obs = torch.tensor(val_data[j]['observations']).unsqueeze(0)
                lang = torch.tensor(val_data[j]['language']).unsqueeze(0)
                action = torch.tensor(val_data[j]['action']).unsqueeze(0)
                
                pred = model(obs, lang)
                val_losses.append(criterion(pred, action).item())
        
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_experiment():
    """Run the hybrid architecture experiment"""
    print("=" * 60)
    print("H1.428: Hybrid Architecture Experiment")
    print("=" * 60)
    
    # Config
    n_demos = 500
    n_objects = 5
    obs_dim = 95
    lang_dim = 32
    action_dim = 7
    hidden_dim = 64
    epochs = 30
    n_runs = 3
    
    results = {
        'task': 'hybrid_architecture',
        'config': {
            'n_demos': n_demos,
            'n_objects': n_objects,
            'obs_dim': obs_dim,
            'lang_dim': lang_dim,
            'action_dim': action_dim,
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'n_runs': n_runs
        },
        'architectures': {}
    }
    
    # Test on both task types
    task_types = ['spatial_relations', 'multi_stage']
    
    for task_idx, task_type in enumerate(task_types):
        print(f"\n--- Task: {task_type} ---")
        
        # Generate data with different seeds per task type
        all_data = generate_task_data(n_demos * 2, n_objects, task_type, seed=42 + task_idx * 100)
        train_data = all_data[:n_demos]
        val_data = all_data[n_demos:]
        
        arch_results = {}
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...")
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            random.seed(42 + run)
            
            # Baseline
            baseline = BaselineModel(obs_dim, lang_dim, action_dim, hidden_dim, n_objects)
            baseline_loss = train_model(baseline, train_data, val_data, epochs)
            
            # Per-Object CG
            per_object = PerObjectCG(obs_dim, lang_dim, action_dim, hidden_dim, n_objects)
            per_object_loss = train_model(per_object, train_data, val_data, epochs)
            
            # 2-Node CG
            two_node = TwoNodeCG(obs_dim, lang_dim, action_dim, hidden_dim, n_objects)
            two_node_loss = train_model(two_node, train_data, val_data, epochs)
            
            # Hybrid CG
            hybrid = HybridCG(obs_dim, lang_dim, action_dim, hidden_dim, n_objects)
            hybrid_loss = train_model(hybrid, train_data, val_data, epochs)
            
            if run == 0:
                arch_results = {
                    'baseline': baseline_loss,
                    'per_object': per_object_loss,
                    'two_node': two_node_loss,
                    'hybrid': hybrid_loss
                }
            else:
                arch_results['baseline'] += baseline_loss
                arch_results['per_object'] += per_object_loss
                arch_results['two_node'] += two_node_loss
                arch_results['hybrid'] += hybrid_loss
        
        # Average
        for k in arch_results:
            arch_results[k] /= n_runs
        
        results['architectures'][task_type] = arch_results
        
        print(f"  Results:")
        print(f"    Baseline:    {arch_results['baseline']:.6f}")
        print(f"    Per-Object:  {arch_results['per_object']:.6f}")
        print(f"    2-Node:      {arch_results['two_node']:.6f}")
        print(f"    Hybrid:      {arch_results['hybrid']:.6f}")
    
    # Compute improvements
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for task_type in task_types:
        r = results['architectures'][task_type]
        baseline_mse = r['baseline']
        
        print(f"\n{task_type}:")
        print(f"  Baseline:    {baseline_mse:.6f} (baseline)")
        
        po_improvement = (baseline_mse - r['per_object']) / baseline_mse * 100
        print(f"  Per-Object:  {r['per_object']:.6f} ({po_improvement:+.2f}%)")
        
        tn_improvement = (baseline_mse - r['two_node']) / baseline_mse * 100
        print(f"  2-Node:      {r['two_node']:.6f} ({tn_improvement:+.2f}%)")
        
        hy_improvement = (baseline_mse - r['hybrid']) / baseline_mse * 100
        print(f"  Hybrid:      {r['hybrid']:.6f} ({hy_improvement:+.2f}%)")
    
    # Determine winner
    all_hybrid = []
    all_best_individual = []
    for task_type in task_types:
        r = results['architectures'][task_type]
        all_hybrid.append(r['hybrid'])
        all_best_individual.append(min(r['per_object'], r['two_node']))
    
    avg_hybrid = np.mean(all_hybrid)
    avg_best_individual = np.mean(all_best_individual)
    
    if avg_hybrid < avg_best_individual:
        conclusion = "SUPPORTED"
        insight = f"Hybrid ({avg_hybrid:.6f}) outperforms best individual ({avg_best_individual:.6f})"
    else:
        conclusion = "NOT_SUPPORTED"
        insight = f"Best individual ({avg_best_individual:.6f}) matches or beats hybrid ({avg_hybrid:.6f})"
    
    results['conclusion'] = conclusion
    results['key_insight'] = insight
    
    print(f"\nConclusion: {conclusion}")
    print(f"Insight: {insight}")
    
    # Save results
    output_path = Path("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/194-hybrid-architecture/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
