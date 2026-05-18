#!/usr/bin/env python3
"""
H1.427 - Task Type Transfer Learning Experiment

Hypothesis: Per-Object CG learns task-specific features that don't transfer well,
while 2-Node CG learns more generalizable representations.

Test: Train on one task type, evaluate on another. Measure transfer gap.

Task Types:
- spatial_relations: Object permanence / spatial reasoning
- multi_stage: Sequential manipulation with multiple stages

Architectures:
- Baseline: Separate encoders + concatenation
- 2-Node CG: Physical + Semantic nodes
- Per-Object CG: One node per object + semantic node

Metrics:
- Same-task performance (baseline)
- Cross-task transfer performance
- Transfer gap = (cross_task_mse - same_task_mse) / same_task_mse
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Use consistent observation dimension across all tasks
OBS_DIM = 32
LANG_DIM = 32
ACTION_DIM = 7

# ============================================================================
# ARCHITECTURES
# ============================================================================

class BaselineArchitecture(nn.Module):
    """Baseline: Separate encoders + concatenation fusion"""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, latent_dim=128):
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


class TwoNodeCognitiveGraph(nn.Module):
    """2-Node CG: Physical node + Semantic node with cross-attention"""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, 
                 physical_dim=144, semantic_dim=368):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        total_dim = physical_dim + semantic_dim
        
        # Encoders
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-attention
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
        z_phys = self.obs_to_physical(obs)
        z_sem = self.lang_to_semantic(lang)
        
        # Create 2 nodes: physical and semantic
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [B, 2, D]
        
        # GNN message passing
        for layer in self.gnn_layers:
            # Mean aggregation from all nodes
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out
        
        return self.decoder(nodes.mean(dim=1))


class PerObjectCognitiveGraph(nn.Module):
    """Per-Object CG: One node per object + semantic node"""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, 
                 n_objects=5, obj_feat_dim=16, semantic_dim=368):
        super().__init__()
        self.n_objects = n_objects
        self.obj_feat_dim = obj_feat_dim
        self.semantic_dim = semantic_dim
        total_dim = obj_feat_dim + semantic_dim
        
        # Object feature extractor (from flattened obs)
        self.obs_to_objects = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_objects * obj_feat_dim),
            nn.LayerNorm(n_objects * obj_feat_dim)
        )
        
        # Language encoder
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-attention
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
        B = obs.size(0)
        
        # Extract object features
        obj_feats = self.obs_to_objects(obs).view(B, self.n_objects, self.obj_feat_dim)
        
        # Get semantic features
        z_sem = self.lang_to_semantic(lang)
        
        # Create nodes: n_objects + 1 semantic node
        obj_nodes = F.pad(obj_feats, (0, self.semantic_dim))  # [B, n_obj, total_dim]
        sem_node = F.pad(z_sem, (self.obj_feat_dim, 0), value=0).unsqueeze(1)  # [B, 1, total_dim]
        nodes = torch.cat([obj_nodes, sem_node], dim=1)  # [B, n_obj+1, total_dim]
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, self.n_objects + 1, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out
        
        return self.decoder(nodes.mean(dim=1))


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_spatial_relation_data(n_demos=500):
    """Generate spatial relation task data (object permanence style)"""
    data = []
    n_objects = 5
    
    for _ in range(n_demos):
        # Object positions (5 objects x 3D = 15 dims)
        objects = np.random.randn(n_objects, 3).astype(np.float32)
        
        # Target object (one-hot)
        target_idx = np.random.randint(n_objects)
        target_onehot = np.zeros(n_objects, dtype=np.float32)
        target_onehot[target_idx] = 1
        
        # Spatial relation query (e.g., "left of", "behind")
        relation_type = np.random.randint(4)
        relation_vec = np.zeros(4, dtype=np.float32)
        relation_vec[relation_type] = 1
        
        # Observation: flattened objects + target + relation + padding
        obs = np.concatenate([
            objects.flatten(),  # 15
            target_onehot,       # 5
            relation_vec,        # 4
            np.zeros(8, dtype=np.float32)  # padding to 32
        ])
        
        # Language embedding (random but consistent)
        lang = np.random.randn(LANG_DIM).astype(np.float32)
        
        # Action: move to target position with relation offset
        offsets = [
            np.array([0.5, 0, 0]),   # right
            np.array([-0.5, 0, 0]),  # left
            np.array([0, 0.5, 0]),   # front
            np.array([0, -0.5, 0]),  # behind
        ]
        target_pos = objects[target_idx] + offsets[relation_type]
        action = np.concatenate([
            target_pos.astype(np.float32), 
            np.zeros(4, dtype=np.float32)
        ])  # pos + gripper
        
        data.append({
            'observation': obs,
            'language': lang,
            'action': action
        })
    
    return data


def generate_multi_stage_data(n_demos=500):
    """Generate multi-stage manipulation task data"""
    data = []
    n_stages = 3
    
    for _ in range(n_demos):
        # Stage information
        current_stage = np.random.randint(n_stages)
        stage_onehot = np.zeros(n_stages, dtype=np.float32)
        stage_onehot[current_stage] = 1
        
        # Object positions for each stage
        stage_positions = np.random.randn(n_stages, 3).astype(np.float32)
        
        # Current object being manipulated
        obj_pos = stage_positions[current_stage]
        
        # Goal position (next stage or final)
        goal_idx = min(current_stage + 1, n_stages - 1)
        goal_pos = stage_positions[goal_idx]
        
        # Observation: current pos + goal pos + stage info + padding
        obs = np.concatenate([
            obj_pos,              # 3
            goal_pos,             # 3
            stage_onehot,         # 3
            np.zeros(23, dtype=np.float32)  # padding to 32
        ])
        
        # Language embedding (random but consistent)
        lang = np.random.randn(LANG_DIM).astype(np.float32)
        
        # Action: move toward goal
        direction = goal_pos - obj_pos
        action = np.concatenate([
            (obj_pos + direction * 0.3).astype(np.float32),  # intermediate target
            direction.astype(np.float32),                     # direction
            np.zeros(1, dtype=np.float32)                      # gripper state
        ])
        
        data.append({
            'observation': obs,
            'language': lang,
            'action': action
        })
    
    return data


def prepare_dataloaders(data, batch_size=32, train_ratio=0.8):
    """Prepare train/val dataloaders"""
    n = len(data)
    n_train = int(n * train_ratio)
    
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    def to_tensor(d):
        obs_arr = np.array([x['observation'] for x in d])
        lang_arr = np.array([x['language'] for x in d])
        action_arr = np.array([x['action'] for x in d])
        
        return {
            'observation': torch.from_numpy(obs_arr),
            'language': torch.from_numpy(lang_arr),
            'action': torch.from_numpy(action_arr)
        }
    
    train_dataset = to_tensor(train_data)
    val_dataset = to_tensor(val_data)
    
    return train_dataset, val_dataset


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_data, val_data, epochs=30, lr=1e-3, batch_size=32):
    """Train model and return validation MSE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    n_train = train_data['observation'].size(0)
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            obs = train_data['observation'][idx]
            lang = train_data['language'][idx]
            action = train_data['action'][idx]
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(val_data['observation'], val_data['language'])
        mse = criterion(pred, val_data['action']).item()
    
    return mse


def run_transfer_experiment(source_task, target_task, architecture_class, arch_name, n_runs=3):
    """Run transfer learning experiment"""
    results = []
    
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # Generate source data
        if source_task == 'spatial_relations':
            source_data = generate_spatial_relation_data(n_demos=500)
        else:
            source_data = generate_multi_stage_data(n_demos=500)
        
        # Generate target data
        if target_task == 'spatial_relations':
            target_data = generate_spatial_relation_data(n_demos=500)
        else:
            target_data = generate_multi_stage_data(n_demos=500)
        
        # Prepare dataloaders
        source_train, source_val = prepare_dataloaders(source_data)
        target_train, target_val = prepare_dataloaders(target_data)
        
        # Train on source
        model = architecture_class()
        source_mse = train_model(model, source_train, source_val)
        
        # Evaluate on target (zero-shot transfer)
        model.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            pred = model(target_val['observation'], target_val['language'])
            transfer_mse = criterion(pred, target_val['action']).item()
        
        # Fine-tune on target (few-shot)
        finetune_mse = train_model(model, target_train, target_val, epochs=10)
        
        results.append({
            'source_mse': source_mse,
            'transfer_mse': transfer_mse,
            'finetune_mse': finetune_mse,
            'transfer_gap': (transfer_mse - source_mse) / source_mse * 100,
            'finetune_gap': (finetune_mse - source_mse) / source_mse * 100
        })
    
    # Aggregate results
    return {
        'source_mse': float(np.mean([r['source_mse'] for r in results])),
        'transfer_mse': float(np.mean([r['transfer_mse'] for r in results])),
        'finetune_mse': float(np.mean([r['finetune_mse'] for r in results])),
        'transfer_gap': float(np.mean([r['transfer_gap'] for r in results])),
        'finetune_gap': float(np.mean([r['finetune_gap'] for r in results])),
        'source_mse_std': float(np.std([r['source_mse'] for r in results])),
        'transfer_mse_std': float(np.std([r['transfer_mse'] for r in results])),
        'n_runs': n_runs
    }


def main():
    print("=" * 80)
    print("H1.427 - Task Type Transfer Learning Experiment")
    print("=" * 80)
    print()
    print("Hypothesis: Per-Object CG learns task-specific features that don't transfer well,")
    print("while 2-Node CG learns more generalizable representations.")
    print()
    
    results = {}
    
    # Define architectures to test
    architectures = [
        ('Baseline', BaselineArchitecture),
        ('2-Node CG', TwoNodeCognitiveGraph),
        ('Per-Object CG', PerObjectCognitiveGraph),
    ]
    
    # Transfer directions
    transfer_directions = [
        ('spatial_relations', 'multi_stage'),
        ('multi_stage', 'spatial_relations')
    ]
    
    for arch_name, arch_class in architectures:
        print(f"\n{'='*60}")
        print(f"Architecture: {arch_name}")
        print(f"{'='*60}")
        
        results[arch_name] = {}
        
        for source_task, target_task in transfer_directions:
            print(f"\nTransfer: {source_task} -> {target_task}")
            print("-" * 40)
            
            result = run_transfer_experiment(
                source_task, target_task, arch_class, arch_name, n_runs=3
            )
            
            results[arch_name][f"{source_task}_to_{target_task}"] = result
            
            print(f"  Source MSE:      {result['source_mse']:.6f} ± {result['source_mse_std']:.6f}")
            print(f"  Transfer MSE:    {result['transfer_mse']:.6f} ± {result['transfer_mse_std']:.6f}")
            print(f"  Finetune MSE:    {result['finetune_mse']:.6f}")
            print(f"  Transfer Gap:    {result['transfer_gap']:+.2f}%")
            print(f"  Finetune Gap:    {result['finetune_gap']:+.2f}%")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Compare transfer gaps
    print("\nTransfer Gap Comparison (lower is better):")
    print("-" * 60)
    
    for direction in ['spatial_relations_to_multi_stage', 'multi_stage_to_spatial_relations']:
        print(f"\n{direction}:")
        for arch_name in ['Baseline', '2-Node CG', 'Per-Object CG']:
            gap = results[arch_name][direction]['transfer_gap']
            print(f"  {arch_name:15s}: {gap:+.2f}%")
    
    # Calculate average transfer gap
    print("\nAverage Transfer Gap:")
    for arch_name in ['Baseline', '2-Node CG', 'Per-Object CG']:
        gaps = [
            results[arch_name]['spatial_relations_to_multi_stage']['transfer_gap'],
            results[arch_name]['multi_stage_to_spatial_relations']['transfer_gap']
        ]
        avg_gap = np.mean(gaps)
        print(f"  {arch_name:15s}: {avg_gap:+.2f}%")
    
    # Save results
    output = {
        'experiment': 'H1.427',
        'hypothesis': 'Per-Object CG learns task-specific features that do not transfer well',
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'conclusion': ''
    }
    
    # Determine conclusion
    per_obj_gap = np.mean([
        results['Per-Object CG']['spatial_relations_to_multi_stage']['transfer_gap'],
        results['Per-Object CG']['multi_stage_to_spatial_relations']['transfer_gap']
    ])
    
    two_node_gap = np.mean([
        results['2-Node CG']['spatial_relations_to_multi_stage']['transfer_gap'],
        results['2-Node CG']['multi_stage_to_spatial_relations']['transfer_gap']
    ])
    
    baseline_gap = np.mean([
        results['Baseline']['spatial_relations_to_multi_stage']['transfer_gap'],
        results['Baseline']['multi_stage_to_spatial_relations']['transfer_gap']
    ])
    
    if per_obj_gap > two_node_gap and per_obj_gap > baseline_gap:
        output['conclusion'] = 'SUPPORTED - Per-Object CG has larger transfer gap, indicating task-specific overfitting'
    elif per_obj_gap < two_node_gap and per_obj_gap < baseline_gap:
        output['conclusion'] = 'REFUTED - Per-Object CG transfers better than other architectures'
    else:
        output['conclusion'] = 'INCONCLUSIVE - Mixed results, no clear pattern'
    
    output['metrics'] = {
        'per_object_avg_transfer_gap': float(per_obj_gap),
        'two_node_avg_transfer_gap': float(two_node_gap),
        'baseline_avg_transfer_gap': float(baseline_gap)
    }
    
    # Save
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'metrics.json'}")
    print(f"\nConclusion: {output['conclusion']}")
    
    return output


if __name__ == '__main__':
    main()