#!/usr/bin/env python3
"""
H1.426: Per-Object CG with Explicit Relational Edges
======================================================
Hypothesis: Adding explicit spatial relational edges between objects 
improves Per-Object CG performance on tasks requiring object relations.

Based on H1.425 showing 2-Node CG outperforms Per-Object CG on multi-stage tasks,
this tests whether explicit relational structure helps Per-Object CG recover.

Test: Per-Object CG with relational edges vs standard Per-Object CG vs 2-Node CG
on tasks requiring understanding of spatial relationships (above, below, beside, etc.)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class SpatialRelationDataset(Dataset):
    """
    Dataset with explicit spatial relationships between objects.
    Tasks require understanding relations: above, below, beside, in_front, behind
    """
    
    def __init__(self, n_demos=1500, seq_len=15, n_objects=5, split='train'):
        self.n_demos = n_demos
        self.seq_len = seq_len
        self.n_objects = n_objects
        self.split = split
        
        if split == 'train':
            np.random.seed(42)
            random.seed(42)
        else:
            np.random.seed(123)
            random.seed(123)
            
        self.data = self._generate_data()
        
    def _generate_data(self):
        """Generate manipulation data with spatial relations."""
        data = []
        
        # Spatial relation types
        relations = ['above', 'below', 'beside', 'in_front', 'behind', 'near', 'far']
        
        for i in range(self.n_demos):
            # Generate object states with explicit relations
            # Each object has: position (x, y, z), color, size
            objects = []
            for o in range(self.n_objects):
                obj = {
                    'position': np.random.randn(3) * 0.5,
                    'color': np.random.randint(0, 6),
                    'size': np.random.uniform(0.3, 0.8),
                }
                objects.append(obj)
            
            # Generate explicit relational edges
            edges = []
            for o1 in range(self.n_objects):
                for o2 in range(o1 + 1, self.n_objects):
                    # Compute spatial relation
                    pos1, pos2 = objects[o1]['position'], objects[o2]['position']
                    diff = pos2 - pos1
                    
                    if diff[2] > 0.3:
                        rel = 'above'
                    elif diff[2] < -0.3:
                        rel = 'below'
                    elif abs(diff[0]) > abs(diff[1]) and abs(diff[0]) > 0.3:
                        rel = 'beside'
                    else:
                        rel = 'near'
                    
                    edges.append((o1, o2, rel))
            
            # Generate trajectory with relation-dependent actions
            traj = []
            for t in range(self.seq_len):
                # Action depends on spatial relations
                # Example: "move object 0 above object 1"
                target_obj = random.randint(0, self.n_objects - 1)
                ref_obj = random.randint(0, self.n_objects - 1)
                while ref_obj == target_obj:
                    ref_obj = random.randint(0, self.n_objects - 1)
                
                # Find relation between them
                rel = 'near'
                for e in edges:
                    if (e[0] == target_obj and e[1] == ref_obj) or (e[0] == ref_obj and e[1] == target_obj):
                        rel = e[2]
                        break
                
                # Action: move toward achieving target relation
                action = np.random.randn(7) * 0.1
                action[0] = objects[target_obj]['position'][0] + random.uniform(-0.2, 0.2)
                action[1] = objects[target_obj]['position'][1] + random.uniform(-0.2, 0.2)
                action[2] = objects[target_obj]['position'][2] + random.uniform(-0.1, 0.1)
                action[3:6] = np.random.randn(3) * 0.1
                action[6] = random.choice([0, 1])
                
                # Language includes relation info
                lang = f"move object {target_obj} {rel} object {ref_obj}"
                lang_enc = self._encode_lang(lang)
                
                # Observation includes all object states + relation info
                obs = self._encode_observation(objects, edges)
                
                traj.append({
                    'observation': obs,
                    'language': lang_enc,
                    'action': action,
                    'target_obj': target_obj,
                    'ref_obj': ref_obj,
                    'relation': rel,
                })
            
            data.append(traj)
        
        return data
    
    def _encode_lang(self, text):
        """Encode language to fixed dimension."""
        # Simple hash-based encoding for relation words
        rel_words = {'above': 0, 'below': 1, 'beside': 2, 'in_front': 3, 'behind': 4, 'near': 5, 'far': 6}
        words = text.split()
        encoding = np.zeros(32)
        for w in words:
            if w in rel_words:
                encoding[rel_words[w]] = 1.0
            # Hash other words
            h = hash(w) % 24
            if h < 24:
                encoding[7 + h] = 1.0
        return encoding
    
    def _encode_observation(self, objects, edges):
        """Encode all objects + relations into observation vector."""
        obs = []
        for obj in objects:
            obs.extend(obj['position'])
            obs.append(obj['color'])
            obs.append(obj['size'])
        
        # Add relation features (one-hot for each pair)
        rel_to_idx = {'above': 0, 'below': 1, 'beside': 2, 'in_front': 3, 'behind': 4, 'near': 5, 'far': 6}
        rel_features = np.zeros(7 * 10)  # Max 10 pairs
        for i, (o1, o2, rel) in enumerate(edges[:10]):
            rel_features[i * 7 + rel_to_idx.get(rel, 6)] = 1.0
        
        obs.extend(rel_features)
        return np.array(obs, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        traj = self.data[idx]
        return {
            'observation': torch.tensor(traj[0]['observation'], dtype=torch.float32),
            'language': torch.tensor(traj[0]['language'], dtype=torch.float32),
            'action': torch.tensor(traj[0]['action'], dtype=torch.float32),
        }


# Calculate dimensions
N_OBJECTS = 5
OBJ_FEAT_PER_OBJ = 5  # 3 pos + 1 color + 1 size
REL_FEATURES = 70  # 7 relations * 10 pairs
OBS_DIM = N_OBJECTS * OBJ_FEAT_PER_OBJ + REL_FEATURES  # 25 + 70 = 95
LANG_DIM = 32
ACTION_DIM = 7
HIDDEN_DIM = 64


class BaselineMLP(nn.Module):
    """Standard MLP baseline."""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.net(torch.cat([obs, lang], dim=-1))


class TwoNodeCG(nn.Module):
    """2-Node Cognitive Graph (physical + semantic)."""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, physical_dim=144, semantic_dim=368, hidden_dim=HIDDEN_DIM):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Encoders to unified space
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
        # Encode to unified space
        z_phys = self.obs_to_physical(obs)
        z_sem = self.lang_to_semantic(lang)
        
        # Pad to same dimension
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        # 2-node graph
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode
        return self.decoder(attn_out.mean(dim=1))


class PerObjectCG(nn.Module):
    """Per-Object Cognitive Graph - one node per object."""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, n_objects=N_OBJECTS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.n_objects = n_objects
        self.obj_feat_dim = OBJ_FEAT_PER_OBJ
        
        # Object encoder
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # GNN for object interactions
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(2)
        ])
        
        # Attention over objects
        self.obj_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + lang_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size = obs.size(0)
        
        # Split observation into per-object features
        obj_features = obs[:, :self.n_objects * self.obj_feat_dim].view(batch_size, self.n_objects, self.obj_feat_dim)
        
        # Encode each object
        obj_embeds = []
        for i in range(self.n_objects):
            obj_embeds.append(self.obj_encoder(obj_features[:, i]))
        obj_embeds = torch.stack(obj_embeds, dim=1)  # (B, n_objects, hidden)
        
        # Encode language
        lang_embed = self.lang_encoder(lang).unsqueeze(1)  # (B, 1, hidden)
        
        # GNN message passing between objects
        nodes = obj_embeds
        for layer in self.gnn_layers:
            # Simple message passing: each node receives mean of others
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, self.n_objects, -1)
            combined = torch.cat([nodes, msgs], dim=-1)
            nodes = nodes + layer(combined)
        
        # Attention
        attn_out, _ = self.obj_attention(nodes, nodes, nodes)
        
        # Aggregate and decode
        pooled = attn_out.mean(dim=1)  # (B, hidden)
        return self.decoder(torch.cat([pooled, lang], dim=-1))


class PerObjectCGWithRelations(nn.Module):
    """Per-Object CG with explicit relational edges."""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, n_objects=N_OBJECTS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.n_objects = n_objects
        self.obj_feat_dim = OBJ_FEAT_PER_OBJ
        self.rel_dim = REL_FEATURES
        
        # Object encoder
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Relation encoder (from relation features in observation)
        self.rel_encoder = nn.Sequential(
            nn.Linear(self.rel_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Relational GNN - edges carry relation information
        self.rel_gnn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # self + neighbor + relation
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(3)
        ])
        
        # Attention with relation-aware queries
        self.obj_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + lang_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size = obs.size(0)
        
        # Split observation into per-object features and relation features
        obj_features = obs[:, :self.n_objects * self.obj_feat_dim].view(batch_size, self.n_objects, self.obj_feat_dim)
        rel_features = obs[:, self.n_objects * self.obj_feat_dim:]  # Last 70 dims are relation features
        
        # Encode each object
        obj_embeds = []
        for i in range(self.n_objects):
            obj_embeds.append(self.obj_encoder(obj_features[:, i]))
        obj_embeds = torch.stack(obj_embeds, dim=1)  # (B, n_objects, hidden)
        
        # Encode relations
        rel_embed = self.rel_encoder(rel_features).unsqueeze(1).expand(-1, self.n_objects, -1)
        
        # Encode language
        lang_embed = self.lang_encoder(lang).unsqueeze(1)  # (B, 1, hidden)
        
        # Relational GNN message passing
        nodes = obj_embeds
        for layer in self.rel_gnn:
            # For each object, gather messages from neighbors with relation info
            new_nodes = []
            for i in range(self.n_objects):
                # Get other objects
                others = torch.cat([nodes[:, :i], nodes[:, i+1:]], dim=1)  # (B, n-1, hidden)
                others_mean = others.mean(dim=1)  # (B, hidden)
                # Combine self + neighbors + relation
                combined = torch.cat([nodes[:, i], others_mean, rel_embed[:, i]], dim=-1)
                new_nodes.append(layer(combined))
            nodes = torch.stack(new_nodes, dim=1)
            nodes = nodes + obj_embeds  # Residual
        
        # Attention
        attn_out, _ = self.obj_attention(nodes, nodes, nodes)
        
        # Aggregate and decode
        pooled = attn_out.mean(dim=1)  # (B, hidden)
        return self.decoder(torch.cat([pooled, lang], dim=-1))


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Train model and return final validation loss."""
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


def main():
    print("=" * 60)
    print("H1.426: Per-Object CG with Explicit Relational Edges")
    print("=" * 60)
    
    # Configuration
    n_demos = 1500
    seq_len = 15
    n_objects = N_OBJECTS
    obs_dim = OBS_DIM
    lang_dim = LANG_DIM
    action_dim = ACTION_DIM
    hidden_dim = HIDDEN_DIM
    epochs = 20
    batch_size = 32
    
    print(f"\nConfig: n_demos={n_demos}, seq_len={seq_len}, n_objects={n_objects}")
    print(f"        obs_dim={obs_dim}, hidden_dim={hidden_dim}, epochs={epochs}")
    
    # Create datasets
    train_dataset = SpatialRelationDataset(n_demos=n_demos, seq_len=seq_len, n_objects=n_objects, split='train')
    val_dataset = SpatialRelationDataset(n_demos=n_demos//5, seq_len=seq_len, n_objects=n_objects, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"\nTrain: {len(train_dataset)} demos, Val: {len(val_dataset)} demos")
    
    # Train and evaluate models
    results = {}
    
    # Baseline MLP
    print("\n--- Training Baseline MLP ---")
    baseline = BaselineMLP(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    baseline_mse = train_model(baseline, train_loader, val_loader, epochs=epochs)
    results['baseline'] = baseline_mse
    print(f"Baseline MSE: {baseline_mse:.6f}")
    
    # 2-Node CG
    print("\n--- Training 2-Node CG ---")
    two_node = TwoNodeCG(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    two_node_mse = train_model(two_node, train_loader, val_loader, epochs=epochs)
    results['two_node'] = two_node_mse
    print(f"2-Node CG MSE: {two_node_mse:.6f} (vs baseline: {(two_node_mse/baseline_mse - 1)*100:+.2f}%)")
    
    # Per-Object CG (standard)
    print("\n--- Training Per-Object CG (standard) ---")
    per_object = PerObjectCG(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_objects=n_objects, hidden_dim=hidden_dim)
    per_object_mse = train_model(per_object, train_loader, val_loader, epochs=epochs)
    results['per_object'] = per_object_mse
    print(f"Per-Object CG MSE: {per_object_mse:.6f} (vs baseline: {(per_object_mse/baseline_mse - 1)*100:+.2f}%)")
    
    # Per-Object CG with Relations
    print("\n--- Training Per-Object CG with Relations ---")
    per_object_rel = PerObjectCGWithRelations(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_objects=n_objects, hidden_dim=hidden_dim)
    per_object_rel_mse = train_model(per_object_rel, train_loader, val_loader, epochs=epochs)
    results['per_object_rel'] = per_object_rel_mse
    print(f"Per-Object+Rel CG MSE: {per_object_rel_mse:.6f} (vs baseline: {(per_object_rel_mse/baseline_mse - 1)*100:+.2f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'MSE':>12} {'vs Baseline':>15}")
    print("-" * 52)
    print(f"{'Baseline MLP':<25} {baseline_mse:>12.6f} {'--':>15}")
    print(f"{'2-Node CG':<25} {two_node_mse:>12.6f} {(two_node_mse/baseline_mse-1)*100:>+14.2f}%")
    print(f"{'Per-Object CG':<25} {per_object_mse:>12.6f} {(per_object_mse/baseline_mse-1)*100:>+14.2f}%")
    print(f"{'Per-Object+Rel CG':<25} {per_object_rel_mse:>12.6f} {(per_object_rel_mse/baseline_mse-1)*100:>+14.2f}%")
    
    # Key comparison
    print("\n--- Key Comparisons ---")
    print(f"Per-Object vs 2-Node: {(per_object_mse/two_node_mse-1)*100:+.2f}%")
    print(f"Per-Object+Rel vs Per-Object: {(per_object_rel_mse/per_object_mse-1)*100:+.2f}%")
    print(f"Per-Object+Rel vs 2-Node: {(per_object_rel_mse/two_node_mse-1)*100:+.2f}%")
    
    # Determine conclusion
    if per_object_rel_mse < per_object_mse:
        rel_improvement = "YES"
        rel_pct = (per_object_mse - per_object_rel_mse) / per_object_mse * 100
    else:
        rel_improvement = "NO"
        rel_pct = (per_object_rel_mse - per_object_mse) / per_object_mse * 100
    
    if per_object_rel_mse < two_node_mse:
        beats_two_node = "YES"
    else:
        beats_two_node = "NO"
    
    print(f"\n--- Conclusion ---")
    print(f"Does adding relations help Per-Object? {rel_improvement} ({rel_pct:.1f}% improvement)")
    print(f"Does Per-Object+Rel beat 2-Node? {beats_two_node}")
    
    # Save results
    output = {
        'experiment_id': 'H1.426',
        'hypothesis': 'Per-Object CG with explicit relational edges improves performance on spatial relation tasks',
        'config': {
            'n_demos': n_demos,
            'seq_len': seq_len,
            'n_objects': n_objects,
            'obs_dim': obs_dim,
            'lang_dim': lang_dim,
            'action_dim': action_dim,
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'batch_size': batch_size,
        },
        'results': {
            'baseline_mse': baseline_mse,
            'two_node_mse': two_node_mse,
            'per_object_mse': per_object_mse,
            'per_object_rel_mse': per_object_rel_mse,
        },
        'comparisons': {
            'per_object_vs_baseline': (per_object_mse/baseline_mse - 1) * 100,
            'per_object_rel_vs_baseline': (per_object_rel_mse/baseline_mse - 1) * 100,
            'two_node_vs_baseline': (two_node_mse/baseline_mse - 1) * 100,
            'per_object_vs_two_node': (per_object_mse/two_node_mse - 1) * 100,
            'per_object_rel_vs_per_object': (per_object_rel_mse/per_object_mse - 1) * 100,
            'per_object_rel_vs_two_node': (per_object_rel_mse/two_node_mse - 1) * 100,
        },
        'conclusion': {
            'relations_help_per_object': rel_improvement == "YES",
            'per_object_rel_beats_two_node': beats_two_node == "YES",
        }
    }
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/192-explicit_relations/results/h1.426_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/h1.426_results.json")
    
    return output


if __name__ == '__main__':
    main()
