"""
H1.410: Test CG on multi-object manipulation with varying object counts.

Hypothesis: CG improvement will increase with object count as relational 
structure becomes more important.

Tests object counts: 2, 3, 4, 5 objects
Architectures: baseline, CG (no GNN), CG (with GNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation: Multi-object manipulation with varying counts
# ============================================================

class MultiObjectManipDataset(Dataset):
    """
    Generate multi-object manipulation data with explicit relational structure.
    
    Each object has: position (3), velocity (3), type (one-hot 4), color (one-hot 3)
    = 13 dims per object
    
    Relations between all pairs: distance (1), contact (1), relative_position (3)
    = 5 dims per relation pair
    
    Task: predict next state + action given language instruction
    """
    
    def __init__(self, n_objects, n_samples=500, seq_len=5, obs_dim_per_object=13, rel_dim=5):
        self.n_objects = n_objects
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.obs_dim_per_object = obs_dim_per_object
        self.rel_dim = rel_dim
        
        # Total observation dim: objects + relations + language embedding
        self.n_object_dims = n_objects * obs_dim_per_object
        self.n_relation_pairs = n_objects * (n_objects - 1) // 2
        self.n_relation_dims = self.n_relation_pairs * rel_dim
        self.lang_dim = 32  # language instruction embedding
        
        self.total_obs_dim = self.n_object_dims + self.n_relation_dims + self.lang_dim
        self.action_dim = 7  # xyz + rotation(3) + gripper
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        np.random.seed(42 + self.n_objects)  # Different seed per object count
        data = []
        
        task_templates = [
            "pick up the {color} {object}",
            "place the {object} on the {target}",
            "push the {object} toward the {target}",
            "stack the {object1} on the {object2}",
        ]
        
        colors = ["red", "blue", "green"]
        objects = ["cube", "sphere", "cylinder", "cone"]
        targets = ["pad", "zone", "area", "spot"]
        
        for i in range(self.n_samples):
            # Generate object states
            objects_state = []
            for j in range(self.n_objects):
                pos = np.random.uniform(-1, 1, 3)
                vel = np.random.uniform(-0.5, 0.5, 3)
                obj_type = np.zeros(4)
                obj_type[np.random.randint(0, 4)] = 1
                color = np.zeros(3)
                color[np.random.randint(0, 3)] = 1
                obj_state = np.concatenate([pos, vel, obj_type, color])
                objects_state.append(obj_state)
            
            objects_state = np.array(objects_state)  # (n_objects, 13)
            
            # Generate relations between all pairs
            relations = []
            for j in range(self.n_objects):
                for k in range(j+1, self.n_objects):
                    dist = np.linalg.norm(objects_state[j][:3] - objects_state[k][:3])
                    contact = 1.0 if dist < 0.3 else 0.0
                    rel_pos = objects_state[j][:3] - objects_state[k][:3]
                    relations.append(np.array([dist, contact, *rel_pos]))
            
            relations = np.array(relations) if relations else np.zeros((0, 5))
            relations = relations.flatten()
            
            # Language instruction
            task = np.random.choice(task_templates)
            lang_vec = np.random.randn(self.lang_dim) * 0.5
            
            # Generate sequence of states and actions
            states_seq = []
            actions_seq = []
            
            current_objects = objects_state.copy()
            
            for t in range(self.seq_len):
                # Flatten current state
                obj_flat = current_objects.flatten()
                # Recompute relations
                rels = []
                for j in range(self.n_objects):
                    for k in range(j+1, self.n_objects):
                        dist = np.linalg.norm(current_objects[j][:3] - current_objects[k][:3])
                        contact = 1.0 if dist < 0.3 else 0.0
                        rel_pos = current_objects[j][:3] - current_objects[k][:3]
                        rels.append(np.array([dist, contact, *rel_pos]))
                rels = np.array(rels).flatten() if rels else np.zeros(0)
                
                obs = np.concatenate([obj_flat, rels, lang_vec])
                states_seq.append(obs)
                
                # Generate action: move toward a target object
                target_idx = np.random.randint(0, self.n_objects)
                target_pos = current_objects[target_idx][:3]
                action = np.zeros(7)
                action[:3] = (target_pos - current_objects[0][:3]) * 0.3 + np.random.randn(3) * 0.05
                action[3:6] = np.random.randn(3) * 0.1
                action[6] = 1.0 if np.random.random() > 0.5 else 0.0
                actions_seq.append(action)
                
                # Update object states based on action
                current_objects[0][:3] += action[:3] * 0.1
                current_objects[0][3:6] += action[:3] * 0.05
                
                # Simple physics: objects near each other interact
                for j in range(1, self.n_objects):
                    dist = np.linalg.norm(current_objects[0][:3] - current_objects[j][:3])
                    if dist < 0.3:
                        current_objects[j][:3] += action[:3] * 0.05  # pushed
                        current_objects[j][3:6] += action[:3] * 0.02
            
            data.append({
                'states': np.array(states_seq),
                'actions': np.array(actions_seq),
                'n_objects': self.n_objects,
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Use first (seq_len-1) states to predict next action
        x = torch.FloatTensor(sample['states'][:-1])
        y = torch.FloatTensor(sample['actions'][1:])
        return x, y


# ============================================================
# Models
# ============================================================

class BaselineModel(nn.Module):
    """Standard transformer encoder-decoder baseline."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128, n_layers=2, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.decoder(h)


class CrossAttentionCG(nn.Module):
    """Cognitive Graph with cross-attention (no GNN)."""
    
    def __init__(self, obs_dim, action_dim, n_objects, hidden_dim=128, 
                 obs_dim_per_object=13, rel_dim=5, lang_dim=32, n_layers=2, n_heads=4):
        super().__init__()
        self.n_objects = n_objects
        self.obs_dim_per_object = obs_dim_per_object
        self.rel_dim = rel_dim
        self.lang_dim = lang_dim
        
        # Separate projections for objects, relations, language
        self.object_proj = nn.Linear(obs_dim_per_object, hidden_dim)
        self.relation_proj = nn.Linear(rel_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        
        # Cross-attention between objects and relations
        self.obj_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.rel_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, obs_dim)
        batch, seq_len, _ = x.shape
        
        # Parse observation into components
        n_obj = self.n_objects
        n_rel_pairs = n_obj * (n_obj - 1) // 2
        
        obj_flat_dim = n_obj * self.obs_dim_per_object
        rel_flat_dim = n_rel_pairs * self.rel_dim
        
        objects = x[:, :, :obj_flat_dim].reshape(batch, seq_len, n_obj, self.obs_dim_per_object)
        relations = x[:, :, obj_flat_dim:obj_flat_dim+rel_flat_dim].reshape(
            batch, seq_len, n_rel_pairs, self.rel_dim) if n_rel_pairs > 0 else None
        language = x[:, :, obj_flat_dim+rel_flat_dim:]
        
        # Project
        obj_h = self.object_proj(objects)  # (B, T, N_obj, H)
        
        if relations is not None and n_rel_pairs > 0:
            rel_h = self.relation_proj(relations)  # (B, T, N_rel, H)
            
            # Cross-attention: objects attend to relations
            obj_flat = obj_h.reshape(batch * seq_len, n_obj, -1)
            rel_flat = rel_h.reshape(batch * seq_len, n_rel_pairs, -1)
            
            obj_attended, _ = self.obj_attn(obj_flat, rel_flat, rel_flat)
            obj_attended = obj_attended.reshape(batch, seq_len, n_obj, -1)
            
            # Pool objects
            pooled = obj_attended.mean(dim=2)  # (B, T, H)
        else:
            pooled = obj_h.mean(dim=2)
        
        # Add language
        lang_h = self.lang_proj(language)  # (B, T, H)
        pooled = pooled + lang_h
        
        # Temporal processing
        h = self.temporal_encoder(pooled)
        return self.output_proj(h)


class GNNCG(nn.Module):
    """Cognitive Graph with GNN message passing + cross-attention."""
    
    def __init__(self, obs_dim, action_dim, n_objects, hidden_dim=128,
                 obs_dim_per_object=13, rel_dim=5, lang_dim=32, 
                 n_layers=2, n_heads=4, gnn_layers=2):
        super().__init__()
        self.n_objects = n_objects
        self.obs_dim_per_object = obs_dim_per_object
        self.rel_dim = rel_dim
        self.lang_dim = lang_dim
        self.gnn_layers = gnn_layers
        
        self.object_proj = nn.Linear(obs_dim_per_object, hidden_dim)
        self.relation_proj = nn.Linear(rel_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        
        # GNN message passing layers
        self.gnn_msg = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(gnn_layers)
        ])
        self.gnn_update = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(gnn_layers)
        ])
        
        # Cross-attention
        self.obj_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        n_obj = self.n_objects
        n_rel_pairs = n_obj * (n_obj - 1) // 2
        
        obj_flat_dim = n_obj * self.obs_dim_per_object
        rel_flat_dim = n_rel_pairs * self.rel_dim
        
        objects = x[:, :, :obj_flat_dim].reshape(batch, seq_len, n_obj, self.obs_dim_per_object)
        relations = x[:, :, obj_flat_dim:obj_flat_dim+rel_flat_dim].reshape(
            batch, seq_len, n_rel_pairs, self.rel_dim) if n_rel_pairs > 0 else None
        language = x[:, :, obj_flat_dim+rel_flat_dim:]
        
        obj_h = self.object_proj(objects)
        
        if relations is not None and n_rel_pairs > 0:
            rel_h = self.relation_proj(relations)
            
            # GNN message passing
            for gnn_idx in range(self.gnn_layers):
                # Build adjacency from relations
                # For each object pair, compute message
                msg = torch.zeros_like(obj_h)
                pair_idx = 0
                for i in range(n_obj):
                    for j in range(i+1, n_obj):
                        if pair_idx < n_rel_pairs:
                            r = rel_h[:, :, pair_idx]  # (B, T, H)
                            # Message from j to i and i to j
                            msg_i = self.gnn_msg[gnn_idx](
                                torch.cat([obj_h[:, :, i], r], dim=-1)
                            )
                            msg_j = self.gnn_msg[gnn_idx](
                                torch.cat([obj_h[:, :, j], r], dim=-1)
                            )
                            msg[:, :, i] = msg[:, :, i] + msg_i
                            msg[:, :, j] = msg[:, :, j] + msg_j
                        pair_idx += 1
                
                # Update node states
                obj_h_flat = obj_h.reshape(-1, obj_h.shape[-1])
                msg_flat = msg.reshape(-1, msg.shape[-1])
                obj_h_flat = self.gnn_update[gnn_idx](msg_flat, obj_h_flat)
                obj_h = obj_h_flat.reshape(batch, seq_len, n_obj, -1)
            
            # Cross-attention with relations
            obj_flat = obj_h.reshape(batch * seq_len, n_obj, -1)
            rel_flat = rel_h.reshape(batch * seq_len, n_rel_pairs, -1)
            obj_attended, _ = self.obj_attn(obj_flat, rel_flat, rel_flat)
            obj_attended = obj_attended.reshape(batch, seq_len, n_obj, -1)
            pooled = obj_attended.mean(dim=2)
        else:
            pooled = obj_h.mean(dim=2)
        
        lang_h = self.lang_proj(language)
        pooled = pooled + lang_h
        
        h = self.temporal_encoder(pooled)
        return self.output_proj(h)


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-4, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def run_experiment():
    device = 'cpu'
    results = {}
    
    object_counts = [2, 3, 4, 5]
    n_train = 400
    n_val = 100
    epochs = 30
    lr = 1e-4
    seq_len = 5
    obs_dim_per_object = 13
    rel_dim = 5
    lang_dim = 32
    action_dim = 7
    
    print("=" * 70)
    print("H1.410: Multi-Object Scalability Experiment")
    print("=" * 70)
    
    for n_obj in object_counts:
        print(f"\n--- Testing with {n_obj} objects ---")
        
        # Generate data
        full_dataset = MultiObjectManipDataset(
            n_objects=n_obj, 
            n_samples=n_train + n_val,
            seq_len=seq_len,
            obs_dim_per_object=obs_dim_per_object,
            rel_dim=rel_dim
        )
        
        obs_dim = full_dataset.total_obs_dim
        print(f"  Observation dim: {obs_dim}")
        print(f"  Relation pairs: {full_dataset.n_relation_pairs}")
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train baseline
        print(f"  Training baseline...")
        baseline = BaselineModel(obs_dim, action_dim, hidden_dim=128, n_layers=2, n_heads=4)
        baseline_loss = train_model(baseline, train_loader, val_loader, epochs, lr, device)
        print(f"  Baseline loss: {baseline_loss:.6f}")
        
        # Train CG (no GNN)
        print(f"  Training CG (no GNN)...")
        cg_no_gnn = CrossAttentionCG(
            obs_dim, action_dim, n_obj, hidden_dim=128,
            obs_dim_per_object=obs_dim_per_object, rel_dim=rel_dim,
            lang_dim=lang_dim, n_layers=2, n_heads=4
        )
        cg_no_gnn_loss = train_model(cg_no_gnn, train_loader, val_loader, epochs, lr, device)
        print(f"  CG (no GNN) loss: {cg_no_gnn_loss:.6f}")
        
        # Train CG (with GNN)
        print(f"  Training CG (with GNN)...")
        cg_with_gnn = GNNCG(
            obs_dim, action_dim, n_obj, hidden_dim=128,
            obs_dim_per_object=obs_dim_per_object, rel_dim=rel_dim,
            lang_dim=lang_dim, n_layers=2, n_heads=4, gnn_layers=2
        )
        cg_with_gnn_loss = train_model(cg_with_gnn, train_loader, val_loader, epochs, lr, device)
        print(f"  CG (with GNN) loss: {cg_with_gnn_loss:.6f}")
        
        # Compute improvements
        no_gnn_improvement = (baseline_loss - cg_no_gnn_loss) / baseline_loss * 100
        with_gnn_improvement = (baseline_loss - cg_with_gnn_loss) / baseline_loss * 100
        best_improvement = max(no_gnn_improvement, with_gnn_improvement)
        
        print(f"  CG (no GNN) improvement: {no_gnn_improvement:+.2f}%")
        print(f"  CG (with GNN) improvement: {with_gnn_improvement:+.2f}%")
        
        results[f"n_obj_{n_obj}"] = {
            "n_objects": n_obj,
            "obs_dim": obs_dim,
            "n_relation_pairs": full_dataset.n_relation_pairs,
            "baseline_loss": round(baseline_loss, 6),
            "cg_no_gnn_loss": round(cg_no_gnn_loss, 6),
            "cg_with_gnn_loss": round(cg_with_gnn_loss, 6),
            "cg_no_gnn_improvement": f"{no_gnn_improvement:+.2f}%",
            "cg_with_gnn_improvement": f"{with_gnn_improvement:+.2f}%",
            "best_cg_improvement": f"{best_improvement:+.2f}%",
            "cg_wins_baseline": cg_no_gnn_loss < baseline_loss or cg_with_gnn_loss < baseline_loss,
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_configs = len(object_counts)
    cg_wins = sum(1 for r in results.values() if r["cg_wins_baseline"])
    
    print(f"Object counts tested: {object_counts}")
    print(f"CG wins: {cg_wins}/{total_configs}")
    
    for n_obj in object_counts:
        r = results[f"n_obj_{n_obj}"]
        print(f"  {n_obj} objects: baseline={r['baseline_loss']:.6f}, "
              f"CG(no GNN)={r['cg_no_gnn_loss']:.6f} ({r['cg_no_gnn_improvement']}), "
              f"CG(GNN)={r['cg_with_gnn_loss']:.6f} ({r['cg_with_gnn_improvement']})")
    
    # Save results
    output = {
        "experiment_id": "H1.410",
        "description": "Test CG on multi-object manipulation with varying object counts (2,3,4,5)",
        "hypothesis": "CG improvement will increase with object count as relational structure becomes more important",
        "n_train": n_train,
        "n_val": n_val,
        "epochs": epochs,
        "lr": lr,
        "seq_len": seq_len,
        "action_dim": action_dim,
        "obs_dim_per_object": obs_dim_per_object,
        "rel_dim": rel_dim,
        "lang_dim": lang_dim,
        "total_configs": total_configs,
        "cg_wins": cg_wins,
        "win_rate": f"{cg_wins/total_configs*100:.1f}%",
        "results": results,
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'metrics.json'}")
    return output


if __name__ == "__main__":
    run_experiment()
