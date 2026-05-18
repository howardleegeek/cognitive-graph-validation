"""
H1.409: Test CG on LIBERO-style data with relational structure
Hypothesis: CG benefits require data with explicit relational structure.
Based on H1.408, we expect +40% improvement on relational tasks.

This experiment:
1. Creates LIBERO-style data with explicit object-entity relationships
2. Tests baseline vs CG variants
3. Measures improvement on relational manipulation tasks
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle

# ============================================================================
# RELATIONAL DATA GENERATION
# ============================================================================

class RelationalLIBERODataset(Dataset):
    """
    LIBERO-style dataset with explicit relational structure.
    
    Key difference from standard synthetic data:
    - Objects have explicit properties (position, velocity, type, color)
    - Relations between objects are encoded (distance, contact, containment)
    - Language instructions reference specific objects and relations
    """
    
    def __init__(self, n_demos=500, split='train', seed=42):
        np.random.seed(seed + hash(split) % 1000)
        self.n_demos = n_demos
        self.split = split
        
        # Object types and properties
        self.object_types = ['cube', 'block', 'plate', 'bowl', 'cup', 'bottle']
        self.colors = ['red', 'blue', 'green', 'yellow', 'white', 'black']
        self.containers = ['basket', 'bin', 'drawer', 'shelf', 'box']
        self.locations = ['left', 'right', 'center', 'front', 'back']
        
        # Task templates with relational structure
        self.tasks = [
            'pick_{color}_{object}',
            'place_{object}_in_{container}',
            'push_{object}_to_{location}',
            'stack_{object1}_on_{object2}',
        ]
        
        self.data = self._generate_relational_data()
        
    def _generate_relational_data(self):
        """Generate data with explicit relational structure."""
        data = []
        
        for i in range(self.n_demos):
            # Sample task type
            task_type = np.random.choice(self.tasks)
            
            # Generate objects with properties
            n_objects = np.random.randint(2, 5)  # 2-4 objects in scene
            objects = []
            
            for obj_id in range(n_objects):
                obj = {
                    'id': obj_id,
                    'type': np.random.choice(self.object_types),
                    'color': np.random.choice(self.colors),
                    'position': np.random.uniform(-1, 1, 3).astype(np.float32),
                    'velocity': np.random.uniform(-0.1, 0.1, 3).astype(np.float32),
                    'size': np.random.uniform(0.05, 0.2),
                    'mass': np.random.uniform(0.1, 1.0),
                }
                objects.append(obj)
            
            # Compute relational features
            relations = self._compute_relations(objects)
            
            # Generate instruction and target object
            instruction, target_obj, target_relation = self._generate_instruction(
                task_type, objects
            )
            
            # Generate trajectory based on task
            trajectory = self._generate_trajectory(
                objects, target_obj, target_relation, task_type
            )
            
            # Encode observation with relational structure
            obs = self._encode_relational_observation(objects, relations, trajectory)
            
            # Encode language
            lang_emb = self._encode_language(instruction)
            
            # Action
            action = trajectory['actions'][0]
            
            data.append({
                'observation': obs,
                'language': lang_emb,
                'action': action,
                'instruction': instruction,
                'n_objects': n_objects,
                'task_type': task_type,
            })
        
        return data
    
    def _compute_relations(self, objects):
        """Compute relational features between objects."""
        n = len(objects)
        relations = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Distance relation
                    dist = np.linalg.norm(
                        objects[i]['position'] - objects[j]['position']
                    )
                    # Contact relation (within threshold)
                    contact = 1.0 if dist < 0.15 else 0.0
                    # Relative position
                    rel_pos = objects[i]['position'] - objects[j]['position']
                    
                    relations.append({
                        'from': i,
                        'to': j,
                        'distance': dist,
                        'contact': contact,
                        'relative_position': rel_pos,
                    })
        
        return relations
    
    def _generate_instruction(self, task_type, objects):
        """Generate language instruction referencing objects."""
        if 'pick' in task_type:
            color = np.random.choice(self.colors)
            obj_type = np.random.choice(self.object_types)
            instruction = f"pick up the {color} {obj_type}"
            target_obj = next((o for o in objects if o['color'] == color), objects[0])
            target_relation = None
            
        elif 'place' in task_type:
            obj_type = np.random.choice(self.object_types)
            container = np.random.choice(self.containers)
            instruction = f"place the {obj_type} in the {container}"
            target_obj = next((o for o in objects if o['type'] == obj_type), objects[0])
            target_relation = container
            
        elif 'push' in task_type:
            obj_type = np.random.choice(self.object_types)
            location = np.random.choice(self.locations)
            instruction = f"push the {obj_type} to the {location}"
            target_obj = next((o for o in objects if o['type'] == obj_type), objects[0])
            target_relation = location
            
        else:  # stack
            obj_type1 = np.random.choice(self.object_types)
            obj_type2 = np.random.choice(self.object_types)
            instruction = f"stack the {obj_type1} on the {obj_type2}"
            target_obj = next((o for o in objects if o['type'] == obj_type1), objects[0])
            target_relation = obj_type2
        
        return instruction, target_obj, target_relation
    
    def _generate_trajectory(self, objects, target_obj, target_relation, task_type):
        """Generate action trajectory for the task."""
        # Start position
        start_pos = target_obj['position'].copy()
        
        # Target position based on task
        if 'pick' in task_type:
            # Move up
            target_pos = start_pos + np.array([0, 0, 0.3])
        elif 'place' in task_type:
            # Move to container location
            target_pos = np.random.uniform(-0.5, 0.5, 3)
            target_pos[2] = 0.1
        elif 'push' in task_type:
            # Push in direction
            target_pos = start_pos + np.random.uniform(-0.3, 0.3, 3)
            target_pos[2] = start_pos[2]
        else:  # stack
            # Stack on another object
            target_pos = start_pos + np.array([0, 0, 0.2])
        
        # Generate action (delta to target)
        action = (target_pos - start_pos) * 0.1  # Scale down
        action = np.clip(action, -1, 1).astype(np.float32)
        
        # Pad to 7 dims (xyz + rotation + gripper)
        full_action = np.zeros(7, dtype=np.float32)
        full_action[:3] = action
        full_action[3:6] = np.random.uniform(-0.1, 0.1, 3)  # Small rotation
        full_action[6] = 1.0 if 'pick' in task_type else 0.0  # Gripper
        
        return {'actions': [full_action], 'target_pos': target_pos}
    
    def _encode_relational_observation(self, objects, relations, trajectory):
        """
        Encode observation with explicit relational structure.
        
        Structure (27 dims):
        - Object 1: position (3) + velocity (3) + type_onehot (6) + color_onehot (6) = 18
        - Object 2: position (3) + velocity (3) + type_onehot (6) = 12 (partial)
        
        Actually let's use a cleaner encoding:
        - Target object: pos (3) + vel (3) + type (1) + color (1) = 8
        - Relation 1: distance (1) + contact (1) + rel_pos (3) = 5
        - Relation 2: distance (1) + contact (1) + rel_pos (3) = 5
        - Scene context: n_objects (1) + task_encoding (4) + padding (4) = 9
        Total: 27 dims (matches H1.408 relational data)
        """
        obs = np.zeros(27, dtype=np.float32)
        
        # Target object (first object for simplicity)
        if objects:
            obj = objects[0]
            obs[0:3] = obj['position']
            obs[3:6] = obj['velocity']
            obs[6] = self.object_types.index(obj['type']) / len(self.object_types)
            obs[7] = self.colors.index(obj['color']) / len(self.colors)
        
        # Relations
        if relations:
            # First relation
            r1 = relations[0]
            obs[8] = r1['distance'] / 2.0  # Normalize
            obs[9] = r1['contact']
            obs[10:13] = r1['relative_position'] / 2.0
            
            # Second relation (if exists)
            if len(relations) > 1:
                r2 = relations[1]
                obs[13] = r2['distance'] / 2.0
                obs[14] = r2['contact']
                obs[15:18] = r2['relative_position'] / 2.0
        
        # Scene context
        obs[18] = len(objects) / 5.0  # Normalize by max objects
        obs[19:23] = np.zeros(4)  # Task encoding placeholder
        obs[23:27] = np.zeros(4)  # Padding
        
        return obs
    
    def _encode_language(self, instruction):
        """Encode language instruction as embedding."""
        # Simple hash-based encoding (in production, use BERT)
        np.random.seed(hash(instruction) % (2**31))
        emb = np.random.randn(32).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # Normalize
        return emb
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'observation': torch.tensor(item['observation'], dtype=torch.float32),
            'language': torch.tensor(item['language'], dtype=torch.float32),
            'action': torch.tensor(item['action'], dtype=torch.float32),
            'instruction': item['instruction'],
            'n_objects': item['n_objects'],
        }


# ============================================================================
# ARCHITECTURES
# ============================================================================

class BaselineArchitecture(nn.Module):
    """Standard separated encoding baseline."""
    
    def __init__(self, obs_dim=27, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        
        # Separate encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
        # Late fusion
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        fused = torch.cat([z_obs, z_lang], dim=-1)
        return self.fusion(fused)


class CognitiveGraphNoGNN(nn.Module):
    """Cognitive Graph with cross-attention only (no GNN)."""
    
    def __init__(self, obs_dim=27, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368):
        super().__init__()
        
        total_dim = physical_dim + semantic_dim  # 512
        
        # Project to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim),
        )
        
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )
        
        # Cross-attention for fusion
        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=8, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, obs, lang):
        # Project to unified space
        z_phys = self.obs_to_unified(obs)  # [B, 144]
        z_sem = self.lang_to_unified(lang)  # [B, 368]
        
        # Create 2-node graph (physical + semantic)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))  # [B, 512]
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)  # [B, 512]
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [B, 2, 512]
        
        # Cross-attention fusion
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode from mean
        return self.decoder(attn_out.mean(dim=1))


class CognitiveGraphWithGNN(nn.Module):
    """Full Cognitive Graph with GNN + cross-attention."""
    
    def __init__(self, obs_dim=27, lang_dim=32, action_dim=7,
                 physical_dim=144, semantic_dim=368):
        super().__init__()
        
        total_dim = physical_dim + semantic_dim  # 512
        
        # Project to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim),
        )
        
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )
        
        # GNN layers for message passing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim),
            )
            for _ in range(3)
        ])
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=8, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, obs, lang):
        # Project to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create 2-node graph
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [B, 2, 512]
        
        # GNN message passing
        for layer in self.gnn_layers:
            # Mean aggregation from neighbors
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


# ============================================================================
# TRAINING
# ============================================================================

def train_and_eval(model, train_loader, val_loader, epochs=30, lr=1e-4):
    """Train and evaluate model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    # Evaluation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def main():
    print("=" * 70)
    print("H1.409: Cognitive Graph on Relational LIBERO Data")
    print("=" * 70)
    print()
    
    # Generate relational LIBERO data
    print("[Data] Generating relational LIBERO-style data...")
    train_data = RelationalLIBERODataset(n_demos=400, split='train')
    val_data = RelationalLIBERODataset(n_demos=100, split='val')
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    print(f"[Data] Train: {len(train_data)} demos")
    print(f"[Data] Val: {len(val_data)} demos")
    print(f"[Data] Observation dim: {train_data[0]['observation'].shape[0]}")
    print()
    
    results = {}
    
    # Test baseline
    print("[Model] Training Baseline...")
    torch.manual_seed(42)
    baseline = BaselineArchitecture(obs_dim=27, lang_dim=32, action_dim=7)
    baseline_loss = train_and_eval(baseline, train_loader, val_loader)
    print(f"  Baseline loss: {baseline_loss:.6f}")
    results['baseline_loss'] = baseline_loss
    
    # Test CG without GNN
    print("[Model] Training Cognitive Graph (no GNN)...")
    torch.manual_seed(42)
    cg_no_gnn = CognitiveGraphNoGNN(obs_dim=27, lang_dim=32, action_dim=7)
    cg_no_gnn_loss = train_and_eval(cg_no_gnn, train_loader, val_loader)
    print(f"  CG (no GNN) loss: {cg_no_gnn_loss:.6f}")
    results['cg_no_gnn_loss'] = cg_no_gnn_loss
    
    # Test CG with GNN
    print("[Model] Training Cognitive Graph (with GNN)...")
    torch.manual_seed(42)
    cg_with_gnn = CognitiveGraphWithGNN(obs_dim=27, lang_dim=32, action_dim=7)
    cg_with_gnn_loss = train_and_eval(cg_with_gnn, train_loader, val_loader)
    print(f"  CG (with GNN) loss: {cg_with_gnn_loss:.6f}")
    results['cg_with_gnn_loss'] = cg_with_gnn_loss
    
    # Compute improvements
    baseline_loss = results['baseline_loss']
    cg_no_gnn_improvement = (baseline_loss - results['cg_no_gnn_loss']) / baseline_loss * 100
    cg_with_gnn_improvement = (baseline_loss - results['cg_with_gnn_loss']) / baseline_loss * 100
    
    results['cg_no_gnn_improvement'] = cg_no_gnn_improvement
    results['cg_with_gnn_improvement'] = cg_with_gnn_improvement
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline loss:        {baseline_loss:.6f}")
    print(f"CG (no GNN) loss:     {results['cg_no_gnn_loss']:.6f} ({cg_no_gnn_improvement:+.2f}%)")
    print(f"CG (with GNN) loss:   {results['cg_with_gnn_loss']:.6f} ({cg_with_gnn_improvement:+.2f}%)")
    print()
    
    # Determine conclusion
    if cg_no_gnn_improvement > 20 or cg_with_gnn_improvement > 20:
        conclusion = "SUPPORTED"
        print(f"CONCLUSION: {conclusion} - CG shows significant improvement on relational LIBERO data")
    elif cg_no_gnn_improvement > 0 or cg_with_gnn_improvement > 0:
        conclusion = "INCONCLUSIVE"
        print(f"CONCLUSION: {conclusion} - CG shows modest improvement")
    else:
        conclusion = "REFUTED"
        print(f"CONCLUSION: {conclusion} - CG underperforms on relational LIBERO data")
    
    results['conclusion'] = conclusion
    
    # Save results
    output = {
        'experiment': 'H1.409',
        'description': 'Test CG on relational LIBERO-style data',
        'hypothesis': 'CG benefits require data with explicit relational structure',
        'results': results,
        'config': {
            'n_train': 400,
            'n_val': 100,
            'epochs': 30,
            'lr': 1e-4,
            'obs_dim': 27,
            'data_type': 'relational_libero',
        }
    }
    
    with open('results/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/metrics.json")
    
    return results


if __name__ == '__main__':
    main()