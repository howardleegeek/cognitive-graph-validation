"""
H1.408: Investigate what data properties enable CG benefits.

Hypothesis: CG benefits require data with explicit relational structure (object-entity relationships).
Test three data types:
(a) Unstructured synthetic data (replicates H1.407 failure)
(b) Relational synthetic data with explicit object graphs
(c) Structured data with multi-object interactions

This will identify the minimum data structure needed for CG to outperform baseline.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


class UnstructuredDataset(Dataset):
    """Unstructured synthetic data - same as H1.407."""
    
    def __init__(self, n_demos=500, seq_len=10, n_steps=1, seed=42):
        np.random.seed(seed)
        self.data = []
        
        obs_dim = 8
        lang_dim = 32
        action_dim = 7
        
        for i in range(n_demos):
            traj_len = seq_len
            observations = []
            actions = []
            state = np.random.randn(obs_dim) * 0.5
            
            for t in range(traj_len):
                action = np.random.randn(action_dim) * 0.3
                if n_steps > 1:
                    phase = t / traj_len
                    action *= (1.0 + 0.5 * np.sin(2 * np.pi * n_steps * phase))
                
                state[:action_dim] = state[:action_dim] + action * 0.1 + np.random.randn(action_dim) * 0.05
                state[action_dim:] += np.random.randn(obs_dim - action_dim) * 0.05
                
                observations.append(state.copy())
                actions.append(action.copy())
            
            lang = np.random.randn(lang_dim) * 0.3
            lang[:n_steps] = np.array([1.0 if i < n_steps else 0.0 for i in range(min(n_steps, lang_dim))])
            
            self.data.append({
                'observations': np.array(observations),
                'actions': np.array(actions),
                'language': lang
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        obs = torch.tensor(item['observations'].mean(axis=0), dtype=torch.float32)
        action = torch.tensor(item['actions'][-1], dtype=torch.float32)
        lang = torch.tensor(item['language'], dtype=torch.float32)
        return {'observation': obs, 'action': action, 'language': lang}


class RelationalDataset(Dataset):
    """
    Relational synthetic data with explicit object-entity relationships.
    
    Simulates a multi-object manipulation task where:
    - Multiple objects exist in the scene (positions, velocities, properties)
    - Actions affect specific objects based on language instructions
    - Objects interact with each other (collisions, stacking, etc.)
    - Language specifies target objects and relationships
    """
    
    def __init__(self, n_demos=500, seq_len=10, n_objects=3, seed=42):
        np.random.seed(seed)
        self.data = []
        
        # Each object has: x, y, z position + velocity (6 dims)
        # Plus object properties: type, color, size (3 dims)
        obj_dim = 9
        self.n_objects = n_objects
        obs_dim = n_objects * obj_dim  # Full scene state
        
        lang_dim = 32
        action_dim = 7  # Target object + action type + parameters
        
        for i in range(n_demos):
            traj_len = seq_len
            observations = []
            actions = []
            
            # Initialize objects
            objects = []
            for obj_idx in range(n_objects):
                obj = {
                    'pos': np.random.randn(3) * 2.0,  # Position
                    'vel': np.zeros(3),  # Velocity
                    'props': np.array([obj_idx, np.random.randint(0, 3), np.random.uniform(0.5, 1.5)])  # type, color, size
                }
                objects.append(obj)
            
            # Pick target object for this demo
            target_obj = np.random.randint(0, n_objects)
            
            for t in range(traj_len):
                # Generate action: move toward target object
                target_pos = objects[target_obj]['pos']
                
                # Action: direction to target + action type
                action = np.zeros(action_dim)
                action[:3] = (target_pos - np.mean([o['pos'] for o in objects], axis=0)) * 0.1
                action[3] = target_obj / n_objects  # Target object index
                action[4] = 1.0 if t < traj_len // 2 else 0.0  # Action phase
                action[5:] = np.random.randn(2) * 0.1
                
                # Update objects based on action
                for obj_idx, obj in enumerate(objects):
                    if obj_idx == target_obj:
                        # Target object moves toward action direction
                        obj['vel'] = obj['vel'] * 0.9 + action[:3] * 0.5
                    else:
                        # Other objects affected by proximity (relational dynamics)
                        dist = np.linalg.norm(obj['pos'] - objects[target_obj]['pos'])
                        if dist < 1.0:
                            obj['vel'] += (objects[target_obj]['pos'] - obj['pos']) * 0.05
                    
                    obj['pos'] += obj['vel'] * 0.1
                    obj['vel'] *= 0.95  # Damping
                
                # Build observation: concatenate all object states
                obs = np.concatenate([
                    np.concatenate([obj['pos'], obj['vel'], obj['props']])
                    for obj in objects
                ])
                
                observations.append(obs.copy())
                actions.append(action.copy())
            
            # Language: encodes target object and task type
            lang = np.random.randn(lang_dim) * 0.1
            lang[target_obj] = 1.0  # Target object indicator
            lang[n_objects:n_objects+3] = np.array([1.0, 0.5, 0.0])  # Task type
            
            self.data.append({
                'observations': np.array(observations),
                'actions': np.array(actions),
                'language': lang
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        obs = torch.tensor(item['observations'].mean(axis=0), dtype=torch.float32)
        action = torch.tensor(item['actions'][-1], dtype=torch.float32)
        lang = torch.tensor(item['language'], dtype=torch.float32)
        return {'observation': obs, 'action': action, 'language': lang}


class StructuredMultiObjectDataset(Dataset):
    """
    Structured data with multi-object interactions and explicit graph structure.
    
    Simulates complex manipulation where:
    - Objects have explicit relationships (on_top_of, next_to, inside)
    - Actions must respect physical constraints
    - Language describes relationships between objects
    """
    
    def __init__(self, n_demos=500, seq_len=10, n_objects=4, seed=42):
        np.random.seed(seed)
        self.data = []
        
        obj_dim = 6  # x, y, z, type, size, mass
        self.n_objects = n_objects
        obs_dim = n_objects * obj_dim + n_objects * n_objects  # Object states + adjacency
        
        lang_dim = 32
        action_dim = 7
        
        for i in range(n_demos):
            traj_len = seq_len
            observations = []
            actions = []
            
            # Initialize objects
            objects = []
            for obj_idx in range(n_objects):
                obj = {
                    'pos': np.random.randn(3) * 1.5,
                    'type': obj_idx,
                    'size': np.random.uniform(0.3, 1.0),
                    'mass': np.random.uniform(0.5, 2.0)
                }
                objects.append(obj)
            
            # Build relationship graph
            relationships = np.zeros((n_objects, n_objects))
            for i in range(n_objects):
                for j in range(i+1, n_objects):
                    if np.random.random() < 0.3:
                        relationships[i, j] = 1.0
                        relationships[j, i] = 1.0
            
            target_obj = np.random.randint(0, n_objects)
            
            for t in range(traj_len):
                # Action: move target object, respecting relationships
                action = np.zeros(action_dim)
                action[:3] = np.random.randn(3) * 0.2
                action[3] = target_obj / n_objects
                action[4] = t / traj_len  # Progress
                action[5:] = np.random.randn(2) * 0.1
                
                # Update objects
                for obj_idx, obj in enumerate(objects):
                    if obj_idx == target_obj:
                        obj['pos'] += action[:3] * 0.1
                    else:
                        # Connected objects move together
                        if relationships[target_obj, obj_idx] > 0:
                            obj['pos'] += action[:3] * 0.05
                
                # Build observation
                obj_states = np.concatenate([
                    np.concatenate([obj['pos'], [obj['type'], obj['size'], obj['mass']]])
                    for obj in objects
                ])
                
                obs = np.concatenate([obj_states, relationships.flatten()])
                
                observations.append(obs.copy())
                actions.append(action.copy())
            
            # Language: encodes target and relationship structure
            lang = np.random.randn(lang_dim) * 0.1
            lang[target_obj] = 1.0
            lang[n_objects:n_objects + n_objects] = relationships[target_obj]
            
            self.data.append({
                'observations': np.array(observations),
                'actions': np.array(actions),
                'language': lang
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        obs = torch.tensor(item['observations'].mean(axis=0), dtype=torch.float32)
        action = torch.tensor(item['actions'][-1], dtype=torch.float32)
        lang = torch.tensor(item['language'], dtype=torch.float32)
        return {'observation': obs, 'action': action, 'language': lang}


# Architectures
class BaselineArchitecture(nn.Module):
    """Separate encoders + late fusion."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CGNoGNN(nn.Module):
    """CG without GNN: unified space + cross-attention only."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


class CGWithGNN(nn.Module):
    """CG with GNN: unified space + GNN + cross-attention."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


def train_and_eval(model, train_loader, val_loader, epochs=30, lr=1e-4):
    """Train model and return validation loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_experiment():
    """Run H1.408 experiment."""
    print("=" * 60)
    print("H1.408: What data properties enable CG benefits?")
    print("=" * 60)
    
    # Data types to test
    data_types = [
        {
            "name": "unstructured",
            "description": "Unstructured synthetic data (replicates H1.407)",
            "train": lambda: UnstructuredDataset(n_demos=400, seq_len=15, seed=42),
            "val": lambda: UnstructuredDataset(n_demos=100, seq_len=15, seed=123),
        },
        {
            "name": "relational",
            "description": "Relational data with explicit object-entity relationships",
            "train": lambda: RelationalDataset(n_demos=400, seq_len=15, n_objects=3, seed=42),
            "val": lambda: RelationalDataset(n_demos=100, seq_len=15, n_objects=3, seed=123),
        },
        {
            "name": "structured_multi_object",
            "description": "Structured data with multi-object interactions and graph structure",
            "train": lambda: StructuredMultiObjectDataset(n_demos=400, seq_len=15, n_objects=4, seed=42),
            "val": lambda: StructuredMultiObjectDataset(n_demos=100, seq_len=15, n_objects=4, seed=123),
        },
    ]
    
    # Model configs
    model_configs = {
        "baseline": lambda obs_dim: BaselineArchitecture(obs_dim=obs_dim),
        "cg_no_gnn": lambda obs_dim: CGNoGNN(obs_dim=obs_dim),
        "cg_with_gnn": lambda obs_dim: CGWithGNN(obs_dim=obs_dim),
    }
    
    results = {}
    
    for data_type in data_types:
        print(f"\n--- {data_type['name']}: {data_type['description']} ---")
        
        # Get obs_dim from dataset
        train_data = data_type['train']()
        val_data = data_type['val']()
        obs_dim = train_data[0]['observation'].shape[0]
        print(f"  Observation dimension: {obs_dim}")
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        data_results = {}
        baseline_loss = None
        
        for model_name, model_fn in model_configs.items():
            print(f"  Training {model_name}...")
            model = model_fn(obs_dim)
            loss = train_and_eval(model, train_loader, val_loader, epochs=30, lr=1e-4)
            
            if baseline_loss is None:
                baseline_loss = loss
                improvement = "baseline"
            else:
                improvement = f"{(baseline_loss - loss) / baseline_loss * 100:+.2f}%"
            
            data_results[model_name] = {
                "loss": round(loss, 6),
                "improvement": improvement
            }
            print(f"    Loss: {loss:.6f} ({improvement})")
        
        results[data_type['name']] = {
            "description": data_type['description'],
            "obs_dim": obs_dim,
            "results": data_results
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for data_name, data_results in results.items():
        print(f"\n{data_name} (obs_dim={data_results['obs_dim']}):")
        for model_name, metrics in data_results['results'].items():
            print(f"  {model_name}: loss={metrics['loss']:.6f} ({metrics['improvement']})")
    
    # Save results
    output = {
        "experiment_id": "H1.408",
        "description": "Investigate what data properties enable CG benefits",
        "hypothesis": "CG benefits require data with explicit relational structure",
        "results": results,
        "config": {
            "lr": 1e-4,
            "epochs": 30,
            "n_demos_train": 400,
            "n_demos_val": 100
        }
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'metrics.json'}")
    return output


if __name__ == "__main__":
    run_experiment()
