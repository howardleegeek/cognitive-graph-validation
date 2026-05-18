"""
H1.407: Test CG with cross-attention only (no GNN) on longer sequences and multi-step tasks.

Hypothesis: Removing GNN from CG will improve performance on longer sequences and multi-step tasks,
since H1.406 showed GNN interferes with cross-attention benefits.

Test configurations:
1. Baseline (separate encoders + late fusion)
2. Full CG (GNN + cross-attention)
3. CG no GNN (cross-attention only) - expected best performer
4. CG no cross-attn (GNN only)

Test conditions:
- seq_len=20 (longer sequences)
- multi-step tasks (n_steps=3)
- seq_len=30 (even longer sequences)
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

class SyntheticDataset(Dataset):
    """Generate synthetic robot manipulation data with configurable complexity."""
    
    def __init__(self, n_demos=500, seq_len=10, n_steps=1, seed=42):
        np.random.seed(seed)
        self.data = []
        
        obs_dim = 8
        lang_dim = 32
        action_dim = 7
        
        for i in range(n_demos):
            # Generate trajectory of given length
            traj_len = seq_len
            observations = []
            actions = []
            
            # Base state
            state = np.random.randn(obs_dim) * 0.5
            
            for t in range(traj_len):
                # Generate action based on state and task complexity
                action = np.random.randn(action_dim) * 0.3
                if n_steps > 1:
                    # Multi-step: actions depend on step phase
                    phase = t / traj_len
                    action *= (1.0 + 0.5 * np.sin(2 * np.pi * n_steps * phase))
                
                # Update state (use first action_dim dims of state)
                state[:action_dim] = state[:action_dim] + action * 0.1 + np.random.randn(action_dim) * 0.05
                state[action_dim:] += np.random.randn(obs_dim - action_dim) * 0.05
                
                observations.append(state.copy())
                actions.append(action.copy())
            
            # Language embedding (task-specific)
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
        # Aggregate trajectory: mean observation, last action
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


class FullCG(nn.Module):
    """Full Cognitive Graph: unified space + GNN + cross-attention."""
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
        # No GNN layers - direct cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


class CGNoCrossAttn(nn.Module):
    """CG without cross-attention: unified space + GNN only."""
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
        # No cross-attention - just mean pooling
        return self.decoder(nodes.mean(dim=1))


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
    """Run H1.407 experiment."""
    print("=" * 60)
    print("H1.407: CG with cross-attention only on longer sequences")
    print("=" * 60)
    
    # Test conditions
    test_conditions = [
        {"name": "seq_len=20", "seq_len": 20, "n_steps": 1},
        {"name": "multi_step (n=3)", "seq_len": 15, "n_steps": 3},
        {"name": "seq_len=30", "seq_len": 30, "n_steps": 1},
    ]
    
    # Model configs
    model_configs = {
        "baseline": lambda: BaselineArchitecture(),
        "full_cg": lambda: FullCG(),
        "cg_no_gnn": lambda: CGNoGNN(),
        "cg_no_cross_attn": lambda: CGNoCrossAttn(),
    }
    
    results = {}
    
    for condition in test_conditions:
        print(f"\n--- {condition['name']} ---")
        
        # Generate data
        train_data = SyntheticDataset(n_demos=400, seq_len=condition['seq_len'], n_steps=condition['n_steps'], seed=42)
        val_data = SyntheticDataset(n_demos=100, seq_len=condition['seq_len'], n_steps=condition['n_steps'], seed=123)
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        condition_results = {}
        baseline_loss = None
        
        for model_name, model_fn in model_configs.items():
            print(f"  Training {model_name}...")
            model = model_fn()
            loss = train_and_eval(model, train_loader, val_loader, epochs=30, lr=1e-4)
            
            if baseline_loss is None:
                baseline_loss = loss
                improvement = "baseline"
            else:
                improvement = f"{(baseline_loss - loss) / baseline_loss * 100:+.2f}%"
            
            condition_results[model_name] = {
                "loss": round(loss, 6),
                "improvement": improvement
            }
            print(f"    Loss: {loss:.6f} ({improvement})")
        
        results[condition['name']] = condition_results
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for condition_name, condition_results in results.items():
        print(f"\n{condition_name}:")
        for model_name, metrics in condition_results.items():
            print(f"  {model_name}: loss={metrics['loss']:.6f} ({metrics['improvement']})")
    
    # Save results
    output = {
        "experiment_id": "H1.407",
        "description": "CG with cross-attention only (no GNN) on longer sequences and multi-step tasks",
        "hypothesis": "Removing GNN from CG will improve performance on longer sequences and multi-step tasks",
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
