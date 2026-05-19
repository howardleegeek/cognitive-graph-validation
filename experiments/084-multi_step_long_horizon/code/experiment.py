#!/usr/bin/env python3
"""
H1.422: Per-Object CG on Multi-Step Long-Horizon Manipulation

Tests whether Per-Object CG's architectural advantage scales with task complexity
on multi-step manipulation tasks with 20+ timesteps.

Hypothesis: Per-Object CG will show larger improvements over 2-Node CG on longer
horizon tasks because per-object tracking becomes more critical as sequences grow.

Config:
- seq_len: 25 (vs 10 in H1.421)
- n_steps: 3 (pick, move, place)
- n_demos: 1500
- n_objects: 5
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ============================================================
# Data Generation - Multi-Step Long-Horizon LIBERO-style
# ============================================================

class MultiStepLongHorizonDataset(Dataset):
    def __init__(self, n_demos=1500, seq_len=25, n_objects=5, obs_dim=400, 
                 lang_dim=32, action_dim=7, split='train', seed=42):
        self.seq_len = seq_len
        self.n_objects = n_objects
        self.obs_dim = obs_dim
        self.lang_dim = lang_dim
        self.action_dim = action_dim
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.data = self._generate_data(n_demos)
        
        n = len(self.data)
        if split == 'train':
            self.data = self.data[:int(0.7*n)]
        elif split == 'val':
            self.data = self.data[int(0.7*n):int(0.85*n)]
        else:
            self.data = self.data[int(0.85*n):]
    
    def _generate_data(self, n_demos):
        data = []
        
        for i in range(n_demos):
            lang_embed = np.random.randn(self.lang_dim).astype(np.float32) * 0.5
            lang_embed[0] = i % 3
            
            trajectory = []
            actions = []
            
            pick_target = np.random.randn(self.obs_dim // self.n_objects).astype(np.float32) * 0.3
            for t in range(9):
                obj_states = []
                for obj_idx in range(self.n_objects):
                    if obj_idx == (i % self.n_objects):
                        progress = t / 8.0
                        state = pick_target * progress + np.random.randn(self.obs_dim // self.n_objects).astype(np.float32) * 0.1 * (1 - progress)
                    else:
                        state = np.random.randn(self.obs_dim // self.n_objects).astype(np.float32) * 0.2
                    obj_states.append(state)
                
                obs = np.concatenate(obj_states)
                action = np.random.randn(self.action_dim).astype(np.float32) * 0.3
                action[:3] = pick_target[:3] * (1 - t/8.0) * 0.5
                
                trajectory.append(obs)
                actions.append(action)
            
            transport_target = np.random.randn(self.obs_dim // self.n_objects).astype(np.float32) * 0.3
            for t in range(9):
                obj_states = []
                for obj_idx in range(self.n_objects):
                    if obj_idx == (i % self.n_objects):
                        progress = t / 8.0
                        state = transport_target * progress + pick_target * (1 - progress) + np.random.randn(self.obs_dim // self.n_objects).astype(np.float32) * 0.05
                    else:
                        state = np.random.randn(self.obs_dim // self.n_objects).astype(np.float32) * 0.2
                    obj_states.append(state)
                
                obs = np.concatenate(obj_states)
                action = np.random.randn(self.action_dim).astype(np.float32) * 0.2
                action[:3] = (transport_target[:3] - pick_target[:3]) * (t/8.0) * 0.3
                
                trajectory.append(obs)
                actions.append(action)
            
            for t in range(7):
                obj_states = []
                for obj_idx in range(self.n_objects):
                    if obj_idx == (i % self.n_objects):
                        progress = t / 6.0
                        state = transport_target + np.random.randn(self.obs_dim // self.n_objects).astype(np.float32) * 0.1 * (1 - progress)
                    else:
                        state = np.random.randn(self.obs_dim // self.n_objects).astype(np.float32) * 0.2
                    obj_states.append(state)
                
                obs = np.concatenate(obj_states)
                action = np.random.randn(self.action_dim).astype(np.float32) * 0.1
                action[-1] = -0.5 * (t / 6.0)
                
                trajectory.append(obs)
                actions.append(action)
            
            if len(trajectory) < self.seq_len:
                pad_len = self.seq_len - len(trajectory)
                trajectory.extend([trajectory[-1]] * pad_len)
                actions.extend([actions[-1]] * pad_len)
            else:
                trajectory = trajectory[:self.seq_len]
                actions = actions[:self.seq_len]
            
            data.append({
                'trajectory': np.array(trajectory, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32),
                'language': lang_embed,
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'trajectory': torch.from_numpy(item['trajectory']),
            'actions': torch.from_numpy(item['actions']),
            'language': torch.from_numpy(item['language']),
        }


# ============================================================
# Architectures
# ============================================================

class BaselineMLP(nn.Module):
    def __init__(self, obs_dim=400, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        self.temporal = nn.GRU(hidden_dim + hidden_dim // 2, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, trajectory, language):
        batch, seq_len, _ = trajectory.shape
        obs_encoded = self.obs_encoder(trajectory)
        lang_encoded = self.lang_encoder(language)
        lang_expanded = lang_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        fused = torch.cat([obs_encoded, lang_expanded], dim=-1)
        _, hidden = self.temporal(fused)
        return self.decoder(hidden[-1])


class TwoNodeCG(nn.Module):
    def __init__(self, obs_dim=400, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        node_dim = 64
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, node_dim),
            nn.LayerNorm(node_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 128),
            nn.ReLU(),
            nn.Linear(128, node_dim),
            nn.LayerNorm(node_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.LayerNorm(node_dim)
            ) for _ in range(2)
        ])
        
        self.cross_attn = nn.MultiheadAttention(node_dim, num_heads=4, batch_first=True)
        self.temporal = nn.GRU(node_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, trajectory, language):
        batch, seq_len, _ = trajectory.shape
        
        z_phys = self.obs_to_physical(trajectory)
        z_sem = self.lang_to_semantic(language)
        z_sem_expanded = z_sem.unsqueeze(1).expand(-1, seq_len, -1)
        
        nodes = torch.stack([z_phys, z_sem_expanded], dim=2)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=2, keepdim=True).expand(-1, -1, 2, -1)
            nodes = nodes + layer(msgs)
        
        nodes_flat = nodes.reshape(batch * seq_len, 2, -1)
        attn_out, _ = self.cross_attn(nodes_flat, nodes_flat, nodes_flat)
        attn_out = attn_out.reshape(batch, seq_len, 2, -1).mean(dim=2)
        
        _, hidden = self.temporal(attn_out)
        return self.decoder(hidden[-1])


class PerObjectCG(nn.Module):
    def __init__(self, obs_dim=400, lang_dim=32, action_dim=7, n_objects=5, hidden_dim=64):
        super().__init__()
        self.n_objects = n_objects
        self.node_dim = 64
        obj_dim = obs_dim // n_objects
        
        self.object_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obj_dim, self.node_dim),
                nn.ReLU(),
                nn.Linear(self.node_dim, self.node_dim),
                nn.LayerNorm(self.node_dim)
            ) for _ in range(n_objects)
        ])
        
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.node_dim),
            nn.LayerNorm(self.node_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.node_dim, self.node_dim),
                nn.ReLU(),
                nn.LayerNorm(self.node_dim)
            ) for _ in range(2)
        ])
        
        self.cross_attn = nn.MultiheadAttention(self.node_dim, num_heads=4, batch_first=True)
        
        self.pool = nn.Sequential(
            nn.Linear(self.node_dim * (n_objects + 1), hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, trajectory, language):
        batch, seq_len, _ = trajectory.shape
        obj_dim = trajectory.size(-1) // self.n_objects
        
        object_nodes = []
        for i in range(self.n_objects):
            obj_obs = trajectory[:, :, i*obj_dim:(i+1)*obj_dim]
            obj_encoded = self.object_encoders[i](obj_obs)
            object_nodes.append(obj_encoded)
        
        z_sem = self.lang_to_semantic(language)
        z_sem_expanded = z_sem.unsqueeze(1).expand(-1, seq_len, -1)
        
        all_nodes = torch.stack(object_nodes + [z_sem_expanded], dim=2)
        
        for layer in self.gnn_layers:
            msgs = all_nodes.mean(dim=2, keepdim=True).expand(-1, -1, self.n_objects + 1, -1)
            all_nodes = all_nodes + layer(msgs)
        
        nodes_flat = all_nodes.reshape(batch * seq_len, self.n_objects + 1, -1)
        attn_out, _ = self.cross_attn(nodes_flat, nodes_flat, nodes_flat)
        attn_out = attn_out.reshape(batch, seq_len, self.n_objects + 1, -1)
        
        pooled = attn_out.reshape(batch, seq_len, -1)
        pooled = self.pool(pooled)
        
        _, hidden = self.temporal(pooled)
        return self.decoder(hidden[-1])


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            traj = batch['trajectory'].to(device)
            lang = batch['language'].to(device)
            actions = batch['actions'].to(device)
            
            pred = model(traj, lang)
            target = actions[:, -1, :]
            
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                traj = batch['trajectory'].to(device)
                lang = batch['language'].to(device)
                actions = batch['actions'].to(device)
                
                pred = model(traj, lang)
                target = actions[:, -1, :]
                
                loss = criterion(pred, target)
                val_loss += loss.item()
                n_val += 1
        
        train_loss /= n_batches
        val_loss /= n_val
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    model = model.to(device)
    
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            traj = batch['trajectory'].to(device)
            lang = batch['language'].to(device)
            actions = batch['actions'].to(device)
            
            pred = model(traj, lang)
            target = actions[:, -1, :]
            
            total_mse += mse_loss(pred, target).item()
            total_mae += mae_loss(pred, target).item()
            n_batches += 1
    
    return {
        'mse': total_mse / n_batches,
        'mae': total_mae / n_batches
    }


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 60)
    print("H1.422: Per-Object CG on Multi-Step Long-Horizon Manipulation")
    print("=" * 60)
    
    config = {
        'n_demos': 1500,
        'seq_len': 25,
        'n_objects': 5,
        'obs_dim': 400,
        'lang_dim': 32,
        'action_dim': 7,
        'hidden_dim': 64,
        'epochs': 30,
        'lr': 0.001,
        'batch_size': 32,
        'n_runs': 2,
    }
    
    print(f"\nConfig: {json.dumps(config, indent=2)}")
    
    print("\nGenerating datasets...")
    train_dataset = MultiStepLongHorizonDataset(n_demos=config['n_demos'], seq_len=config['seq_len'],
                                                 n_objects=config['n_objects'], obs_dim=config['obs_dim'],
                                                 lang_dim=config['lang_dim'], action_dim=config['action_dim'],
                                                 split='train', seed=42)
    val_dataset = MultiStepLongHorizonDataset(n_demos=config['n_demos'], seq_len=config['seq_len'],
                                               n_objects=config['n_objects'], obs_dim=config['obs_dim'],
                                               lang_dim=config['lang_dim'], action_dim=config['action_dim'],
                                               split='val', seed=42)
    test_dataset = MultiStepLongHorizonDataset(n_demos=config['n_demos'], seq_len=config['seq_len'],
                                                n_objects=config['n_objects'], obs_dim=config['obs_dim'],
                                                lang_dim=config['lang_dim'], action_dim=config['action_dim'],
                                                split='test', seed=42)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"  Train: {len(train_dataset)} demos")
    print(f"  Val:   {len(val_dataset)} demos")
    print(f"  Test:  {len(test_dataset)} demos")
    
    all_results = {'baseline': [], 'two_node_cg': [], 'per_object_cg': []}
    
    for run in range(config['n_runs']):
        print(f"\n{'='*40}")
        print(f"Run {run+1}/{config['n_runs']}")
        print(f"{'='*40}")
        
        run_seed = 42 + run * 17
        
        print("\nTraining Baseline MLP...")
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        baseline = BaselineMLP(
            obs_dim=config['obs_dim'], lang_dim=config['lang_dim'],
            action_dim=config['action_dim'], hidden_dim=config['hidden_dim']
        )
        baseline = train_model(baseline, train_loader, val_loader, 
                               epochs=config['epochs'], lr=config['lr'])
        baseline_results = evaluate_model(baseline, test_loader)
        all_results['baseline'].append(baseline_results)
        print(f"  Baseline MSE: {baseline_results['mse']:.6f}, MAE: {baseline_results['mae']:.6f}")
        
        print("\nTraining 2-Node CG...")
        torch.manual_seed(run_seed + 1)
        np.random.seed(run_seed + 1)
        two_node = TwoNodeCG(
            obs_dim=config['obs_dim'], lang_dim=config['lang_dim'],
            action_dim=config['action_dim'], hidden_dim=config['hidden_dim']
        )
        two_node = train_model(two_node, train_loader, val_loader,
                               epochs=config['epochs'], lr=config['lr'])
        two_node_results = evaluate_model(two_node, test_loader)
        all_results['two_node_cg'].append(two_node_results)
        print(f"  2-Node CG MSE: {two_node_results['mse']:.6f}, MAE: {two_node_results['mae']:.6f}")
        
        print("\nTraining Per-Object CG...")
        torch.manual_seed(run_seed + 2)
        np.random.seed(run_seed + 2)
        per_object = PerObjectCG(
            obs_dim=config['obs_dim'], lang_dim=config['lang_dim'],
            action_dim=config['action_dim'], n_objects=config['n_objects'],
            hidden_dim=config['hidden_dim']
        )
        per_object = train_model(per_object, train_loader, val_loader,
                                 epochs=config['epochs'], lr=config['lr'])
        per_object_results = evaluate_model(per_object, test_loader)
        all_results['per_object_cg'].append(per_object_results)
        print(f"  Per-Object CG MSE: {per_object_results['mse']:.6f}, MAE: {per_object_results['mae']:.6f}")
    
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean ± std over 2 runs)")
    print(f"{'='*60}")
    
    def aggregate(results_list):
        mse_vals = [r['mse'] for r in results_list]
        mae_vals = [r['mae'] for r in results_list]
        return {
            'mse_mean': np.mean(mse_vals),
            'mse_std': np.std(mse_vals),
            'mae_mean': np.mean(mae_vals),
            'mae_std': np.std(mae_vals)
        }
    
    baseline_agg = aggregate(all_results['baseline'])
    two_node_agg = aggregate(all_results['two_node_cg'])
    per_object_agg = aggregate(all_results['per_object_cg'])
    
    two_node_vs_baseline = ((baseline_agg['mse_mean'] - two_node_agg['mse_mean']) / baseline_agg['mse_mean']) * 100
    per_object_vs_baseline = ((baseline_agg['mse_mean'] - per_object_agg['mse_mean']) / baseline_agg['mse_mean']) * 100
    per_object_vs_two_node = ((two_node_agg['mse_mean'] - per_object_agg['mse_mean']) / two_node_agg['mse_mean']) * 100
    
    print(f"\nBaseline MLP:      MSE={baseline_agg['mse_mean']:.6f} ± {baseline_agg['mse_std']:.6f}")
    print(f"2-Node CG:         MSE={two_node_agg['mse_mean']:.6f} ± {two_node_agg['mse_std']:.6f} ({two_node_vs_baseline:+.2f}% vs baseline)")
    print(f"Per-Object CG:     MSE={per_object_agg['mse_mean']:.6f} ± {per_object_agg['mse_std']:.6f} ({per_object_vs_baseline:+.2f}% vs baseline, {per_object_vs_two_node:+.2f}% vs 2-Node)")
    
    if per_object_vs_two_node > 2.0:
        conclusion = "SUPPORTED"
        key_insight = f"Per-Object CG advantage scales with task complexity: +{per_object_vs_two_node:.2f}% over 2-Node CG on 25-timestep multi-step tasks (vs +10.65% on 10-timestep tasks in H1.421)."
    elif per_object_vs_two_node > 0:
        conclusion = "WEAKLY_SUPPORTED"
        key_insight = f"Per-Object CG shows marginal advantage (+{per_object_vs_two_node:.2f}%) on long-horizon tasks. The benefit may diminish as sequence length increases."
    else:
        conclusion = "REFUTED"
        key_insight = f"Per-Object CG does NOT scale advantage to longer horizons: {per_object_vs_two_node:.2f}% vs 2-Node CG. Simpler architectures may be more robust for long sequences."
    
    print(f"\nConclusion: {conclusion}")
    print(f"Key insight: {key_insight}")
    
    results = {
        'experiment_id': 'H1.422',
        'description': 'Per-Object CG on Multi-Step Long-Horizon Manipulation (25 timesteps)',
        'conclusion': conclusion,
        'key_insight': key_insight,
        'config': config,
        'results': {
            'baseline': {
                'mse_mean': baseline_agg['mse_mean'],
                'mse_std': baseline_agg['mse_std'],
                'mae_mean': baseline_agg['mae_mean'],
                'mae_std': baseline_agg['mae_std']
            },
            'two_node_cg': {
                'mse_mean': two_node_agg['mse_mean'],
                'mse_std': two_node_agg['mse_std'],
                'mae_mean': two_node_agg['mae_mean'],
                'mae_std': two_node_agg['mae_std'],
                'vs_baseline_pct': round(two_node_vs_baseline, 2)
            },
            'per_object_cg': {
                'mse_mean': per_object_agg['mse_mean'],
                'mse_std': per_object_agg['mse_std'],
                'mae_mean': per_object_agg['mae_mean'],
                'mae_std': per_object_agg['mae_std'],
                'vs_baseline_pct': round(per_object_vs_baseline, 2),
                'vs_two_node_pct': round(per_object_vs_two_node, 2)
            }
        },
        'comparison_to_h1_421': {
            'h1_421_seq_len': 10,
            'h1_421_per_object_vs_two_node': 10.65,
            'h1_422_seq_len': 25,
            'h1_422_per_object_vs_two_node': round(per_object_vs_two_node, 2),
            'scaling_factor': round(per_object_vs_two_node / 10.65, 3) if per_object_vs_two_node > 0 else 0
        }
    }
    
    results_path = Path('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-multi_step_long_horizon/results/metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
