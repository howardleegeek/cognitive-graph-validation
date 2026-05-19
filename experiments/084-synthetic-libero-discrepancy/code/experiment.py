"""
H1.443 - Investigate Synthetic vs LIBERO Task Discrepancy (OPTIMIZED)

Hypothesis: GraphCG's failure on LIBERO tasks (vs success on synthetic) is due to:
1. Input representation complexity (structured vs noisy)
2. Task type (transformation prediction vs action prediction)
3. Data scale relative to model capacity

This experiment creates a controlled bridge between synthetic and LIBERO-style tasks.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Data Generation
# ============================================================

def generate_bridge_data(n_samples=500, noise_level=0.0, task_type='transformation', n_objects=3, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    obj_dim = 8  # 3 position + 5 type one-hot
    scenes = []
    for _ in range(n_samples):
        centers = np.random.randn(2, 2) * 2
        objects = []
        for i in range(n_objects):
            center = centers[i % 2]
            pos = center + np.random.randn(2) * 0.5
            z = np.random.uniform(0, 1)
            obj_type = np.zeros(5)
            obj_type[i % 5] = 1
            objects.append(np.concatenate([pos, [z], obj_type]))
        scenes.append(np.array(objects))
    
    scenes = np.array(scenes)
    if noise_level > 0:
        scenes += np.random.randn(*scenes.shape) * noise_level
    
    flat_scenes = scenes.reshape(n_samples, -1)
    
    if task_type == 'transformation':
        goals = np.random.randn(n_samples, n_objects, 3) * 0.3
        transformations = goals.reshape(n_samples, -1)
        transformations += np.random.randn(*transformations.shape) * noise_level * 0.5
        targets = transformations
        target_dim = n_objects * 3
    elif task_type == 'action':
        actions = np.zeros((n_samples, 7))
        for i in range(n_samples):
            type0_mask = scenes[i, :, 3] == 1
            if type0_mask.any():
                target_obj = scenes[i, type0_mask][0]
                actions[i, :3] = target_obj[:3] - np.array([0, 0, 0.5])
                actions[i, 6] = 1.0
            else:
                actions[i, :3] = np.random.randn(3) * 0.5
        actions += np.random.randn(*actions.shape) * noise_level * 0.3
        targets = actions
        target_dim = 7
    
    return torch.FloatTensor(flat_scenes), torch.FloatTensor(targets), target_dim


# ============================================================
# Architectures
# ============================================================

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )
    def forward(self, x):
        return self.net(x)


class GraphCG(nn.Module):
    def __init__(self, input_dim, target_dim, n_objects, obj_dim=8, hidden_dim=64, n_gnn_layers=2):
        super().__init__()
        self.n_objects = n_objects
        self.obj_dim = obj_dim
        
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
            )
            for _ in range(n_gnn_layers)
        ])
        
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        objects = x.view(batch_size, self.n_objects, self.obj_dim)
        node_features = self.obj_encoder(objects)
        
        for gnn in self.gnn_layers:
            mean_msg = node_features.mean(dim=1, keepdim=True).expand(-1, self.n_objects, -1)
            combined = torch.cat([node_features, mean_msg], dim=-1)
            node_features = node_features + gnn(combined)
        
        attn_out, _ = self.attn(node_features, node_features, node_features)
        global_repr = attn_out.mean(dim=1)
        return self.decoder(global_repr)


# ============================================================
# Training
# ============================================================

def train_and_eval(input_dim, target_dim, n_objects, X, y, epochs=40, lr=3e-4):
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    train_loader = DataLoader(TensorDataset(X[:n_train], y[:n_train]), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[n_train:n_train+n_val], y[n_train:n_train+n_val]), batch_size=64)
    test_loader = DataLoader(TensorDataset(X[n_train+n_val:], y[n_train+n_val:]), batch_size=64)
    
    criterion = nn.MSELoss()
    
    # MLP
    mlp = MLPBaseline(input_dim, target_dim, hidden_dim=64)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    for _ in range(epochs):
        mlp.train()
        for bx, by in train_loader:
            opt.zero_grad()
            loss = criterion(mlp(bx), by)
            loss.backward()
            opt.step()
    
    mlp.eval()
    mlp_loss = 0
    with torch.no_grad():
        for bx, by in test_loader:
            mlp_loss += criterion(mlp(bx), by).item()
    mlp_loss /= len(test_loader)
    
    # GraphCG
    graphcg = GraphCG(input_dim, target_dim, n_objects=n_objects, obj_dim=8, hidden_dim=64)
    opt = torch.optim.Adam(graphcg.parameters(), lr=lr)
    for _ in range(epochs):
        graphcg.train()
        for bx, by in train_loader:
            opt.zero_grad()
            loss = criterion(graphcg(bx), by)
            loss.backward()
            opt.step()
    
    graphcg.eval()
    graphcg_loss = 0
    with torch.no_grad():
        for bx, by in test_loader:
            graphcg_loss += criterion(graphcg(bx), by).item()
    graphcg_loss /= len(test_loader)
    
    return mlp_loss, graphcg_loss


# ============================================================
# Experiment
# ============================================================

def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_experiment():
    results = {
        'noise_sweep': [],
        'task_type_comparison': [],
        'data_scale_sweep': [],
        'object_count_sweep': [],
        'combined_stress_test': []
    }
    
    n_trials = 2
    
    # Sweep 1: Noise Level
    print("=" * 60)
    print("SWEEP 1: Noise Level")
    print("=" * 60)
    
    for noise in [0.0, 0.05, 0.1, 0.15, 0.2]:
        mlp_losses, graphcg_losses = [], []
        for trial in range(n_trials):
            X, y, td = generate_bridge_data(n_samples=500, noise_level=noise, task_type='transformation', n_objects=3, seed=42+trial)
            ml, gl = train_and_eval(X.shape[1], td, n_objects=3, X=X, y=y, epochs=40)
            mlp_losses.append(ml)
            graphcg_losses.append(gl)
        
        mlp_mean = np.mean(mlp_losses)
        graphcg_mean = np.mean(graphcg_losses)
        imp = (mlp_mean - graphcg_mean) / mlp_mean * 100
        results['noise_sweep'].append({'noise': noise, 'mlp_loss': float(mlp_mean), 'graphcg_loss': float(graphcg_mean), 'improvement_pct': float(imp), 'graphcg_wins': bool(imp > 0)})
        print(f"  noise={noise:.2f}: MLP={mlp_mean:.6f} GraphCG={graphcg_mean:.6f} imp={imp:+.1f}%")
    
    # Sweep 2: Task Type
    print("\n" + "=" * 60)
    print("SWEEP 2: Task Type")
    print("=" * 60)
    
    for task in ['transformation', 'action']:
        mlp_losses, graphcg_losses = [], []
        for trial in range(n_trials):
            X, y, td = generate_bridge_data(n_samples=500, noise_level=0.05, task_type=task, n_objects=3, seed=42+trial)
            ml, gl = train_and_eval(X.shape[1], td, n_objects=3, X=X, y=y, epochs=40)
            mlp_losses.append(ml)
            graphcg_losses.append(gl)
        
        mlp_mean = np.mean(mlp_losses)
        graphcg_mean = np.mean(graphcg_losses)
        imp = (mlp_mean - graphcg_mean) / mlp_mean * 100
        results['task_type_comparison'].append({'task_type': task, 'mlp_loss': float(mlp_mean), 'graphcg_loss': float(graphcg_mean), 'improvement_pct': float(imp), 'graphcg_wins': bool(imp > 0)})
        print(f"  task={task}: MLP={mlp_mean:.6f} GraphCG={graphcg_mean:.6f} imp={imp:+.1f}%")
    
    # Sweep 3: Data Scale
    print("\n" + "=" * 60)
    print("SWEEP 3: Data Scale")
    print("=" * 60)
    
    for ns in [200, 500, 1000, 2000]:
        mlp_losses, graphcg_losses = [], []
        for trial in range(n_trials):
            X, y, td = generate_bridge_data(n_samples=ns, noise_level=0.05, task_type='transformation', n_objects=3, seed=42+trial)
            ml, gl = train_and_eval(X.shape[1], td, n_objects=3, X=X, y=y, epochs=40)
            mlp_losses.append(ml)
            graphcg_losses.append(gl)
        
        mlp_mean = np.mean(mlp_losses)
        graphcg_mean = np.mean(graphcg_losses)
        imp = (mlp_mean - graphcg_mean) / mlp_mean * 100
        results['data_scale_sweep'].append({'n_samples': int(ns), 'mlp_loss': float(mlp_mean), 'graphcg_loss': float(graphcg_mean), 'improvement_pct': float(imp), 'graphcg_wins': bool(imp > 0)})
        print(f"  samples={ns}: MLP={mlp_mean:.6f} GraphCG={graphcg_mean:.6f} imp={imp:+.1f}%")
    
    # Sweep 4: Object Count
    print("\n" + "=" * 60)
    print("SWEEP 4: Object Count")
    print("=" * 60)
    
    for n_obj in [2, 3, 5, 7]:
        mlp_losses, graphcg_losses = [], []
        for trial in range(n_trials):
            X, y, td = generate_bridge_data(n_samples=500, noise_level=0.05, task_type='transformation', n_objects=n_obj, seed=42+trial)
            ml, gl = train_and_eval(X.shape[1], td, n_objects=n_obj, X=X, y=y, epochs=40)
            mlp_losses.append(ml)
            graphcg_losses.append(gl)
        
        mlp_mean = np.mean(mlp_losses)
        graphcg_mean = np.mean(graphcg_losses)
        imp = (mlp_mean - graphcg_mean) / mlp_mean * 100
        results['object_count_sweep'].append({'n_objects': int(n_obj), 'mlp_loss': float(mlp_mean), 'graphcg_loss': float(graphcg_mean), 'improvement_pct': float(imp), 'graphcg_wins': bool(imp > 0)})
        print(f"  objects={n_obj}: MLP={mlp_mean:.6f} GraphCG={graphcg_mean:.6f} imp={imp:+.1f}%")
    
    # Combined Stress Test
    print("\n" + "=" * 60)
    print("COMBINED STRESS TEST")
    print("=" * 60)
    
    conditions = [
        {'name': 'clean_synthetic', 'noise': 0.0, 'task': 'transformation', 'samples': 500, 'objects': 3},
        {'name': 'noisy_synthetic', 'noise': 0.05, 'task': 'transformation', 'samples': 500, 'objects': 3},
        {'name': 'action_pred', 'noise': 0.05, 'task': 'action', 'samples': 500, 'objects': 3},
        {'name': 'libero_like', 'noise': 0.1, 'task': 'action', 'samples': 500, 'objects': 5},
        {'name': 'libero_hard', 'noise': 0.15, 'task': 'action', 'samples': 300, 'objects': 7},
    ]
    
    for cond in conditions:
        mlp_losses, graphcg_losses = [], []
        for trial in range(n_trials):
            X, y, td = generate_bridge_data(n_samples=cond['samples'], noise_level=cond['noise'], task_type=cond['task'], n_objects=cond['objects'], seed=42+trial)
            ml, gl = train_and_eval(X.shape[1], td, n_objects=cond['objects'], X=X, y=y, epochs=40)
            mlp_losses.append(ml)
            graphcg_losses.append(gl)
        
        mlp_mean = np.mean(mlp_losses)
        graphcg_mean = np.mean(graphcg_losses)
        imp = (mlp_mean - graphcg_mean) / mlp_mean * 100
        results['combined_stress_test'].append({'condition': cond['name'], 'config': cond, 'mlp_loss': float(mlp_mean), 'graphcg_loss': float(graphcg_mean), 'improvement_pct': float(imp), 'graphcg_wins': bool(imp > 0)})
        print(f"  {cond['name']}: MLP={mlp_mean:.6f} GraphCG={graphcg_mean:.6f} imp={imp:+.1f}%")
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    # Convert numpy types before JSON serialization
    results = convert_numpy(results)
    
    output_path = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-synthetic-libero-discrepancy/results/metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
