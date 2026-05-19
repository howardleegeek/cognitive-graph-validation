"""
H1.444 - Architectural Modifications to Fix GraphCG Underperformance

Hypothesis: GraphCG's underperformance on action prediction tasks can be fixed by:
1. Edge-aware attention (replace mean-pooling message passing)
2. Increased object representation dimension (8 → 32)
3. Residual connections between GNN layers

Prediction: At least one modification will reduce or eliminate the MLP deficit on action prediction tasks.
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
# Data Generation (same as H1.443)
# ============================================================

def generate_bridge_data(n_samples=500, noise_level=0.05, task_type='action', n_objects=3, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    obj_dim = 8
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


class GraphCG_Original(nn.Module):
    """Original GraphCG from H1.443 (baseline for comparison)."""
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


class GraphCG_EdgeAware(nn.Module):
    """Modification 1: Edge-aware attention instead of mean-pooling."""
    def __init__(self, input_dim, target_dim, n_objects, obj_dim=8, hidden_dim=64, n_gnn_layers=2):
        super().__init__()
        self.n_objects = n_objects
        self.obj_dim = obj_dim
        
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        # Edge-aware message passing: compute pairwise interactions
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
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
            # Compute pairwise edge messages
            # node_features: (batch, n_obj, hidden)
            n = self.n_objects
            # Expand for pairwise: (batch, n, n, hidden)
            src = node_features.unsqueeze(2).expand(-1, -1, n, -1)
            dst = node_features.unsqueeze(1).expand(-1, n, -1, -1)
            edge_input = torch.cat([src, dst], dim=-1)  # (batch, n, n, 2*hidden)
            edge_messages = self.edge_mlp(edge_input)  # (batch, n, n, hidden)
            # Aggregate messages: sum over source nodes
            aggregated = edge_messages.sum(dim=2)  # (batch, n, hidden)
            node_features = node_features + gnn(aggregated)
        
        attn_out, _ = self.attn(node_features, node_features, node_features)
        global_repr = attn_out.mean(dim=1)
        return self.decoder(global_repr)


class GraphCG_HighDim(nn.Module):
    """Modification 2: Increased object representation dimension (8 → 32)."""
    def __init__(self, input_dim, target_dim, n_objects, obj_dim=32, hidden_dim=64, n_gnn_layers=2):
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


class GraphCG_Residual(nn.Module):
    """Modification 3: Residual connections between GNN layers."""
    def __init__(self, input_dim, target_dim, n_objects, obj_dim=8, hidden_dim=64, n_gnn_layers=3):
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
            # Residual connection
            node_features = node_features + 0.1 * gnn(combined)
        
        attn_out, _ = self.attn(node_features, node_features, node_features)
        global_repr = attn_out.mean(dim=1)
        return self.decoder(global_repr)


class GraphCG_Combined(nn.Module):
    """All modifications combined: edge-aware + high-dim + residual."""
    def __init__(self, input_dim, target_dim, n_objects, obj_dim=32, hidden_dim=64, n_gnn_layers=3):
        super().__init__()
        self.n_objects = n_objects
        self.obj_dim = obj_dim
        
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
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
            n = self.n_objects
            src = node_features.unsqueeze(2).expand(-1, -1, n, -1)
            dst = node_features.unsqueeze(1).expand(-1, n, -1, -1)
            edge_input = torch.cat([src, dst], dim=-1)
            edge_messages = self.edge_mlp(edge_input)
            aggregated = edge_messages.sum(dim=2)
            node_features = node_features + 0.1 * gnn(aggregated)
        
        attn_out, _ = self.attn(node_features, node_features, node_features)
        global_repr = attn_out.mean(dim=1)
        return self.decoder(global_repr)


# ============================================================
# Training
# ============================================================

def train_and_eval(model_class, model_kwargs, input_dim, target_dim, X, y, epochs=50, lr=3e-4):
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    train_loader = DataLoader(TensorDataset(X[:n_train], y[:n_train]), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[n_train:n_train+n_val], y[n_train:n_train+n_val]), batch_size=64)
    test_loader = DataLoader(TensorDataset(X[n_train+n_val:], y[n_train+n_val:]), batch_size=64)
    
    criterion = nn.MSELoss()
    
    model = model_class(input_dim=input_dim, target_dim=target_dim, **model_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_val_loss = float('inf')
    best_state = None
    
    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
        scheduler.step()
        
        # Quick val check
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                val_loss += criterion(model(bx), by).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for bx, by in test_loader:
            test_loss += criterion(model(bx), by).item()
    test_loss /= len(test_loader)
    
    return test_loss


# ============================================================
# Experiment
# ============================================================

def convert_numpy(obj):
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
        'baseline_comparison': [],
        'modification_comparison': [],
        'scaling_analysis': []
    }
    
    n_trials = 2
    n_objects = 3
    noise = 0.05
    task_type = 'action'
    n_samples = 500
    
    print("=" * 60)
    print("H1.444: Architectural Modifications for GraphCG")
    print("=" * 60)
    
    # Baseline: MLP vs Original GraphCG
    print("\n--- Baseline Comparison ---")
    mlp_losses, orig_losses = [], []
    for trial in range(n_trials):
        X, y, td = generate_bridge_data(n_samples=n_samples, noise_level=noise, task_type=task_type, n_objects=n_objects, seed=42+trial)
        
        ml = train_and_eval(MLPBaseline, {'hidden_dim': 64}, X.shape[1], td, X, y, epochs=50)
        gl = train_and_eval(GraphCG_Original, {'n_objects': n_objects, 'obj_dim': 8, 'hidden_dim': 64}, X.shape[1], td, X, y, epochs=50)
        
        mlp_losses.append(ml)
        orig_losses.append(gl)
    
    mlp_mean = np.mean(mlp_losses)
    orig_mean = np.mean(orig_losses)
    orig_imp = (mlp_mean - orig_mean) / mlp_mean * 100
    
    results['baseline_comparison'].append({
        'mlp_loss': float(mlp_mean),
        'graphcg_original_loss': float(orig_mean),
        'improvement_pct': float(orig_imp),
        'graphcg_wins': bool(orig_imp > 0)
    })
    print(f"  MLP: {mlp_mean:.6f}, GraphCG_Original: {orig_mean:.6f}, Improvement: {orig_imp:+.1f}%")
    
    # Test each modification
    print("\n--- Modification Comparison ---")
    
    modifications = [
        {'name': 'edge_aware', 'class': GraphCG_EdgeAware, 'kwargs': {'n_objects': n_objects, 'obj_dim': 8, 'hidden_dim': 64}},
        {'name': 'high_dim', 'class': GraphCG_HighDim, 'kwargs': {'n_objects': n_objects, 'obj_dim': 32, 'hidden_dim': 64}},
        {'name': 'residual', 'class': GraphCG_Residual, 'kwargs': {'n_objects': n_objects, 'obj_dim': 8, 'hidden_dim': 64, 'n_gnn_layers': 3}},
        {'name': 'combined', 'class': GraphCG_Combined, 'kwargs': {'n_objects': n_objects, 'obj_dim': 32, 'hidden_dim': 64, 'n_gnn_layers': 3}},
    ]
    
    for mod in modifications:
        mod_losses = []
        for trial in range(n_trials):
            X, y, td = generate_bridge_data(n_samples=n_samples, noise_level=noise, task_type=task_type, n_objects=n_objects, seed=42+trial)
            
            # For high-dim and combined, we need to expand input
            if mod['name'] in ['high_dim', 'combined']:
                # Expand 8-dim objects to 32-dim by padding
                X_expanded = torch.zeros(X.shape[0], n_objects * 32)
                for i in range(n_objects):
                    X_expanded[:, i*32:(i+1)*32] = F.pad(X[:, i*8:(i+1)*8], (0, 24))
                X_use = X_expanded
                input_dim = n_objects * 32
            else:
                X_use = X
                input_dim = X.shape[1]
            
            gl = train_and_eval(mod['class'], mod['kwargs'], input_dim, td, X_use, y, epochs=50)
            mod_losses.append(gl)
        
        mod_mean = np.mean(mod_losses)
        mod_imp = (mlp_mean - mod_mean) / mlp_mean * 100
        
        results['modification_comparison'].append({
            'modification': mod['name'],
            'loss': float(mod_mean),
            'improvement_pct': float(mod_imp),
            'graphcg_wins': bool(mod_imp > 0),
            'vs_original_pct': float((orig_mean - mod_mean) / orig_mean * 100)
        })
        print(f"  {mod['name']}: {mod_mean:.6f}, Improvement vs MLP: {mod_imp:+.1f}%, vs Original: {(orig_mean - mod_mean) / orig_mean * 100:+.1f}%")
    
    # Scaling analysis: test best modification across object counts
    print("\n--- Scaling Analysis (best modification) ---")
    
    # Find best modification
    best_mod = max(results['modification_comparison'], key=lambda x: x['improvement_pct'])
    best_mod_name = best_mod['modification']
    best_mod_info = next(m for m in modifications if m['name'] == best_mod_name)
    
    for n_obj in [2, 3, 5, 7]:
        mlp_losses, mod_losses = [], []
        for trial in range(n_trials):
            X, y, td = generate_bridge_data(n_samples=n_samples, noise_level=noise, task_type=task_type, n_objects=n_obj, seed=42+trial)
            
            if best_mod_name in ['high_dim', 'combined']:
                X_expanded = torch.zeros(X.shape[0], n_obj * 32)
                for i in range(n_obj):
                    X_expanded[:, i*32:(i+1)*32] = F.pad(X[:, i*8:(i+1)*8], (0, 24))
                X_use = X_expanded
                input_dim = n_obj * 32
            else:
                X_use = X
                input_dim = X.shape[1]
            
            ml = train_and_eval(MLPBaseline, {'hidden_dim': 64}, input_dim, td, X_use, y, epochs=50)
            
            mod_kwargs = {**best_mod_info['kwargs'], 'n_objects': n_obj}
            gl = train_and_eval(best_mod_info['class'], mod_kwargs, input_dim, td, X_use, y, epochs=50)
            
            mlp_losses.append(ml)
            mod_losses.append(gl)
        
        mlp_mean = np.mean(mlp_losses)
        mod_mean = np.mean(mod_losses)
        imp = (mlp_mean - mod_mean) / mlp_mean * 100
        
        results['scaling_analysis'].append({
            'n_objects': int(n_obj),
            'mlp_loss': float(mlp_mean),
            'best_mod_loss': float(mod_mean),
            'improvement_pct': float(imp),
            'graphcg_wins': bool(imp > 0)
        })
        print(f"  objects={n_obj}: MLP={mlp_mean:.6f}, {best_mod_name}={mod_mean:.6f}, Improvement: {imp:+.1f}%")
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    results = convert_numpy(results)
    
    output_path = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/085-graphcg-arch-mods/results/metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
