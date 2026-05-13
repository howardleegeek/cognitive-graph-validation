#!/usr/bin/env python3
"""
H1.239: Sweet spot verification - 10-20 steps with different regularization
Based on H1.237 (+88.9% at 15-25 steps) and H1.238 (fails at 30-40 steps)
Find the optimal complexity range and regularization
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

def generate_complex_trajectory(n_steps, n_samples=200, obs_dim=8, action_dim=7, autocorrelation=0.95):
    """Generate complex multi-step trajectories with high autocorrelation."""
    trajectories = []
    for _ in range(n_samples):
        state = np.random.randn(obs_dim) * 0.5
        traj = []
        for _ in range(n_steps):
            noise = np.random.randn(obs_dim) * 0.1
            state = autocorrelation * state + (1 - autocorrelation) * noise
            
            action = np.random.randn(action_dim) * 0.3
            action[-1] = np.sin(state[0]) * 0.5
            action[-2] = np.cos(state[1]) * 0.5
            
            traj.append((state.copy(), action.copy()))
        trajectories.append(traj)
    return trajectories

def create_dataset(n_steps_list, n_train=200, n_val=50, autocorrelation=0.95):
    """Create dataset for complex multi-step tasks."""
    all_obs = []
    all_actions = []
    
    for n_steps in n_steps_list:
        trajs = generate_complex_trajectory(n_steps, n_train + n_val, autocorrelation=autocorrelation)
        for traj in trajs:
            for step_idx, (obs, action) in enumerate(traj):
                all_obs.append(obs)
                all_actions.append(action)
    
    obs_tensor = torch.FloatTensor(np.array(all_obs))
    action_tensor = torch.FloatTensor(np.array(all_actions))
    
    n = len(obs_tensor)
    n_train_actual = int(n * 0.8)
    
    train_ds = TensorDataset(obs_tensor[:n_train_actual], action_tensor[:n_train_actual])
    val_ds = TensorDataset(obs_tensor[n_train_actual:], action_tensor[n_train_actual:])
    
    return train_ds, val_ds

class BaselineModel(nn.Module):
    """Baseline: Simple concatenation."""
    def __init__(self, obs_dim=8, action_dim=7, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class UnifiedAttentionRegModel(nn.Module):
    """Unified + Attention + Regularization."""
    def __init__(self, obs_dim=8, action_dim=7, hidden=256, reg=0.1):
        super().__init__()
        self.reg = reg
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        
    def forward(self, x):
        z = self.obs_encoder(x)
        
        if x.shape[0] > 1 or hasattr(self, 'attn'):
            z_expanded = z.unsqueeze(1)
            attn_out, _ = self.attn(z_expanded, z_expanded, z_expanded)
            z = self.norm(z + attn_out.squeeze(1))
        
        return self.action_head(z)

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    """Train model with regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=model.reg if hasattr(model, 'reg') else 0.01)
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for obs, actions in train_loader:
            optimizer.zero_grad()
            pred = model(obs)
            loss = criterion(pred, actions)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, actions in val_loader:
                pred = model(obs)
                val_losses.append(criterion(pred, actions).item())
        
        val_loss = np.mean(val_losses)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    return best_val

def run_experiment():
    print("=" * 70)
    print("H1.239: Sweet spot verification - 10-20 steps with different reg")
    print("=" * 70)
    
    n_steps_list = [10, 15, 20]
    reg_values = [0.1, 0.15]
    autocorrelation = 0.95
    
    results = {}
    best_config = None
    best_improvement = -float('inf')
    
    for n_steps in n_steps_list:
        print(f"\n--- Testing {n_steps} step trajectories ---")
        
        train_ds, val_ds = create_dataset([n_steps], n_train=100, n_val=30, autocorrelation=autocorrelation)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        
        # Baseline
        baseline = BaselineModel()
        baseline_val = train_model(baseline, train_loader, val_loader)
        
        step_results = {'baseline_mse': baseline_val, 'configs': {}}
        
        for reg in reg_values:
            unified = UnifiedAttentionRegModel(reg=reg)
            unified_val = train_model(unified, train_loader, val_loader)
            
            improvement = (baseline_val - unified_val) / baseline_val * 100
            
            step_results['configs'][reg] = {
                'unified_mse': unified_val,
                'improvement': improvement,
                'unified_wins': unified_val < baseline_val
            }
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_config = {'n_steps': n_steps, 'reg': reg}
            
            print(f"  reg={reg}: {improvement:+.1f}% (unified={unified_val:.6f})")
        
        results[n_steps] = step_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best config: n_steps={best_config['n_steps']}, reg={best_config['reg']}")
    print(f"Best improvement: {best_improvement:.1f}%")
    
    # Calculate average improvement for each config
    avg_by_steps = {}
    for n_steps, data in results.items():
        improvements = [c['improvement'] for c in data['configs'].values()]
        avg_by_steps[n_steps] = np.mean(improvements)
    
    avg_by_reg = {}
    for reg in reg_values:
        improvements = [results[ns]['configs'][reg]['improvement'] for ns in n_steps_list]
        avg_by_reg[reg] = np.mean(improvements)
    
    print(f"\nAverage by steps: {avg_by_steps}")
    print(f"Average by reg: {avg_by_reg}")
    
    overall_avg = np.mean([avg_by_steps[ns] for ns in n_steps_list])
    wins = sum(1 for ns in n_steps_list if avg_by_steps[ns] > 0)
    
    # Determine status
    if overall_avg > 50 and wins >= 4:
        status = "SUPPORTED"
    elif overall_avg > 20:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    # Convert for JSON
    results_converted = {}
    for k, v in results.items():
        results_converted[int(k)] = {
            'baseline_mse': float(v['baseline_mse']),
            'configs': {float(rk): {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in rv.items()} for rk, rv in v['configs'].items()}
        }
    
    final_results = {
        'experiment_id': 'H1.239',
        'hypothesis': 'H1.239',
        'description': 'Sweet spot verification - 10-20 steps with different regularization',
        'status': status,
        'result': {
            'avg_improvement': float(overall_avg),
            'best_config': best_config,
            'best_improvement': float(best_improvement),
            'avg_by_steps': {int(k): float(v) for k, v in avg_by_steps.items()},
            'avg_by_reg': {float(k): float(v) for k, v in avg_by_reg.items()},
            'wins': int(wins),
            'total': int(len(n_steps_list))
        },
        'note': f'{status}: {overall_avg:.1f}% avg, best at {best_config["n_steps"]} steps with reg={best_config["reg"]}'
    }
    
    print(json.dumps(final_results, indent=2))
    return final_results

if __name__ == "__main__":
    results = run_experiment()