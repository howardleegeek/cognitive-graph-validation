#!/usr/bin/env python3
"""
H1.238: Ultra-complex multi-step tasks (30-40 steps) with optimal config
Based on H1.237 success (+88.9% on 15-25 steps), test even more complex tasks
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
    """Generate ultra-complex multi-step trajectories with high autocorrelation."""
    trajectories = []
    for _ in range(n_samples):
        # Start with random state
        state = np.random.randn(obs_dim) * 0.5
        traj = []
        for _ in range(n_steps):
            # Add temporal structure with autocorrelation
            noise = np.random.randn(obs_dim) * 0.1
            state = autocorrelation * state + (1 - autocorrelation) * noise
            
            # Complex multi-step task: each step depends on previous
            action = np.random.randn(action_dim) * 0.3
            action[-1] = np.sin(state[0]) * 0.5  # Non-linear dependency
            action[-2] = np.cos(state[1]) * 0.5
            
            traj.append((state.copy(), action.copy()))
        trajectories.append(traj)
    return trajectories

def create_dataset(n_steps_list, n_train=200, n_val=50, autocorrelation=0.95):
    """Create dataset for ultra-complex multi-step tasks."""
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
    
    # Split into train/val
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
    """Unified + Attention + Regularization (optimal config from H1.237)."""
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
        # Attention for temporal modeling
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        
    def forward(self, x):
        # Encode observations
        z = self.obs_encoder(x)
        
        # Apply attention if sequence is long enough
        if x.shape[0] > 1 or hasattr(self, 'attn'):
            z_expanded = z.unsqueeze(1)  # [batch, 1, hidden]
            attn_out, _ = self.attn(z_expanded, z_expanded, z_expanded)
            z = self.norm(z + attn_out.squeeze(1))
        
        # Regularization via weight decay (applied in optimizer)
        return self.action_head(z)

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
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
        
        # Validation
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
    print("H1.238: Ultra-complex multi-step tasks (30-40 steps)")
    print("Based on H1.237 success (+88.9% on 15-25 steps)")
    print("=" * 70)
    
    # Test configurations
    n_steps_list = [30, 35, 40]
    autocorrelation = 0.95  # Optimal from H3.140
    
    results = {}
    
    for n_steps in n_steps_list:
        print(f"\n--- Testing {n_steps} step trajectories ---")
        
        # Create dataset
        train_ds, val_ds = create_dataset([n_steps], n_train=200, n_val=50, autocorrelation=autocorrelation)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        
        # Baseline
        baseline = BaselineModel()
        baseline_val = train_model(baseline, train_loader, val_loader)
        
        # Unified + Attention + Reg (optimal config: reg=0.1)
        unified = UnifiedAttentionRegModel(reg=0.1)
        unified_val = train_model(unified, train_loader, val_loader)
        
        improvement = (baseline_val - unified_val) / baseline_val * 100
        
        print(f"  Baseline MSE: {baseline_val:.6f}")
        print(f"  Unified+Attn+Reg MSE: {unified_val:.6f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        results[n_steps] = {
            'baseline_mse': baseline_val,
            'unified_mse': unified_val,
            'improvement': improvement,
            'unified_wins': unified_val < baseline_val
        }
    
    # Summary
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    wins = sum(1 for r in results.values() if r['unified_wins'])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average improvement: {avg_improvement:.1f}%")
    print(f"Unified wins: {wins}/{len(results)}")
    
    # Determine status
    if avg_improvement > 20 and wins == len(results):
        status = "SUPPORTED"
    elif avg_improvement > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    # Convert numpy types to Python types for JSON serialization
    details_converted = {}
    for k, v in results.items():
        details_converted[int(k)] = {
            'baseline_mse': float(v['baseline_mse']),
            'unified_mse': float(v['unified_mse']),
            'improvement': float(v['improvement']),
            'unified_wins': bool(v['unified_wins'])
        }
    
    final_results = {
        'experiment_id': 'H1.238',
        'hypothesis': 'H1.238',
        'description': 'Ultra-complex multi-step tasks (30-40 steps) with optimal config',
        'status': status,
        'result': {
            'avg_improvement': float(avg_improvement),
            'wins': int(wins),
            'total': int(len(results)),
            'details': details_converted
        },
        'note': f'{status}: {avg_improvement:.1f}% avg improvement on {wins}/{len(results)} configs'
    }
    
    print(json.dumps(final_results, indent=2))
    return final_results

if __name__ == "__main__":
    results = run_experiment()