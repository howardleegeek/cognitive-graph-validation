#!/usr/bin/env python3
"""
H3.141: Attention boundary test at 380-420 steps
Based on H3.135 (fails at 400-410) and H3.134 (fails at 420-440)
Test the exact boundary where attention fails
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

def generate_long_trajectory(n_steps, n_samples=100, obs_dim=8, action_dim=7, autocorrelation=0.95):
    """Generate long trajectories with high autocorrelation - fast version."""
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

def create_dataset(n_steps_list, n_train=100, n_val=30, autocorrelation=0.95):
    """Create dataset for long sequence tasks."""
    all_obs = []
    all_actions = []
    
    for n_steps in n_steps_list:
        trajs = generate_long_trajectory(n_steps, n_train + n_val, autocorrelation=autocorrelation)
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

class ConcatModel(nn.Module):
    """Concatenation baseline."""
    def __init__(self, obs_dim=8, action_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class AttentionModel(nn.Module):
    """Attention model for long sequences."""
    def __init__(self, obs_dim=8, action_dim=7, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        z_expanded = z.unsqueeze(1)
        attn_out, _ = self.attn(z_expanded, z_expanded, z_expanded)
        z = self.norm(z + attn_out.squeeze(1))
        return self.decoder(z)

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    """Train model - fast version."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        
        # Early stopping check every 10 epochs
        if epoch % 10 == 0:
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
    
    if best_state:
        model.load_state_dict(best_state)
    return best_val

def run_experiment():
    print("=" * 70)
    print("H3.141: Attention boundary test at 380-420 steps")
    print("Testing exact boundary where attention fails")
    print("=" * 70)
    
    # Test at boundary steps - reduced for speed
    n_steps_list = [390, 400, 410]
    autocorrelation = 0.95  # Optimal from previous experiments
    
    results = {}
    
    for n_steps in n_steps_list:
        print(f"\n--- Testing {n_steps} step trajectories ---")
        
        train_ds, val_ds = create_dataset([n_steps], n_train=200, n_val=50, autocorrelation=autocorrelation)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        
        # Concatenation baseline
        concat = ConcatModel()
        concat_val = train_model(concat, train_loader, val_loader)
        
        # Attention model
        attn = AttentionModel()
        attn_val = train_model(attn, train_loader, val_loader)
        
        improvement = (concat_val - attn_val) / concat_val * 100
        
        print(f"  Concat MSE: {concat_val:.6f}")
        print(f"  Attention MSE: {attn_val:.6f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        results[n_steps] = {
            'concat_mse': float(concat_val),
            'attn_mse': float(attn_val),
            'improvement': float(improvement),
            'attn_wins': bool(attn_val < concat_val)
        }
    
    # Summary
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    wins = sum(1 for r in results.values() if r['attn_wins'])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average improvement: {avg_improvement:.1f}%")
    print(f"Attention wins: {wins}/{len(results)}")
    
    # Determine status
    if avg_improvement > 10 and wins >= len(results) * 0.8:
        status = "SUPPORTED"
    elif avg_improvement > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    final_results = {
        'experiment_id': 'H3.141',
        'hypothesis': 'H3.141',
        'description': 'Attention boundary test at 380-420 steps',
        'status': status,
        'result': {
            'avg_improvement': float(avg_improvement),
            'attn_wins': int(wins),
            'total': int(len(results)),
            'details': results
        },
        'note': f'{status}: {avg_improvement:.1f}% avg, attention wins {wins}/{len(results)}'
    }
    
    print(json.dumps(final_results, indent=2))
    return final_results

if __name__ == "__main__":
    results = run_experiment()