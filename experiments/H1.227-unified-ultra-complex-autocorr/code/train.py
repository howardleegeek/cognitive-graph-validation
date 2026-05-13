#!/usr/bin/env python3
"""
H1.227: Unified architecture on ultra-complex multi-step tasks with autocorrelation
Extends H1.226 (inconclusive) with more structured complexity levels
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset


class UltraComplexDataset(Dataset):
    def __init__(self, n_samples, n_steps, obs_dim=8, action_dim=7, lang_dim=32, autocorrelation=0.95):
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lang_dim = lang_dim
        
        self.observations = []
        self.actions = []
        self.languages = []
        
        for _ in range(n_samples):
            # Generate multi-step trajectory with subgoals
            obs_seq = []
            action_seq = []
            
            # Initial state
            state = np.random.randn(obs_dim) * 0.5
            
            for step in range(n_steps):
                # Each step has multiple timesteps with autocorrelation
                for t in range(10):  # 10 timesteps per step
                    # Apply autocorrelation
                    state = autocorrelation * state + (1 - autocorrelation) * np.random.randn(obs_dim) * 0.2
                    
                    # Action depends on step (subgoal)
                    action = state[:action_dim] + np.random.randn(action_dim) * 0.1
                    action[action_dim-1] = step / n_steps  # Progress indicator
                    
                    obs_seq.append(state.copy())
                    action_seq.append(action.copy())
            
            # Language: describes the full task (endpoint + subgoals)
            lang = np.random.randn(lang_dim)
            lang[:5] = np.random.choice([1, -1], 5)
            lang[5:10] = np.random.choice([1, -1], 5)  # Subgoal tokens
            
            self.observations.append(np.array(obs_seq))
            self.actions.append(np.array(action_seq))
            self.languages.append(lang)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'observation': torch.FloatTensor(self.observations[idx]),
            'action': torch.FloatTensor(self.actions[idx]),
            'language': torch.FloatTensor(self.languages[idx])
        }


class BaselineModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_goal):
        # Use final observation
        final_obs = obs_seq[:, -1, :]
        obs_enc = self.obs_encoder(final_obs)
        lang_enc = self.lang_encoder(lang_goal)
        fused = torch.cat([obs_enc, lang_enc], dim=-1)
        return self.decoder(fused)


class UnifiedModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256, physical_dim=112, semantic_dim=144):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Unified embedding space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Processing in unified space
        self.processor = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128)
        )
        
        self.decoder = nn.Linear(128, action_dim)
    
    def forward(self, obs_seq, lang_goal):
        # Use final observation
        final_obs = obs_seq[:, -1, :]
        
        # Unified embedding
        z_phys = self.obs_to_unified(final_obs)
        z_sem = self.lang_to_unified(lang_goal)
        
        # Concatenate in unified space
        unified = torch.cat([z_phys, z_sem], dim=-1)
        
        # Process
        processed = self.processor(unified)
        
        return self.decoder(processed)


class UnifiedWithAttention(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256, n_heads=4):
        super().__init__()
        
        # Shared encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        
        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128)
        )
        
        self.decoder = nn.Linear(128, action_dim)
    
    def forward(self, obs_seq, lang_goal):
        batch_size = obs_seq.size(0)
        seq_len = obs_seq.size(1)
        
        # Encode all timesteps
        obs_enc = self.obs_encoder(obs_seq)  # (batch, seq_len, hidden)
        
        # Language goal
        lang_enc = self.lang_encoder(lang_goal)  # (batch, hidden)
        
        # Add language as first token
        lang_token = lang_enc.unsqueeze(1)
        combined = torch.cat([lang_token, obs_enc], dim=1)
        
        # Self-attention
        attn_out, _ = self.self_attn(combined, combined, combined)
        
        # Use final timestep
        final = attn_out[:, -1, :]
        
        processed = self.processor(final)
        return self.decoder(processed)


def train_and_evaluate(model, train_data, val_data, epochs=30):
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            obs = batch['observation']
            lang = batch['language']
            actions = batch['action']
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, actions[:, -1, :])
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                lang = batch['language']
                actions = batch['action']
                pred = model(obs, lang)
                val_losses.append(criterion(pred, actions[:, -1, :]).item())
        
        avg_val = np.mean(val_losses)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
    
    return best_val_loss


def run_experiment():
    results = {}
    
    # Test different complexity levels with autocorrelation
    n_steps_list = [5, 8, 12, 15, 20]
    rho = 0.95  # High autocorrelation
    
    print("=" * 70)
    print("H1.227: Unified on ultra-complex multi-step with autocorrelation (rho=0.95)")
    print("=" * 70)
    
    for n_steps in n_steps_list:
        print(f"\n--- Testing n_steps={n_steps} ---")
        
        seq_len = n_steps * 10  # 10 timesteps per step
        
        train_data = UltraComplexDataset(n_samples=200, n_steps=n_steps, autocorrelation=rho)
        val_data = UltraComplexDataset(n_samples=50, n_steps=n_steps, autocorrelation=rho)
        
        # Baseline
        baseline = BaselineModel()
        baseline_loss = train_and_evaluate(baseline, train_data, val_data)
        
        # Unified
        unified = UnifiedModel()
        unified_loss = train_and_evaluate(unified, train_data, val_data)
        
        # Unified + Attention
        unified_attn = UnifiedWithAttention()
        unified_attn_loss = train_and_evaluate(unified_attn, train_data, val_data)
        
        # Calculate improvements
        unified_improvement = ((baseline_loss - unified_loss) / baseline_loss) * 100
        unified_attn_improvement = ((baseline_loss - unified_attn_loss) / baseline_loss) * 100
        
        results[n_steps] = {
            'baseline_mse': float(baseline_loss),
            'unified_mse': float(unified_loss),
            'unified_attn_mse': float(unified_attn_loss),
            'unified_improvement': float(unified_improvement),
            'unified_attn_improvement': float(unified_attn_improvement),
            'unified_wins': bool(unified_improvement > 0),
            'unified_attn_wins': bool(unified_attn_improvement > 0)
        }
        
        print(f"  Baseline MSE: {baseline_loss:.6f}")
        print(f"  Unified MSE: {unified_loss:.6f} ({unified_improvement:+.2f}%)")
        print(f"  Unified+Attn MSE: {unified_attn_loss:.6f} ({unified_attn_improvement:+.2f}%)")
    
    # Summary
    unified_wins = sum(1 for r in results.values() if r['unified_wins'])
    unified_attn_wins = sum(1 for r in results.values() if r['unified_attn_wins'])
    avg_unified_imp = np.mean([r['unified_improvement'] for r in results.values()])
    avg_unified_attn_imp = np.mean([r['unified_attn_improvement'] for r in results.values()])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Unified wins: {unified_wins}/{len(n_steps_list)} (avg: {avg_unified_imp:+.2f}%)")
    print(f"Unified+Attn wins: {unified_attn_wins}/{len(n_steps_list)} (avg: {avg_unified_attn_imp:+.2f}%)")
    
    # Determine status
    if unified_wins >= len(n_steps_list) * 0.6 and avg_unified_imp > 10:
        status = "SUPPORTED"
    elif unified_wins >= len(n_steps_list) * 0.4:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    # Save results
    results_json = {
        'experiment': 'H1.227',
        'status': status,
        'results': {str(k): v for k, v in results.items()},
        'unified_wins': int(unified_wins),
        'unified_attn_wins': int(unified_attn_wins),
        'avg_unified_improvement': float(avg_unified_imp),
        'avg_unified_attn_improvement': float(avg_unified_attn_imp)
    }
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.227-unified-ultra-complex-autocorr/results/metrics.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    return status, results


if __name__ == "__main__":
    status, results = run_experiment()