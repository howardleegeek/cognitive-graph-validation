#!/usr/bin/env python3
"""
H3.126: Attention on 200+ timesteps with maximum autocorrelation
Extends H3.125's success (120-150 steps, +94.6%) to 200+ steps
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, n_samples, seq_len, obs_dim=8, action_dim=7, lang_dim=32, autocorrelation=0.98):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lang_dim = lang_dim
        
        # Generate trajectories with autocorrelation (real robot characteristic)
        self.observations = []
        self.actions = []
        self.languages = []
        
        for _ in range(n_samples):
            # Generate smooth trajectory with autocorrelation
            obs_seq = []
            action_seq = []
            
            # Initial state
            state = np.random.randn(obs_dim) * 0.5
            
            for t in range(seq_len):
                # Apply autocorrelation - smooth transitions
                if t > 0:
                    state = autocorrelation * state + (1 - autocorrelation) * np.random.randn(obs_dim) * 0.3
                
                # Action is derived from state with some noise
                action = state[:action_dim] + np.random.randn(action_dim) * 0.1
                
                obs_seq.append(state)
                action_seq.append(action)
            
            # Language: goal description (endpoint)
            goal_lang = np.random.randn(lang_dim)
            goal_lang[:5] = np.random.choice([1, -1], 5)  # Sparse goal tokens
            
            self.observations.append(np.array(obs_seq))
            self.actions.append(np.array(action_seq))
            self.languages.append(goal_lang)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'observation': torch.FloatTensor(self.observations[idx]),
            'action': torch.FloatTensor(self.actions[idx]),
            'language': torch.FloatTensor(self.languages[idx])
        }


class ConcatenationBaseline(nn.Module):
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
        # obs_seq: (batch, seq_len, obs_dim)
        batch_size = obs_seq.size(0)
        seq_len = obs_seq.size(1)
        
        # Use only final observation for prediction
        final_obs = obs_seq[:, -1, :]  # (batch, obs_dim)
        
        obs_enc = self.obs_encoder(final_obs)  # (batch, hidden)
        
        # Language goal
        lang_enc = self.lang_encoder(lang_goal)  # (batch, hidden)
        
        # Concatenate
        fused = torch.cat([obs_enc, lang_enc], dim=-1)
        
        # Predict final action
        return self.decoder(fused)


class AttentionModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256, n_heads=8):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        # Self-attention across sequence
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_goal):
        # obs_seq: (batch, seq_len, obs_dim)
        batch_size = obs_seq.size(0)
        seq_len = obs_seq.size(1)
        
        obs_enc = self.obs_encoder(obs_seq)  # (batch, seq_len, hidden)
        
        # Add language goal as first token
        lang_enc = self.lang_encoder(lang_goal)  # (batch, hidden)
        lang_enc = lang_enc.unsqueeze(1)  # (batch, 1, hidden)
        
        # Concatenate language goal with observations
        combined = torch.cat([lang_enc, obs_enc], dim=1)  # (batch, seq_len+1, hidden)
        
        # Self-attention across sequence
        attn_out, _ = self.self_attn(combined, combined, combined)
        
        # Use final timestep for prediction
        final_out = attn_out[:, -1, :]  # (batch, hidden)
        
        # Decode
        return self.decoder(final_out)


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
            loss = criterion(pred, actions[:, -1, :])  # Predict final action
            loss.backward()
            optimizer.step()
        
        # Validation
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
    
    # Test different sequence lengths with maximum autocorrelation
    seq_lengths = [180, 200, 220, 240, 260]
    rho = 0.98  # Maximum autocorrelation (from H3.125's success)
    
    print("=" * 70)
    print("H3.126: Attention on 200+ timesteps with max autocorrelation (rho=0.98)")
    print("=" * 70)
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing seq_len={seq_len} ---")
        
        # Create datasets
        train_data = TrajectoryDataset(n_samples=200, seq_len=seq_len, autocorrelation=rho)
        val_data = TrajectoryDataset(n_samples=50, seq_len=seq_len, autocorrelation=rho)
        
        # Baseline (concatenation)
        concat_model = ConcatenationBaseline()
        concat_loss = train_and_evaluate(concat_model, train_data, val_data)
        
        # Attention model
        attn_model = AttentionModel()
        attn_loss = train_and_evaluate(attn_model, train_data, val_data)
        
        # Calculate improvement
        improvement = ((concat_loss - attn_loss) / concat_loss) * 100
        
        results[seq_len] = {
            'concat_mse': concat_loss,
            'attn_mse': attn_loss,
            'improvement': improvement,
            'attn_wins': improvement > 0
        }
        
        print(f"  Concat MSE: {concat_loss:.6f}")
        print(f"  Attn MSE: {attn_loss:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")
    
    # Summary
    attn_wins = sum(1 for r in results.values() if r['attn_wins'])
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Attention wins: {attn_wins}/{len(seq_lengths)}")
    print(f"Average improvement: {avg_improvement:+.2f}%")
    
    # Determine status
    if attn_wins >= len(seq_lengths) * 0.8 and avg_improvement > 20:
        status = "SUPPORTED"
    elif attn_wins >= len(seq_lengths) * 0.5:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    # Save results
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.126-attention-200-plus-steps/results/metrics.json', 'w') as f:
        json.dump({
            'experiment': 'H3.126',
            'status': status,
            'results': results,
            'attn_wins': attn_wins,
            'avg_improvement': avg_improvement
        }, f, indent=2)
    
    return status, results


if __name__ == "__main__":
    status, results = run_experiment()