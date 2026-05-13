#!/usr/bin/env python3
"""
H3.140: Attention on 20-30 Step Sequences with Autocorrelation
Based on H3 findings: attention works WITH task structure (goals, autocorrelation)
Tests whether attention can handle 20-30 step sequences with high autocorrelation.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from datetime import datetime


class LongSequenceDataset(Dataset):
    """Dataset with 20-30 step sequences and high autocorrelation."""
    
    def __init__(self, n_samples=500, n_steps_list=[20, 22, 25, 28, 30], rho=0.95):
        self.n_samples = n_samples
        self.n_steps_list = n_steps_list
        self.rho = rho
        self.obs_dim = 8
        self.lang_dim = 32
        self.action_dim = 7
        
        self.data = []
        for _ in range(n_samples):
            n_steps = np.random.choice(n_steps_list)
            
            # Generate smooth trajectory with autocorrelation
            obs = np.random.randn(n_steps, self.obs_dim) * 0.1
            for t in range(1, n_steps):
                obs[t] = rho * obs[t-1] + (1 - rho) * obs[t]
            
            # Language: goal-conditioned
            lang = np.random.randn(n_steps, self.lang_dim) * 0.1
            
            # Actions: smooth with autocorrelation
            action = np.random.randn(n_steps, self.action_dim) * 0.1
            for t in range(1, n_steps):
                action[t] = rho * action[t-1] + (1 - rho) * action[t]
            
            self.data.append({
                'observation': obs,
                'language': lang,
                'action': action,
                'n_steps': n_steps
            })
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def pad_sequence(seq, max_len=30):
    if len(seq) < max_len:
        padding = np.zeros((max_len - len(seq), seq.shape[1]))
        return np.vstack([seq, padding])
    return seq[:max_len]


def collate_fn(batch):
    max_steps = max(item['n_steps'] for item in batch)
    max_steps = min(max_steps, 30)
    
    obs = []
    lang = []
    action = []
    
    for item in batch:
        obs.append(pad_sequence(item['observation'], max_steps))
        lang.append(pad_sequence(item['language'], max_steps))
        action.append(pad_sequence(item['action'], max_steps))
    
    return {
        'observation': np.array(obs),
        'language': np.array(lang),
        'action': np.array(action),
        'n_steps': [item['n_steps'] for item in batch]
    }


class ConcatenationBaseline(nn.Module):
    """Baseline: Concatenation fusion."""
    
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
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        obs_flat = obs.reshape(-1, obs.shape[-1])
        lang_flat = lang.reshape(-1, lang.shape[-1])
        
        obs_enc = self.obs_encoder(obs_flat)
        lang_enc = self.lang_encoder(lang_flat)
        
        fused = self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))
        return fused.reshape(batch_size, seq_len, -1)


class AttentionModel(nn.Module):
    """Attention-based fusion."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        # Cross-attention between obs and lang
        self.cross_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        obs_flat = obs.reshape(-1, obs.shape[-1])
        lang_flat = lang.reshape(-1, lang.shape[-1])
        
        obs_enc = self.obs_encoder(obs_flat)
        lang_enc = self.lang_encoder(lang_flat)
        
        # Reshape for attention
        obs_enc = obs_enc.reshape(batch_size, seq_len, -1)
        lang_enc = lang_enc.reshape(batch_size, seq_len, -1)
        
        # Cross-attention
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        attn_out, _ = self.cross_attn(combined, combined, combined)
        
        # Average over sequence
        attn_out = attn_out.mean(dim=1)
        
        return self.decoder(attn_out)


def train_and_evaluate(model, train_data, val_data, epochs=50, is_attention=False):
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=16, collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss(reduction='mean')
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            obs = torch.FloatTensor(batch['observation'])
            lang = torch.FloatTensor(batch['language'])
            action = torch.FloatTensor(batch['action'])
            
            pred = model(obs, lang)
            
            if is_attention:
                target = action[:, -1, :]
            else:
                target = action
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs = torch.FloatTensor(batch['observation'])
                lang = torch.FloatTensor(batch['language'])
                action = torch.FloatTensor(batch['action'])
                
                pred = model(obs, lang)
                
                if is_attention:
                    target = action[:, -1, :]
                else:
                    target = action
                
                val_losses.append(criterion(pred, target).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


def main():
    print("=" * 70)
    print("H3.140: Attention on 20-30 Step Sequences with Autocorrelation")
    print("Testing whether attention can handle 20-30 step sequences with high rho")
    print("=" * 70)
    
    results = {}
    
    # Test different autocorrelation levels
    for rho in [0.90, 0.93, 0.95, 0.98]:
        print(f"\n--- Testing rho={rho} ---")
        
        train_data = LongSequenceDataset(n_samples=500, n_steps_list=[20, 22, 25, 28, 30], rho=rho)
        val_data = LongSequenceDataset(n_samples=100, n_steps_list=[20, 22, 25, 28, 30], rho=rho)
        
        # Baseline (concatenation)
        print(f"  Training Concatenation (rho={rho})...")
        concat = ConcatenationBaseline()
        concat_loss = train_and_evaluate(concat, train_data, val_data, epochs=50, is_attention=False)
        results[f'concat_rho_{rho}'] = concat_loss
        print(f"    Concatenation MSE: {concat_loss:.6f}")
        
        # Attention
        print(f"  Training Attention (rho={rho})...")
        attn = AttentionModel()
        attn_loss = train_and_evaluate(attn, train_data, val_data, epochs=50, is_attention=True)
        results[f'attn_rho_{rho}'] = attn_loss
        print(f"    Attention MSE: {attn_loss:.6f}")
        
        # Calculate improvement
        improvement = (concat_loss - attn_loss) / concat_loss * 100
        results[f'improvement_rho_{rho}'] = improvement
        print(f"    Improvement: +{improvement:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    best_rho = None
    best_improvement = float('-inf')
    
    for rho in [0.90, 0.93, 0.95, 0.98]:
        improvement = results.get(f'improvement_rho_{rho}', 0)
        print(f"rho={rho}: +{improvement:.1f}%")
        if improvement > best_improvement:
            best_improvement = improvement
            best_rho = rho
    
    print(f"\nBest rho: {best_rho} (+{best_improvement:.1f}%)")
    
    # Determine status
    avg_improvement = np.mean([results.get(f'improvement_rho_{rho}', 0) for rho in [0.90, 0.93, 0.95, 0.98]])
    status = 'SUPPORTED' if avg_improvement > 0 else 'REFUTED'
    
    print(f"Average improvement: +{avg_improvement:.1f}%")
    print(f"Status: {status}")
    
    # Save results
    output = {
        'experiment_id': 'H3.140',
        'hypothesis': 'H3.140',
        'description': 'Attention on 20-30 step sequences with autocorrelation',
        'result': {
            'avg_improvement': avg_improvement,
            'best_rho': best_rho,
            'best_improvement': best_improvement,
            'all_results': results
        },
        'status': status,
        'note': f'Attention on 20-30 steps with autocorrelation: best at rho={best_rho} (+{best_improvement:.1f}%)',
        'timestamp': datetime.now().isoformat()
    }
    
    import os
    results_dir = os.path.dirname(os.path.abspath(__file__)) + '/../results'
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/metrics.json")
    
    return output


if __name__ == '__main__':
    main()