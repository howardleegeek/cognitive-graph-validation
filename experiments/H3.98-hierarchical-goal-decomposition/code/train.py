#!/usr/bin/env python3
"""
H3.98: Hierarchical Goal Decomposition with Attention
Tests if decomposing endpoint goals into hierarchical subgoals further improves attention.

Key insights from prior experiments:
- H3.92: Goal state is CRITICAL (+87.2% with full structure)
- H3.94: Endpoint goal best representation (+94.1%)
- H1.80: Hierarchical planning (+86.6%)
- H3.95: Attention wins 5/5 at 100+ steps with endpoint goal

Hypothesis: Hierarchical goal decomposition (break endpoint into subgoals) will further improve
attention performance on very long sequences (100+ steps).
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset


class HierarchicalGoalDataset(Dataset):
    """Generate manipulation trajectories with hierarchical goal decomposition."""
    
    def __init__(self, n_samples, seq_len, n_subgoals=3):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_subgoals = n_subgoals
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        base = torch.randn(4) * 0.5
        
        positions = torch.zeros(self.seq_len, 4)
        for i in range(4):
            t = torch.arange(self.seq_len).float() / self.seq_len
            positions[:, i] = base[i] * torch.sin(t * np.pi) + torch.randn(self.seq_len) * 0.05
        
        endpoint = positions[-1].clone()
        
        subgoal_indices = torch.linspace(0, self.seq_len - 1, self.n_subgoals + 1, dtype=torch.long)[1:-1]
        subgoals = positions[subgoal_indices]
        
        lang = torch.randn(32)
        
        actions = positions[1:] - positions[:-1]
        actions = torch.cat([actions, torch.zeros(1, 4)], dim=0)
        
        return {
            'observation': positions,
            'endpoint': endpoint,
            'subgoals': subgoals,
            'language': lang,
            'action': actions
        }


class BaselineConcat(nn.Module):
    """Baseline: Concatenate observation + endpoint + language."""
    
    def __init__(self, obs_dim=4, endpoint_dim=4, lang_dim=32, action_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + endpoint_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
    
    def forward(self, obs, endpoint, lang):
        return self.net(torch.cat([obs[:, -1], endpoint, lang], dim=-1))


class EndpointAttention(nn.Module):
    """Attention with endpoint goal conditioning."""
    
    def __init__(self, obs_dim=4, endpoint_dim=4, lang_dim=32, action_dim=4, hidden=128, n_heads=4):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.endpoint_proj = nn.Linear(endpoint_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        
    def forward(self, obs_seq, endpoint, lang):
        B, T, _ = obs_seq.shape
        
        obs_emb = self.obs_proj(obs_seq)
        endpoint_emb = self.endpoint_proj(endpoint).unsqueeze(1)
        lang_emb = self.lang_proj(lang).unsqueeze(1)
        
        context = torch.cat([endpoint_emb, lang_emb], dim=1)
        
        attn_out, _ = self.attn(obs_emb, context, context)
        obs_emb = self.norm1(obs_emb + attn_out)
        
        pooled = obs_emb.mean(dim=1)
        return self.fc(self.norm2(pooled))


class HierarchicalGoalAttention(nn.Module):
    """Attention with hierarchical goal decomposition (endpoint + subgoals)."""
    
    def __init__(self, obs_dim=4, endpoint_dim=4, subgoal_dim=4, n_subgoals=3, lang_dim=32, action_dim=4, hidden=128, n_heads=4):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.endpoint_proj = nn.Linear(endpoint_dim, hidden)
        self.subgoal_proj = nn.Linear(subgoal_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        
        self.n_subgoals = n_subgoals
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        
    def forward(self, obs_seq, endpoint, subgoals, lang):
        B, T, _ = obs_seq.shape
        
        obs_emb = self.obs_proj(obs_seq)
        endpoint_emb = self.endpoint_proj(endpoint).unsqueeze(1)
        subgoal_emb = self.subgoal_proj(subgoals)
        lang_emb = self.lang_proj(lang).unsqueeze(1)
        
        context = torch.cat([endpoint_emb, subgoal_emb, lang_emb], dim=1)
        
        attn_out, _ = self.attn(obs_emb, context, context)
        obs_emb = self.norm1(obs_emb + attn_out)
        
        pooled = obs_emb.mean(dim=1)
        return self.fc(self.norm2(pooled))


def train_and_evaluate(model, train_loader, val_loader, epochs=10):
    """Train and evaluate model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            obs = batch['observation']
            endpoint = batch['endpoint']
            action = batch['action'][:, -1]
            
            optimizer.zero_grad()
            if isinstance(model, EndpointAttention):
                pred = model(obs, endpoint, batch['language'])
            elif isinstance(model, HierarchicalGoalAttention):
                pred = model(obs, endpoint, batch['subgoals'], batch['language'])
            else:
                pred = model(obs, endpoint, batch['language'])
            
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                endpoint = batch['endpoint']
                action = batch['action'][:, -1]
                
                if isinstance(model, EndpointAttention):
                    pred = model(obs, endpoint, batch['language'])
                elif isinstance(model, HierarchicalGoalAttention):
                    pred = model(obs, endpoint, batch['subgoals'], batch['language'])
                else:
                    pred = model(obs, endpoint, batch['language'])
                
                val_losses.append(criterion(pred, action).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


def run_experiment():
    """Run H3.98 experiment."""
    print("=" * 70)
    print("H3.98: Hierarchical Goal Decomposition with Attention")
    print("=" * 70)
    
    results = {}
    seq_lengths = [50, 75, 100, 125, 150]
    n_subgoals_list = [2, 3, 4, 5]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing {seq_len} steps ---")
        
        train_data = HierarchicalGoalDataset(n_samples=80, seq_len=seq_len, n_subgoals=3)
        val_data = HierarchicalGoalDataset(n_samples=20, seq_len=seq_len, n_subgoals=3)
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        
        baseline = BaselineConcat()
        base_loss = train_and_evaluate(baseline, train_loader, val_loader)
        
        endpoint = EndpointAttention()
        endpoint_loss = train_and_evaluate(endpoint, train_loader, val_loader)
        
        hierarchical = HierarchicalGoalAttention(n_subgoals=3)
        hier_loss = train_and_evaluate(hierarchical, train_loader, val_loader)
        
        endpoint_delta = (base_loss - endpoint_loss) / base_loss * 100
        hier_delta = (base_loss - hier_loss) / base_loss * 100
        hier_vs_endpoint = (endpoint_loss - hier_loss) / endpoint_loss * 100
        
        results[seq_len] = {
            'baseline_mse': float(base_loss),
            'endpoint_mse': float(endpoint_loss),
            'hierarchical_mse': float(hier_loss),
            'endpoint_delta': float(endpoint_delta),
            'hier_delta': float(hier_delta),
            'hier_vs_endpoint': float(hier_vs_endpoint),
            'hier_wins': bool(hier_loss < endpoint_loss)
        }
        
        print(f"  Baseline MSE: {base_loss:.6f}")
        print(f"  Endpoint Attn MSE: {endpoint_loss:.6f} ({endpoint_delta:+.1f}%)")
        print(f"  Hierarchical Attn MSE: {hier_loss:.6f} ({hier_delta:+.1f}%)")
        print(f"  Hier vs Endpoint: {hier_vs_endpoint:+.1f}%")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_hier_delta = np.mean([r['hier_delta'] for r in results.values()])
    avg_endpoint_delta = np.mean([r['endpoint_delta'] for r in results.values()])
    avg_hier_vs_endpoint = np.mean([r['hier_vs_endpoint'] for r in results.values()])
    hier_wins = sum(1 for r in results.values() if r['hier_wins'])
    
    print(f"Average Endpoint vs Baseline: {avg_endpoint_delta:+.1f}%")
    print(f"Average Hierarchical vs Baseline: {avg_hier_delta:+.1f}%")
    print(f"Average Hier vs Endpoint: {avg_hier_vs_endpoint:+.1f}%")
    print(f"Hierarchical wins: {hier_wins}/{len(seq_lengths)}")
    
    if hier_wins >= len(seq_lengths) // 2 + 1 and avg_hier_vs_endpoint > 0:
        status = "SUPPORTED"
    elif hier_wins <= len(seq_lengths) // 2 and avg_hier_vs_endpoint < 0:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    
    print(f"Status: {status}")
    
    final_results = {
        'experiment': 'H3.98',
        'hypothesis': 'Hierarchical goal decomposition improves attention on long sequences',
        'avg_endpoint_delta': float(avg_endpoint_delta),
        'avg_hier_delta': float(avg_hier_delta),
        'avg_hier_vs_endpoint': float(avg_hier_vs_endpoint),
        'hier_wins': hier_wins,
        'total_tests': len(seq_lengths),
        'status': status,
        'per_length': results
    }
    
    print("\n" + json.dumps(final_results, indent=2))
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.98-hierarchical-goal-decomposition/results/metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


if __name__ == "__main__":
    run_experiment()