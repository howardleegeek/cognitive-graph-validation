#!/usr/bin/env python3
"""
H3.100: Combined Task Structure (Endpoint + Subgoals + Actions)
Tests if combining all successful structure types provides synergistic benefit.

Key insights:
- H3.98: Hierarchical goal decomposition (+16.4%)
- H3.99: Action-consequence modeling (+19.0%)
- H3.92: Full structure (goal+subgoals+actions) achieved +87.2%

Hypothesis: Combined structure will outperform individual components.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset


class CombinedStructureDataset(Dataset):
    """Dataset with full task structure: endpoint + subgoals + actions + consequences."""
    
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
        subgoal_idx = torch.linspace(0, self.seq_len - 1, self.n_subgoals + 2, dtype=torch.long)[1:-1]
        subgoals = positions[subgoal_idx]
        
        actions = positions[1:] - positions[:-1]
        actions = torch.cat([actions, torch.zeros(1, 4)], dim=0)
        consequences = positions.clone()
        
        lang = torch.randn(32)
        
        return {
            'observation': positions,
            'endpoint': endpoint,
            'subgoals': subgoals,
            'action': actions,
            'consequence': consequences,
            'language': lang
        }


class BaselineConcat(nn.Module):
    def __init__(self, obs_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 4)
        )
    
    def forward(self, obs):
        return self.net(obs[:, -1])


class EndpointOnlyAttention(nn.Module):
    """Endpoint goal attention (from H3.94)."""
    def __init__(self, obs_dim=4, goal_dim=4, lang_dim=32, hidden=128, n_heads=4):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.goal_proj = nn.Linear(goal_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, 4)
        
    def forward(self, obs_seq, goal, lang):
        obs_emb = self.obs_proj(obs_seq)
        goal_emb = self.goal_proj(goal).unsqueeze(1)
        lang_emb = self.lang_proj(lang).unsqueeze(1)
        context = torch.cat([goal_emb, lang_emb], dim=1)
        attn_out, _ = self.attn(obs_emb, context, context)
        obs_emb = self.norm(obs_emb + attn_out)
        pooled = obs_emb.mean(dim=1)
        return self.fc(pooled)


class CombinedStructureAttention(nn.Module):
    """Combined structure: endpoint + subgoals + actions + consequences."""
    def __init__(self, obs_dim=4, goal_dim=4, subgoal_dim=4, action_dim=4, cons_dim=4, lang_dim=32, hidden=128, n_heads=4):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.goal_proj = nn.Linear(goal_dim, hidden)
        self.subgoal_proj = nn.Linear(subgoal_dim, hidden)
        self.action_proj = nn.Linear(action_dim, hidden)
        self.cons_proj = nn.Linear(cons_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, 4)
        
    def forward(self, obs_seq, endpoint, subgoals, actions, consequences, lang):
        obs_emb = self.obs_proj(obs_seq)
        goal_emb = self.goal_proj(endpoint).unsqueeze(1)
        subgoal_emb = self.subgoal_proj(subgoals)
        action_emb = self.action_proj(actions.mean(1))  # Average actions
        cons_emb = self.cons_proj(consequences.mean(1))  # Average consequences
        lang_emb = self.lang_proj(lang).unsqueeze(1)
        context = torch.cat([goal_emb, subgoal_emb, action_emb.unsqueeze(1), cons_emb.unsqueeze(1), lang_emb], dim=1)
        attn_out, _ = self.attn(obs_emb, context, context)
        obs_emb = self.norm(obs_emb + attn_out)
        pooled = obs_emb.mean(dim=1)
        return self.fc(pooled)


def train_and_evaluate(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            action = batch['action'][:, -1]
            
            if isinstance(model, CombinedStructureAttention):
                pred = model(batch['observation'], batch['endpoint'], batch['subgoals'], batch['action'], batch['consequence'], batch['language'])
            elif isinstance(model, EndpointOnlyAttention):
                pred = model(batch['observation'], batch['endpoint'], batch['language'])
            else:
                pred = model(batch['observation'])
            
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                action = batch['action'][:, -1]
                if isinstance(model, CombinedStructureAttention):
                    pred = model(batch['observation'], batch['endpoint'], batch['subgoals'], batch['action'], batch['consequence'], batch['language'])
                elif isinstance(model, EndpointOnlyAttention):
                    pred = model(batch['observation'], batch['endpoint'], batch['language'])
                else:
                    pred = model(batch['observation'])
                val_losses.append(criterion(pred, action).item())
        
        best_loss = min(best_loss, np.mean(val_losses))
    
    return best_loss


def run_experiment():
    print("=" * 70)
    print("H3.100: Combined Task Structure (Endpoint + Subgoals + Actions)")
    print("=" * 70)
    
    results = {}
    seq_lengths = [75, 100, 125, 150]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing {seq_len} steps ---")
        
        train_data = CombinedStructureDataset(n_samples=80, seq_len=seq_len, n_subgoals=3)
        val_data = CombinedStructureDataset(n_samples=20, seq_len=seq_len, n_subgoals=3)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        
        baseline = BaselineConcat()
        base_loss = train_and_evaluate(baseline, train_loader, val_loader)
        
        endpoint = EndpointOnlyAttention()
        endpoint_loss = train_and_evaluate(endpoint, train_loader, val_loader)
        
        combined = CombinedStructureAttention()
        combined_loss = train_and_evaluate(combined, train_loader, val_loader)
        
        endpoint_delta = (base_loss - endpoint_loss) / base_loss * 100
        combined_delta = (base_loss - combined_loss) / base_loss * 100
        combined_vs_endpoint = (endpoint_loss - combined_loss) / endpoint_loss * 100
        
        results[seq_len] = {
            'baseline_mse': float(base_loss),
            'endpoint_mse': float(endpoint_loss),
            'combined_mse': float(combined_loss),
            'endpoint_delta': float(endpoint_delta),
            'combined_delta': float(combined_delta),
            'combined_vs_endpoint': float(combined_vs_endpoint),
            'combined_wins': bool(combined_loss < endpoint_loss)
        }
        
        print(f"  Baseline MSE: {base_loss:.6f}")
        print(f"  Endpoint MSE: {endpoint_loss:.6f} ({endpoint_delta:+.1f}%)")
        print(f"  Combined MSE: {combined_loss:.6f} ({combined_delta:+.1f}%)")
        print(f"  Combined vs Endpoint: {combined_vs_endpoint:+.1f}%")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_combined_delta = np.mean([r['combined_delta'] for r in results.values()])
    avg_combined_vs_endpoint = np.mean([r['combined_vs_endpoint'] for r in results.values()])
    combined_wins = sum(1 for r in results.values() if r['combined_wins'])
    
    print(f"Average Combined vs Baseline: {avg_combined_delta:+.1f}%")
    print(f"Average Combined vs Endpoint: {avg_combined_vs_endpoint:+.1f}%")
    print(f"Combined wins: {combined_wins}/{len(seq_lengths)}")
    
    status = "SUPPORTED" if combined_wins >= len(seq_lengths) // 2 + 1 and avg_combined_vs_endpoint > 0 else "REFUTED"
    print(f"Status: {status}")
    
    final_results = {
        'experiment': 'H3.100',
        'hypothesis': 'Combined structure (endpoint+subgoals+actions) outperforms individual components',
        'avg_combined_delta': float(avg_combined_delta),
        'avg_combined_vs_endpoint': float(avg_combined_vs_endpoint),
        'combined_wins': combined_wins,
        'total_tests': len(seq_lengths),
        'status': status,
        'per_length': results
    }
    
    print("\n" + json.dumps(final_results, indent=2))
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.100-combined-structure/results/metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


if __name__ == "__main__":
    run_experiment()