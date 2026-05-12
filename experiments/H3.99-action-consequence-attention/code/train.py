#!/usr/bin/env python3
"""
H3.99: Action-Consequence Modeling with Attention
Tests if modeling action outcomes (consequences) further enables attention.

Key insights:
- H3.92: Full task structure (goal + subgoals + actions + constraints) best (+87.2%)
- H3.93: Action-consequence modeling critical for attention
- H3.98: Hierarchical goal decomposition helps (+16% vs endpoint alone)

Hypothesis: Action-consequence modeling (predicting action outcomes) combined with
endpoint goals will enable attention on even longer sequences (150-300 steps).
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset


class ActionConsequenceDataset(Dataset):
    """Generate manipulation trajectories with action-consequence structure."""
    
    def __init__(self, n_samples, seq_len):
        self.n_samples = n_samples
        self.seq_len = seq_len
        
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
        
        actions = positions[1:] - positions[:-1]
        actions = torch.cat([actions, torch.zeros(1, 4)], dim=0)
        
        consequences = positions.clone()
        
        lang = torch.randn(32)
        
        return {
            'observation': positions,
            'endpoint': endpoint,
            'action': actions,
            'consequence': consequences,
            'language': lang
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
        obs_emb = self.obs_proj(obs_seq)
        endpoint_emb = self.endpoint_proj(endpoint).unsqueeze(1)
        lang_emb = self.lang_proj(lang).unsqueeze(1)
        
        context = torch.cat([endpoint_emb, lang_emb], dim=1)
        
        attn_out, _ = self.attn(obs_emb, context, context)
        obs_emb = self.norm1(obs_emb + attn_out)
        
        pooled = obs_emb.mean(dim=1)
        return self.fc(self.norm2(pooled))


class ActionConsequenceAttention(nn.Module):
    """Attention with action-consequence modeling."""
    
    def __init__(self, obs_dim=4, action_dim=4, consequence_dim=4, endpoint_dim=4, lang_dim=32, hidden=128, n_heads=4):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.action_proj = nn.Linear(action_dim, hidden)
        self.consequence_proj = nn.Linear(consequence_dim, hidden)
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
        
    def forward(self, obs_seq, actions, consequences, endpoint, lang):
        obs_emb = self.obs_proj(obs_seq)
        action_emb = self.action_proj(actions)
        consequence_emb = self.consequence_proj(consequences)
        endpoint_emb = self.endpoint_proj(endpoint).unsqueeze(1)
        lang_emb = self.lang_proj(lang).unsqueeze(1)
        
        context = torch.cat([action_emb, consequence_emb, endpoint_emb, lang_emb], dim=1)
        
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
            action = batch['action'][:, -1]
            
            optimizer.zero_grad()
            if isinstance(model, ActionConsequenceAttention):
                pred = model(obs, batch['action'], batch['consequence'], batch['endpoint'], batch['language'])
            else:
                pred = model(obs, batch['endpoint'], batch['language'])
            
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                action = batch['action'][:, -1]
                
                if isinstance(model, ActionConsequenceAttention):
                    pred = model(obs, batch['action'], batch['consequence'], batch['endpoint'], batch['language'])
                else:
                    pred = model(obs, batch['endpoint'], batch['language'])
                
                val_losses.append(criterion(pred, action).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


def run_experiment():
    """Run H3.99 experiment."""
    print("=" * 70)
    print("H3.99: Action-Consequence Modeling with Attention")
    print("=" * 70)
    
    results = {}
    seq_lengths = [75, 100, 125, 150]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing {seq_len} steps ---")
        
        train_data = ActionConsequenceDataset(n_samples=80, seq_len=seq_len)
        val_data = ActionConsequenceDataset(n_samples=20, seq_len=seq_len)
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        
        baseline = BaselineConcat()
        base_loss = train_and_evaluate(baseline, train_loader, val_loader)
        
        endpoint = EndpointAttention()
        endpoint_loss = train_and_evaluate(endpoint, train_loader, val_loader)
        
        action_consequence = ActionConsequenceAttention()
        ac_loss = train_and_evaluate(action_consequence, train_loader, val_loader)
        
        endpoint_delta = (base_loss - endpoint_loss) / base_loss * 100
        ac_delta = (base_loss - ac_loss) / base_loss * 100
        ac_vs_endpoint = (endpoint_loss - ac_loss) / endpoint_loss * 100
        
        results[seq_len] = {
            'baseline_mse': float(base_loss),
            'endpoint_mse': float(endpoint_loss),
            'ac_mse': float(ac_loss),
            'endpoint_delta': float(endpoint_delta),
            'ac_delta': float(ac_delta),
            'ac_vs_endpoint': float(ac_vs_endpoint),
            'ac_wins': bool(ac_loss < endpoint_loss)
        }
        
        print(f"  Baseline MSE: {base_loss:.6f}")
        print(f"  Endpoint Attn MSE: {endpoint_loss:.6f} ({endpoint_delta:+.1f}%)")
        print(f"  Action-Conseque MSE: {ac_loss:.6f} ({ac_delta:+.1f}%)")
        print(f"  AC vs Endpoint: {ac_vs_endpoint:+.1f}%")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_ac_delta = np.mean([r['ac_delta'] for r in results.values()])
    avg_endpoint_delta = np.mean([r['endpoint_delta'] for r in results.values()])
    avg_ac_vs_endpoint = np.mean([r['ac_vs_endpoint'] for r in results.values()])
    ac_wins = sum(1 for r in results.values() if r['ac_wins'])
    
    print(f"Average Endpoint vs Baseline: {avg_endpoint_delta:+.1f}%")
    print(f"Average AC vs Baseline: {avg_ac_delta:+.1f}%")
    print(f"Average AC vs Endpoint: {avg_ac_vs_endpoint:+.1f}%")
    print(f"AC wins: {ac_wins}/{len(seq_lengths)}")
    
    if ac_wins >= len(seq_lengths) // 2 + 1 and avg_ac_vs_endpoint > 0:
        status = "SUPPORTED"
    elif ac_wins <= len(seq_lengths) // 2 and avg_ac_vs_endpoint < 0:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    
    print(f"Status: {status}")
    
    final_results = {
        'experiment': 'H3.99',
        'hypothesis': 'Action-consequence modeling enables attention on very long sequences',
        'avg_endpoint_delta': float(avg_endpoint_delta),
        'avg_ac_delta': float(avg_ac_delta),
        'avg_ac_vs_endpoint': float(avg_ac_vs_endpoint),
        'ac_wins': ac_wins,
        'total_tests': len(seq_lengths),
        'status': status,
        'per_length': results
    }
    
    print("\n" + json.dumps(final_results, indent=2))
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.99-action-consequence-attention/results/metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


if __name__ == "__main__":
    run_experiment()