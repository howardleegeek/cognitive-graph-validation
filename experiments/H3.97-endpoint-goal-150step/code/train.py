#!/usr/bin/env python3
"""
H3.97: Endpoint Goal on 150+ Step Sequences
Tests if endpoint goal representation continues to enable attention on even longer sequences.

Key insight from H3.94-96:
- Endpoint goal enables attention (+94.1%)
- Attention advantage grows with sequence length (+95.3% at 100+ steps)
- Goal state is the CRITICAL component of task structure

Hypothesis: Endpoint goal will continue to enable attention on 150+ step sequences.
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
    """Generate manipulation trajectories with task structure."""
    
    def __init__(self, n_samples, seq_len, include_goal=True, goal_type='endpoint'):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.include_goal = include_goal
        self.goal_type = goal_type
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generate trajectory with temporal autocorrelation (robot-like)
        t = torch.arange(self.seq_len).float()
        base = torch.randn(4)  # random trajectory parameters
        
        # Position trajectory (smooth motion)
        positions = torch.zeros(self.seq_len, 4)
        for i in range(4):
            positions[:, i] = base[i] * torch.sin(t / self.seq_len * np.pi) + torch.randn(self.seq_len) * 0.1
        
        # Add goal state at the end
        if self.include_goal:
            if self.goal_type == 'endpoint':
                goal = positions[-1].clone()  # Final position as goal
            elif self.goal_type == 'trajectory':
                goal = positions.clone()  # Full trajectory as goal
            elif self.goal_type == 'keypoint':
                goal = positions[::self.seq_len//4].mean(0)  # Keypoint summary
            else:
                goal = positions[-1] - positions[0]  # Delta
        else:
            goal = torch.zeros(4)
        
        # Action: next position prediction
        actions = positions[1:] - positions[:-1]
        actions = torch.cat([actions, torch.zeros(1, 4)], dim=0)
        
        # Language: task description
        lang = torch.randn(32)
        
        return {
            'observation': positions,
            'goal': goal,
            'language': lang,
            'action': actions
        }


class BaselineConcat(nn.Module):
    """Baseline: Concatenate observation + goal + language."""
    
    def __init__(self, obs_dim=4, goal_dim=4, lang_dim=32, action_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
    
    def forward(self, obs, goal, lang):
        return self.net(torch.cat([obs, goal, lang], dim=-1))


class AttentionModel(nn.Module):
    """Attention model with goal conditioning."""
    
    def __init__(self, obs_dim=4, goal_dim=4, lang_dim=32, action_dim=4, hidden=128, n_heads=4):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.goal_proj = nn.Linear(goal_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        
    def forward(self, obs_seq, goal, lang):
        # obs_seq: (batch, seq_len, obs_dim)
        B, T, _ = obs_seq.shape
        
        # Project all inputs
        obs_emb = self.obs_proj(obs_seq)  # (B, T, hidden)
        goal_emb = self.goal_proj(goal).unsqueeze(1)  # (B, 1, hidden)
        lang_emb = self.lang_proj(lang).unsqueeze(1)  # (B, 1, hidden)
        
        # Concatenate goal and language as context
        context = torch.cat([goal_emb, lang_emb], dim=1)  # (B, 2, hidden)
        
        # Cross-attention: obs attends to goal and language
        attn_out, _ = self.attn(obs_emb, context, context)
        obs_emb = self.norm1(obs_emb + attn_out)
        
        # Pool and predict
        pooled = obs_emb.mean(dim=1)  # (B, hidden)
        return self.fc(self.norm2(pooled))


def train_and_evaluate(model, train_loader, val_loader, epochs=30):
    """Train and evaluate model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            obs = batch['observation']
            goal = batch['goal']
            lang = batch['language']
            action = batch['action'][:, -1]  # Predict final action
            
            optimizer.zero_grad()
            if isinstance(model, AttentionModel):
                pred = model(obs, goal, lang)
            else:
                # For baseline, use final observation
                pred = model(obs[:, -1], goal, lang)
            
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                goal = batch['goal']
                lang = batch['language']
                action = batch['action'][:, -1]
                
                if isinstance(model, AttentionModel):
                    pred = model(obs, goal, lang)
                else:
                    pred = model(obs[:, -1], goal, lang)
                
                val_losses.append(criterion(pred, action).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


def run_experiment():
    """Run H3.97 experiment."""
    print("=" * 70)
    print("H3.97: Endpoint Goal on 150+ Step Sequences")
    print("=" * 70)
    
    results = {}
    
    # Test different sequence lengths
    seq_lengths = [150, 175, 200, 225, 250]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing {seq_len} steps ---")
        
        # Create datasets
        train_data = TrajectoryDataset(n_samples=200, seq_len=seq_len, include_goal=True, goal_type='endpoint')
        val_data = TrajectoryDataset(n_samples=50, seq_len=seq_len, include_goal=True, goal_type='endpoint')
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        
        # Baseline (concatenation)
        baseline = BaselineConcat()
        base_loss = train_and_evaluate(baseline, train_loader, val_loader)
        
        # Attention with goal
        attention = AttentionModel()
        attn_loss = train_and_evaluate(attention, train_loader, val_loader)
        
        # Calculate improvement
        delta = (base_loss - attn_loss) / base_loss * 100
        
        results[seq_len] = {
            'baseline_mse': float(base_loss),
            'attention_mse': float(attn_loss),
            'delta_percent': float(delta),
            'attention_wins': attn_loss < base_loss
        }
        
        print(f"  Baseline MSE: {base_loss:.6f}")
        print(f"  Attention MSE: {attn_loss:.6f}")
        print(f"  Delta: {delta:+.1f}%")
    
    # Summary
    avg_delta = np.mean([r['delta_percent'] for r in results.values()])
    attn_wins = sum(1 for r in results.values() if r['attention_wins'])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average Delta: {avg_delta:+.1f}%")
    print(f"Attention Wins: {attn_wins}/{len(seq_lengths)}")
    
    if avg_delta > 0 and attn_wins >= len(seq_lengths) // 2:
        status = "SUPPORTED"
    elif avg_delta < 0 and attn_wins <= len(seq_lengths) // 2:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    
    print(f"Status: {status}")
    
    # Final results
    final_results = {
        'experiment': 'H3.97',
        'hypothesis': 'Endpoint goal enables attention on 150+ step sequences',
        'avg_delta': float(avg_delta),
        'attn_wins': attn_wins,
        'total_tests': len(seq_lengths),
        'status': status,
        'per_length': results
    }
    
    print("\n" + json.dumps(final_results, indent=2))
    
    # Save results
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.97-endpoint-goal-150step/results/metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


if __name__ == "__main__":
    run_experiment()