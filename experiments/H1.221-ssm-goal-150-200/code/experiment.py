#!/usr/bin/env python3
"""
H1.221: SSM + HierGoals + Goal Conditioning on 150-200 Steps

Based on H3.115 showing attention works at 25,35 steps with goal conditioning,
test if SSM can be similarly enabled with goal conditioning on 150-200 step sequences.

Hypothesis: Goal conditioning enables SSM on 150-200 step sequences
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset


class RobotDatasetWithGoal(Dataset):
    """Robot data with goal state information embedded."""
    
    def __init__(self, n_samples=300, seq_len=175, autocorrelation=0.85):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rho = autocorrelation
        np.random.seed(42)
        
        self.data = []
        for _ in range(n_samples):
            # Generate smooth trajectory with autocorrelation (robot-like)
            base = np.random.randn(seq_len, 8) * 0.5
            smooth = np.zeros_like(base)
            smooth[0] = base[0]
            for t in range(1, seq_len):
                smooth[t] = self.rho * smooth[t-1] + np.sqrt(1-self.rho**2) * base[t]
            
            # Add goal state to language representation (key insight from H3.115)
            goal_pos = smooth[-1, :3]  # Final position as goal
            goal_repr = np.tile(np.concatenate([goal_pos, np.zeros(4)]), (seq_len, 1))
            
            obs = smooth + np.random.randn(seq_len, 8) * 0.05
            act = np.gradient(smooth, axis=0)[:, :7] * 0.5 + np.random.randn(seq_len, 7) * 0.02
            
            self.data.append({
                'observation': obs.astype(np.float32),
                'action': act.astype(np.float32),
                'language': goal_repr.astype(np.float32),
                'goal': goal_pos.astype(np.float32),
            })
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class SSMWithGoal(nn.Module):
    """SSM with goal conditioning."""
    
    def __init__(self, obs_dim=8, lang_dim=7, action_dim=7, hidden=128, state_dim=64):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        self.hidden_dim = hidden
        self.state_dim = state_dim
        
        # Goal projection (from goal state)
        self.goal_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, hidden)
        )
        
        # SSM state transition
        self.ssm_A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.ssm_B = nn.Linear(hidden, state_dim, bias=False)
        self.ssm_C = nn.Linear(state_dim, hidden, bias=False)
        
        # Fusion with goal
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),  # obs + lang + goal
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, action_dim)
        
    def forward(self, obs, lang, goal=None):
        B, T, _ = obs.shape
        
        # Project inputs
        x = self.obs_proj(obs) + self.lang_proj(lang)
        
        # Goal conditioning
        if goal is not None:
            goal_signal = self.goal_proj(goal)
        else:
            goal_signal = self.goal_proj(lang[:, 0, :3])
        
        # SSM processing
        hidden = torch.zeros(B, self.state_dim, device=obs.device)
        outputs = []
        
        for t in range(T):
            # Goal-conditioned SSM update
            hidden = torch.sigmoid(self.ssm_A @ hidden.T + self.ssm_B(x[:, t]).T).T
            h = self.ssm_C(hidden)
            
            # Fuse with goal
            fused = torch.cat([x[:, t], h, goal_signal], dim=-1)
            out_t = self.out(self.fusion(fused))
            outputs.append(out_t)
        
        return torch.stack(outputs, dim=1)


class HierGoalsSSMWithGoal(nn.Module):
    """Hierarchical SSM with goal conditioning."""
    
    def __init__(self, obs_dim=8, lang_dim=7, action_dim=7, hidden=128, state_dim=64, n_levels=3):
        super().__init__()
        self.hidden_dim = hidden
        self.state_dim = state_dim
        self.n_levels = n_levels
        
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        
        # Goal projection
        self.goal_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, hidden)
        )
        
        # Goal extractors at multiple levels
        self.goal_extractors = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 32), nn.Tanh()) for _ in range(n_levels)
        ])
        
        # SSM state transition
        self.ssm_A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.ssm_B = nn.Linear(hidden, state_dim, bias=False)
        self.ssm_C = nn.Linear(state_dim, hidden, bias=False)
        
        # Fusion with hierarchical goals
        self.hier_fusion = nn.Sequential(
            nn.Linear(hidden * 2 + 64, hidden),  # obs + SSM + hierarchical goals
            nn.LayerNorm(hidden),
            nn.GELU()
        )
        
        self.out = nn.Linear(hidden, action_dim)
        
    def forward(self, obs, lang, goal=None):
        B, T, _ = obs.shape
        
        # Project inputs
        x = self.obs_proj(obs) + self.lang_proj(lang)
        
        # Goal conditioning
        if goal is not None:
            goal_signal = self.goal_proj(goal)
        else:
            goal_signal = self.goal_proj(lang[:, 0, :3])
        
        # Extract hierarchical goals
        goals = [goal_signal]
        for i in range(self.n_levels - 1):
            chunk_size = max(1, T // (2 ** (i + 1)))
            n_chunks = T // chunk_size
            if n_chunks > 0:
                chunked = x[:, :n_chunks * chunk_size].reshape(B, n_chunks, chunk_size, -1)
                pooled = chunked.mean(dim=2)
                goals.append(self.goal_extractors[i](pooled.mean(dim=1)))
        
        # Combine all goal signals - project to hidden dim first
        goal_combined = goals[0].clone()  # Start with main goal signal
        for g in goals[1:]:
            goal_combined = goal_combined + g  # Sum goal signals
        
        # SSM processing
        hidden = torch.zeros(B, self.state_dim, device=obs.device)
        outputs = []
        
        for t in range(T):
            x_t = x[:, t]
            hidden = torch.sigmoid(self.ssm_A @ hidden.T + self.ssm_B(x_t).T).T
            h = self.ssm_C(hidden)
            
            # Fuse with goal combined
            fused = torch.cat([x_t, h, goal_combined], dim=-1)
            out_t = self.out(self.hier_fusion(fused))
            outputs.append(out_t)
        
        return torch.stack(outputs, dim=1)


class ConcatBaseline(nn.Module):
    """Simple concatenation baseline."""
    
    def __init__(self, obs_dim=8, lang_dim=7, action_dim=7, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim)
        )
        
    def forward(self, obs, lang):
        x = torch.cat([obs, lang], dim=-1)
        return self.net(x)


def train_model(model, train_loader, epochs=30, has_goal=False):
    """Train model and return final loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        for batch in train_loader:
            opt.zero_grad()
            if has_goal:
                pred = model(batch['observation'], batch['language'], batch.get('goal'))
            else:
                pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in train_loader:
            if has_goal:
                pred = model(batch['observation'], batch['language'], batch.get('goal'))
            else:
                pred = model(batch['observation'], batch['language'])
            total_loss += crit(pred, batch['action']).item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    print("=" * 70)
    print("H1.221: SSM + HierGoals + Goal Conditioning on 150-200 Steps")
    print("=" * 70)
    
    results = {
        'experiment': 'H1.221',
        'description': 'SSM + HierGoals with goal conditioning on 150-200 step sequences',
        'results': []
    }
    
    seq_lengths = [150, 160, 175, 185, 200]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing seq_len={seq_len} ---")
        
        train_data = RobotDatasetWithGoal(n_samples=300, seq_len=seq_len, autocorrelation=0.85)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        
        concat_loss = train_model(ConcatBaseline(), train_loader, epochs=30, has_goal=False)
        ssm_goal_loss = train_model(SSMWithGoal(), train_loader, epochs=30, has_goal=True)
        hier_loss = train_model(HierGoalsSSMWithGoal(), train_loader, epochs=30, has_goal=True)
        
        ssm_goal_delta = (concat_loss - ssm_goal_loss) / concat_loss * 100
        hier_delta = (concat_loss - hier_loss) / concat_loss * 100
        
        print(f"  Concat:      {concat_loss:.6f} (baseline)")
        print(f"  SSM+Goal:    {ssm_goal_loss:.6f} ({ssm_goal_delta:+.1f}%)")
        print(f"  Hier+Goal:   {hier_loss:.6f} ({hier_delta:+.1f}%)")
        
        results['results'].append({
            'seq_len': seq_len,
            'concat_loss': float(concat_loss),
            'ssm_goal_loss': float(ssm_goal_loss),
            'hier_goal_loss': float(hier_loss),
            'ssm_goal_delta': float(ssm_goal_delta),
            'hier_delta': float(hier_delta),
            'ssm_goal_wins': ssm_goal_loss < concat_loss,
            'hier_wins': hier_loss < concat_loss
        })
    
    # Summary
    avg_ssm_goal_delta = np.mean([r['ssm_goal_delta'] for r in results['results']])
    avg_hier_delta = np.mean([r['hier_delta'] for r in results['results']])
    ssm_goal_wins = sum(1 for r in results['results'] if r['ssm_goal_wins'])
    hier_wins = sum(1 for r in results['results'] if r['hier_wins'])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"SSM+Goal wins: {ssm_goal_wins}/{len(seq_lengths)} lengths (avg {avg_ssm_goal_delta:+.1f}%)")
    print(f"Hier+Goal wins: {hier_wins}/{len(seq_lengths)} lengths (avg {avg_hier_delta:+.1f}%)")
    
    if avg_ssm_goal_delta > 0 and ssm_goal_wins >= len(seq_lengths) * 0.6:
        status = "SUPPORTED"
        note = "Goal conditioning enables SSM on 150-200 step sequences"
    elif avg_hier_delta > 0 and hier_wins >= len(seq_lengths) * 0.6:
        status = "PARTIAL"
        note = "Hierarchical goal helps, SSM works with conditioning"
    else:
        status = "REFUTED"
        note = "Goal conditioning does NOT enable SSM on 150-200 steps"
    
    results['summary'] = {
        'status': status,
        'note': note,
        'avg_ssm_goal_delta': float(avg_ssm_goal_delta),
        'avg_hier_delta': float(avg_hier_delta),
        'ssm_goal_wins': ssm_goal_wins,
        'hier_wins': hier_wins
    }
    
    print(f"\nStatus: {status}")
    print(f"Note: {note}")
    
    with open('/tmp/H1.221_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults: {json.dumps(results, indent=2)}")
    
    return results


if __name__ == "__main__":
    main()