#!/usr/bin/env python3
"""
H3.115: Attention on 20-40 Step Sequences WITH Task Structure

Based on H3.91's success (+86.6%), test if attention can work
on shorter sequences (20-40) when given proper task structure.
This follows H1.202's finding that goal states are the key enabler.

Hypothesis: Attention will win on 20-40 steps WITH goal conditioning
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
    """Robot data WITH task structure: goal states, action outcomes."""
    
    def __init__(self, n_samples=300, seq_len=30):
        self.n_samples = n_samples
        self.seq_len = seq_len
        np.random.seed(42)
        
        self.data = []
        for _ in range(n_samples):
            # Generate manipulation trajectory with goal state
            # Robot picks object, moves it, places at goal location
            start_pos = np.random.randn(3) * 0.3
            goal_pos = np.random.randn(3) * 0.3
            current_pos = start_pos + np.random.randn(3) * 0.02
            
            observations = []
            actions = []
            
            for t in range(seq_len):
                # Phase 1: reach (0-30%), Phase 2: grasp/move (30-70%), Phase 3: place (70-100%)
                if t < seq_len * 0.3:
                    phase = 'reach'
                    target = start_pos + (goal_pos - start_pos) * (t / (seq_len * 0.3))
                    action_scale = 0.5
                elif t < seq_len * 0.7:
                    phase = 'move'
                    frac = (t - seq_len * 0.3) / (seq_len * 0.4)
                    target = start_pos + (goal_pos - start_pos) * frac
                    action_scale = 0.3
                else:
                    phase = 'place'
                    target = goal_pos
                    action_scale = 0.1
                
                # Current position follows target with smooth dynamics
                current_pos = current_pos * 0.9 + target * 0.1 + np.random.randn(3) * 0.01
                
                obs = np.concatenate([
                    current_pos,  # 3D position
                    np.random.randn(4) * 0.1,  # orientation
                    np.array([1.0 if phase == 'reach' else 0.0]),  # gripper state
                ]).astype(np.float32)
                
                # Action is delta to target (3D) + gripper (1D) + noise (3D)
                delta = (target - current_pos) * action_scale
                action = np.concatenate([delta, np.array([action_scale * 0.5]), np.random.randn(3) * 0.01])
                
                observations.append(obs)
                actions.append(action)
            
            observations = np.stack(observations).astype(np.float32)
            actions = np.stack(actions).astype(np.float32)
            
            # Include goal state in language representation (key finding from H1.202, H3.92)
            goal_repr = np.concatenate([goal_pos, np.zeros(4)])  # Goal position + zeros
            lang_base = np.tile(goal_repr, (seq_len, 1))
            # Add temporal variation
            lang_base[:, :3] = goal_pos + np.sin(np.linspace(0, np.pi, seq_len))[:, None] * 0.1
            
            self.data.append({
                'observation': observations,
                'action': actions,
                'language': lang_base.astype(np.float32),
                'goal': goal_pos.astype(np.float32),
            })
        
        self.data = self.data  # Ensure we store data properly
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class AttentionModel(nn.Module):
    """Attention-based model with goal conditioning."""
    
    def __init__(self, obs_dim=8, lang_dim=7, action_dim=7, hidden=128, n_heads=4):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        self.hidden = hidden
        
        # Self-attention for sequence modeling
        self.self_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 4, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)
        
        # Goal conditioning layer
        self.goal_proj = nn.Linear(3, hidden)  # Goal position only
        
        self.out = nn.Linear(hidden, action_dim)
        
    def forward(self, obs, lang):
        # obs: (B, T, 8), lang: (B, T, 7) - goal-augmented
        B, T, _ = obs.shape
        
        # Project inputs
        x = self.obs_proj(obs) + self.lang_proj(lang)
        
        # Extract goal from first language token
        goal = lang[:, 0, :3]  # First 3 dims are goal position
        goal_signal = self.goal_proj(goal).unsqueeze(1)  # (B, 1, hidden)
        
        # Self-attention with goal conditioning
        # Add goal as first "token"
        x_goal = torch.cat([goal_signal, x], dim=1)  # (B, T+1, hidden)
        
        # Attend
        attn_out, _ = self.self_attn(x_goal, x_goal, x_goal)
        x = self.norm1(x_goal + attn_out)
        x = self.norm2(x + self.ff(x))
        
        # Remove goal token, predict actions
        x = x[:, 1:, :]  # (B, T, hidden)
        
        return self.out(x)


class SSMModel(nn.Module):
    """SSM baseline for comparison."""
    
    def __init__(self, obs_dim=8, lang_dim=7, action_dim=7, hidden=128):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        self.state_dim = 64
        
        self.ssm_A = nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.01)
        self.ssm_B = nn.Linear(hidden, self.state_dim, bias=False)
        self.ssm_C = nn.Linear(self.state_dim, hidden, bias=False)
        
        self.out = nn.Linear(hidden, action_dim)
        
    def forward(self, obs, lang):
        B, T, _ = obs.shape
        hidden = torch.zeros(B, self.state_dim, device=obs.device)
        
        x = self.obs_proj(obs) + self.lang_proj(lang)
        outputs = []
        
        for t in range(T):
            hidden = torch.sigmoid(self.ssm_A @ hidden.T + self.ssm_B(x[:, t]).T).T
            h = self.ssm_C(hidden)
            out = self.out(x[:, t] + h)
            outputs.append(out)
        
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


def train_model(model, train_loader, epochs=30):
    """Train model and return final loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        for batch in train_loader:
            opt.zero_grad()
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
            pred = model(batch['observation'], batch['language'])
            total_loss += crit(pred, batch['action']).item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    print("=" * 70)
    print("H3.115: Attention on 20-40 Steps WITH Task Structure (Goal Conditioning)")
    print("=" * 70)
    
    results = {
        'experiment': 'H3.115',
        'description': 'Attention on 20-40 step sequences WITH goal states',
        'results': []
    }
    
    seq_lengths = [20, 25, 30, 35, 40]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing seq_len={seq_len} ---")
        
        train_data = RobotDatasetWithGoal(n_samples=300, seq_len=seq_len)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        
        concat_loss = train_model(ConcatBaseline(), train_loader, epochs=30)
        ssm_loss = train_model(SSMModel(), train_loader, epochs=30)
        attn_loss = train_model(AttentionModel(), train_loader, epochs=30)
        
        concat_delta = 0
        ssm_delta = (concat_loss - ssm_loss) / concat_loss * 100
        attn_delta = (concat_loss - attn_loss) / concat_loss * 100
        
        print(f"  Concat:  {concat_loss:.6f} (baseline)")
        print(f"  SSM:     {ssm_loss:.6f} ({ssm_delta:+.1f}%)")
        print(f"  Attn:    {attn_loss:.6f} ({attn_delta:+.1f}%)")
        
        results['results'].append({
            'seq_len': seq_len,
            'concat_loss': float(concat_loss),
            'ssm_loss': float(ssm_loss),
            'attn_loss': float(attn_loss),
            'ssm_delta': float(ssm_delta),
            'attn_delta': float(attn_delta),
            'ssm_wins': ssm_loss < concat_loss,
            'attn_wins': attn_loss < concat_loss
        })
    
    # Summary
    avg_attn_delta = np.mean([r['attn_delta'] for r in results['results']])
    avg_ssm_delta = np.mean([r['ssm_delta'] for r in results['results']])
    attn_wins = sum(1 for r in results['results'] if r['attn_wins'])
    ssm_wins = sum(1 for r in results['results'] if r['ssm_wins'])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Attention wins: {attn_wins}/{len(seq_lengths)} lengths (avg {avg_attn_delta:+.1f}%)")
    print(f"SSM wins: {ssm_wins}/{len(seq_lengths)} lengths (avg {avg_ssm_delta:+.1f}%)")
    
    if avg_attn_delta > 0 and attn_wins >= len(seq_lengths) * 0.6:
        status = "SUPPORTED"
        note = "Attention with goal conditioning dominates on 20-40 steps"
    elif avg_ssm_delta > 0 and ssm_wins >= len(seq_lengths) * 0.6:
        status = "PARTIAL (SSM)"
        note = "SSM dominates, attention overhead not justified"
    else:
        status = "REFUTED"
        note = "Concat/SSM win on 20-40 steps"
    
    results['summary'] = {
        'status': status,
        'note': note,
        'avg_attn_delta': float(avg_attn_delta),
        'avg_ssm_delta': float(avg_ssm_delta),
        'attn_wins': attn_wins,
        'ssm_wins': ssm_wins
    }
    
    print(f"\nStatus: {status}")
    print(f"Note: {note}")
    
    with open('/tmp/H3.115_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults: {json.dumps(results, indent=2)}")
    
    return results


if __name__ == "__main__":
    main()