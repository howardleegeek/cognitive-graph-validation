#!/usr/bin/env python3
"""
H1.364 / H3.364: Simpler Multi-Step + Goal-Conditioned Attention
Building on:
- H1.351: Simple multi-step (5-10 steps) CG +32.4%
- H3.363: Attention on 200-250 steps +11.0%

Tests:
1. CG on 10-20 step simpler multi-step (building on H1.351 success)
2. Attention with goal conditioning on 250-300 step sequences (extending H3.363)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from data_loader import prepare_datasets


class SimpleMultiStepDataset(Dataset):
    """Dataset with simpler multi-step tasks (10-20 steps, 3-5 sub-tasks)"""
    def __init__(self, base_dataset, seq_len=15, n_steps=5):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.n_steps = n_steps
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        seq_obs = []
        seq_lang = []
        seq_action = []
        
        base_obs = item['observation'].numpy().astype(np.float32)
        base_action = item['action'].numpy().astype(np.float32)
        
        for t in range(self.seq_len):
            # Simpler multi-step: fewer sub-tasks within sequence
            step_progress = (t * self.n_steps) / self.seq_len
            current_step = int(step_progress)
            step_phase = step_progress - current_step
            
            # Simpler dynamics - just 2-3 patterns
            if current_step % 2 == 0:
                obs_t = base_obs + np.sin(step_phase * np.pi) * 0.08
            else:
                obs_t = base_obs + np.cos(step_phase * np.pi) * 0.06
            
            obs_t += np.random.randn(*base_obs.shape).astype(np.float32) * 0.01
            
            # Simpler action evolution
            action_t = base_action + np.sin(t * 0.15) * 0.05
            
            seq_obs.append(torch.FloatTensor(obs_t))
            seq_lang.append(item['language'])
            seq_action.append(torch.FloatTensor(action_t))
        
        obs_seq = torch.stack(seq_obs)
        lang_seq = torch.stack([torch.FloatTensor(l) if isinstance(l, np.ndarray) else l for l in seq_lang])
        action_seq = torch.stack(seq_action)
        
        return {
            'observation': obs_seq,
            'language': lang_seq,
            'action': action_seq,
            'seq_len': self.seq_len,
            'n_steps': self.n_steps
        }


class GoalConditionedLongSequenceDataset(Dataset):
    """Dataset with goal-conditioned very long sequences (250-300 timesteps)"""
    def __init__(self, base_dataset, seq_len=275):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        seq_obs = []
        seq_lang = []
        seq_action = []
        goal_obs = []
        
        base_obs = item['observation'].numpy().astype(np.float32)
        base_action = item['action'].numpy().astype(np.float32)
        
        # Goal state (final observation)
        goal_state = base_obs + np.random.randn(*base_obs.shape).astype(np.float32) * 0.05
        
        for t in range(self.seq_len):
            # Temporal structure with goal conditioning
            progress = t / self.seq_len
            phase = (t % 25) / 25.0
            
            # Interpolate towards goal
            obs_t = base_obs * (1 - progress) + goal_state * progress
            obs_t += np.sin(phase * np.pi) * 0.08 + np.random.randn(*base_obs.shape).astype(np.float32) * 0.01
            
            # Actions evolve with goal direction
            action_t = base_action + np.sin(t * 0.1) * 0.06
            
            seq_obs.append(torch.FloatTensor(obs_t))
            seq_lang.append(item['language'])
            seq_action.append(torch.FloatTensor(action_t))
            goal_obs.append(torch.FloatTensor(goal_state))
        
        obs_seq = torch.stack(seq_obs)
        lang_seq = torch.stack([torch.FloatTensor(l) if isinstance(l, np.ndarray) else l for l in seq_lang])
        action_seq = torch.stack(seq_action)
        goal_seq = torch.stack(goal_obs)
        
        return {
            'observation': obs_seq,
            'language': lang_seq,
            'action': action_seq,
            'goal': goal_seq,
            'seq_len': self.seq_len
        }


class BaselineModel(nn.Module):
    """Simple baseline for comparison"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq.mean(dim=1))
        lang_enc = self.lang_encoder(lang_seq.mean(dim=1))
        return self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))


class CognitiveGraphModel(nn.Module):
    """Cognitive Graph model with unified representation"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=256):
        super().__init__()
        physical_dim = int(latent_dim * 0.22)
        semantic_dim = latent_dim - physical_dim
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq.mean(dim=1))
        lang_enc = self.lang_encoder(lang_seq.mean(dim=1))
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        return self.fusion(combined)


class AttentionModel(nn.Module):
    """Attention model for long sequences"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=256, n_heads=8):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq):
        obs_enc = self.obs_encoder(obs_seq)
        lang_enc = self.lang_encoder(lang_seq)
        
        attn_out, _ = self.cross_attn(obs_enc, lang_enc, lang_enc)
        
        obs_pooled = attn_out.mean(dim=1)
        lang_pooled = lang_enc.mean(dim=1)
        
        return self.fusion(torch.cat([obs_pooled, lang_pooled], dim=-1))


class GoalConditionedAttentionModel(nn.Module):
    """Attention model with goal conditioning for long sequences"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=256, n_heads=8):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs_seq, lang_seq, goal_seq):
        obs_enc = self.obs_encoder(obs_seq)
        lang_enc = self.lang_encoder(lang_seq)
        goal_enc = self.goal_encoder(goal_seq)
        
        # Attend to language and goal
        attn_out, _ = self.cross_attn(obs_enc, torch.cat([lang_enc, goal_enc], dim=1), torch.cat([lang_enc, goal_enc], dim=1))
        
        obs_pooled = attn_out.mean(dim=1)
        lang_pooled = lang_enc.mean(dim=1)
        goal_pooled = goal_enc.mean(dim=1)
        
        return self.fusion(torch.cat([obs_pooled, lang_pooled, goal_pooled], dim=-1))


def train_and_eval(model, train_loader, val_loader, epochs=30, use_goal=False):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            if use_goal:
                pred = model(batch['observation'], batch['language'], batch['goal'])
            else:
                pred = model(batch['observation'], batch['language'])
            target = batch['action'].mean(dim=1)
            loss = crit(pred, target)
            loss.backward()
            opt.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if use_goal:
                    pred = model(batch['observation'], batch['language'], batch['goal'])
                else:
                    pred = model(batch['observation'], batch['language'])
                target = batch['action'].mean(dim=1)
                val_losses.append(crit(pred, target).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


print("Loading data...")
train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)

# Part 1: Simpler multi-step (10-20 steps, 3-5 sub-tasks)
print("\n" + "="*60)
print("PART 1: Simpler Multi-Step (10-20 steps, 3-5 sub-tasks)")
print("="*60)

multi_step_results = {}
for seq_len, n_steps in [(10, 3), (15, 5), (20, 5)]:
    print(f"\n--- Testing {seq_len}-step, {n_steps} sub-tasks ---")
    train_seq = SimpleMultiStepDataset(train_data, seq_len=seq_len, n_steps=n_steps)
    val_seq = SimpleMultiStepDataset(val_data, seq_len=seq_len, n_steps=n_steps)
    
    train_loader = DataLoader(train_seq, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_seq, batch_size=16)
    
    print(f"  Training Baseline ({seq_len} steps)...")
    baseline = BaselineModel()
    base_loss = train_and_eval(baseline, train_loader, val_loader)
    
    print(f"  Training Cognitive Graph ({seq_len} steps)...")
    cg = CognitiveGraphModel()
    cg_loss = train_and_eval(cg, train_loader, val_loader)
    
    cg_improvement = (base_loss - cg_loss) / base_loss * 100
    
    multi_step_results[f"{seq_len}_{n_steps}"] = {
        'baseline_loss': float(base_loss),
        'cg_loss': float(cg_loss),
        'cg_improvement': float(cg_improvement)
    }
    
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  CG: {cg_loss:.6f} ({cg_improvement:+.1f}%)")

avg_cg_improvement = np.mean([r['cg_improvement'] for r in multi_step_results.values()])
print(f"\n  Average CG improvement: {avg_cg_improvement:+.1f}%")

# Part 2: Goal-conditioned attention (250-300 steps)
print("\n" + "="*60)
print("PART 2: Goal-Conditioned Attention (250-300 steps)")
print("="*60)

attention_results = {}
for seq_len in [250, 275, 300]:
    print(f"\n--- Testing {seq_len}-step sequences with goal conditioning ---")
    train_seq = GoalConditionedLongSequenceDataset(train_data, seq_len=seq_len)
    val_seq = GoalConditionedLongSequenceDataset(val_data, seq_len=seq_len)
    
    train_loader = DataLoader(train_seq, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_seq, batch_size=16)
    
    print(f"  Training Baseline ({seq_len} steps)...")
    baseline = BaselineModel()
    base_loss = train_and_eval(baseline, train_loader, val_loader)
    
    print(f"  Training Standard Attention ({seq_len} steps)...")
    attn = AttentionModel()
    attn_loss = train_and_eval(attn, train_loader, val_loader)
    
    print(f"  Training Goal-Conditioned Attention ({seq_len} steps)...")
    goal_attn = GoalConditionedAttentionModel()
    goal_loss = train_and_eval(goal_attn, train_loader, val_loader, use_goal=True)
    
    attn_improvement = (base_loss - attn_loss) / base_loss * 100
    goal_improvement = (base_loss - goal_loss) / base_loss * 100
    
    best_improvement = max(attn_improvement, goal_improvement)
    best_type = "goal" if goal_improvement > attn_improvement else "standard"
    
    attention_results[seq_len] = {
        'baseline_loss': float(base_loss),
        'attn_loss': float(attn_loss),
        'goal_attn_loss': float(goal_loss),
        'attn_improvement': float(attn_improvement),
        'goal_improvement': float(goal_improvement),
        'best_improvement': float(best_improvement),
        'best_type': best_type
    }
    
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  Standard Attn: {attn_loss:.6f} ({attn_improvement:+.1f}%)")
    print(f"  Goal-Cond Attn: {goal_loss:.6f} ({goal_improvement:+.1f}%)")
    print(f"  Best: {best_type} ({best_improvement:+.1f}%)")

avg_attn_improvement = np.mean([r['best_improvement'] for r in attention_results.values()])
print(f"\n  Average Best Attention improvement: {avg_attn_improvement:+.1f}%")

# Summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Simpler multi-step (10-20 steps) CG improvement: {avg_cg_improvement:+.1f}%")
print(f"Goal-conditioned attention (250-300 steps) improvement: {avg_attn_improvement:+.1f}%")

cg_wins = avg_cg_improvement > 0
attn_wins = avg_attn_improvement > 0

output = {
    'multi_step_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in multi_step_results.items()},
    'attention_results': {k: {kk: float(vv) if isinstance(vv, (int, float, str)) else str(vv) for kk, vv in v.items()} for k, v in attention_results.items()},
    'avg_cg_improvement': float(avg_cg_improvement),
    'avg_attn_improvement': float(avg_attn_improvement),
    'cg_wins': bool(cg_wins),
    'attn_wins': bool(attn_wins),
    'config': {
        'hypothesis': 'H1.364 / H3.364',
        'multi_step_lengths': ['10_3', '15_5', '20_5'],
        'attention_lengths': [250, 275, 300]
    }
}

print(json.dumps(output, indent=2))