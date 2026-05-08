#!/usr/bin/env python3
"""
H1.150: Attention on 200-250 step ultra-extreme multi-step tasks
Based on H1.149 success (+90.7% on 150-200 steps), test even longer sequences
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset

class UltraLongSequenceDataset(Dataset):
    """Dataset for ultra-long sequence (200-250 steps) manipulation tasks."""
    
    def __init__(self, n_samples=200, min_seq_length=200, max_seq_length=250, obs_dim=8, action_dim=7):
        self.n_samples = n_samples
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Generate synthetic manipulation data with temporal structure
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.observations = []
        self.actions = []
        
        for _ in range(n_samples):
            # Create realistic manipulation trajectory with uniform distribution
            seq_len = np.random.randint(min_seq_length, max_seq_length + 1)
            
            # State: object positions, gripper state, velocities
            obs = np.random.randn(seq_len, obs_dim) * 0.1
            
            # Add temporal structure: smooth motion
            for i in range(1, seq_len):
                obs[i] = obs[i-1] * 0.95 + obs[i] * 0.05
            
            # Actions: joint positions, gripper
            actions = np.random.randn(seq_len, action_dim) * 0.1
            
            # Add temporal structure to actions
            for i in range(1, seq_len):
                actions[i] = actions[i-1] * 0.9 + actions[i] * 0.1
            
            self.observations.append(torch.FloatTensor(obs))
            self.actions.append(torch.FloatTensor(actions))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'observation': self.observations[idx],
            'action': self.actions[idx]
        }


class BaselineConcat(nn.Module):
    """Baseline: Concatenation fusion."""
    def __init__(self, obs_dim=8, action_dim=7, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, obs_dim)
        # Take last timestep
        h = self.encoder(x[:, -1, :])
        return self.decoder(h)


class AttentionFusion(nn.Module):
    """Attention-based fusion for long sequences."""
    def __init__(self, obs_dim=8, action_dim=7, hidden_dim=256, num_heads=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, obs_dim)
        h = self.encoder(x)  # (batch, seq_len, hidden_dim)
        
        # Apply attention
        attn_out, _ = self.attention(h, h, h)
        
        # Take last timestep
        out = self.decoder(attn_out[:, -1, :])
        return out


class ActionGatedAttention(nn.Module):
    """Action-conditioned attention (from H1.39)."""
    def __init__(self, obs_dim=8, action_dim=7, hidden_dim=256, num_heads=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x, action_context=None):
        h = self.encoder(x)
        
        if action_context is not None:
            # Modulate with action context
            action_mod = self.action_proj(action_context)
            h = h + action_mod.unsqueeze(1)
        
        attn_out, _ = self.attention(h, h, h)
        out = self.decoder(attn_out[:, -1, :])
        return out


def train_model(model, train_loader, val_loader, epochs=30):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            obs = batch[0]  # TensorDataset returns (obs, action)
            action = batch[1]
            
            # Get last action as target
            target = action[:, -1, :]
            
            pred = model(obs)
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs = batch[0]
                action = batch[1]
                target = action[:, -1, :]
                pred = model(obs)
                val_losses.append(criterion(pred, target).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    max_len = max(b['observation'].shape[0] for b in batch)
    
    observations = []
    actions = []
    
    for b in batch:
        obs = b['observation']
        act = b['action']
        
        # Pad sequences
        obs_pad = F.pad(obs, (0, 0, 0, max_len - obs.shape[0]))
        act_pad = F.pad(act, (0, 0, 0, max_len - act.shape[0]))
        
        observations.append(obs_pad)
        actions.append(act_pad)
    
    return {
        'observation': torch.stack(observations),
        'action': torch.stack(actions)
    }


def main():
    print("=" * 70)
    print("H1.150: Attention on 200-250 step ultra-extreme multi-step tasks")
    print("=" * 70)
    
    # Create datasets
    print("\nGenerating datasets...")
    train_dataset = UltraLongSequenceDataset(n_samples=300)
    val_dataset = UltraLongSequenceDataset(n_samples=100)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    
    # Test different sequence lengths
    results = {}
    
    for seq_len in [200, 225, 250]:
        print(f"\n--- Testing sequence length: {seq_len} ---")
        
        # Filter data by sequence length (within +/- 5 of target)
        train_filtered = [d for d in train_dataset if abs(d['observation'].shape[0] - seq_len) <= 5]
        val_filtered = [d for d in val_dataset if abs(d['observation'].shape[0] - seq_len) <= 5]
        
        if len(train_filtered) < 10:
            print(f"  Skipping {seq_len} - insufficient samples ({len(train_filtered)})")
            continue
        
        # Pad sequences to same length
        max_len = max(d['observation'].shape[0] for d in train_filtered + val_filtered)
        
        def pad_tensor(t, max_len):
            if t.shape[0] < max_len:
                return F.pad(t, (0, 0, 0, max_len - t.shape[0]))
            return t
        
        train_obs = torch.stack([pad_tensor(d['observation'], max_len) for d in train_filtered])
        train_act = torch.stack([pad_tensor(d['action'], max_len) for d in train_filtered])
        val_obs = torch.stack([pad_tensor(d['observation'], max_len) for d in val_filtered])
        val_act = torch.stack([pad_tensor(d['action'], max_len) for d in val_filtered])
        
        train_ds = torch.utils.data.TensorDataset(train_obs, train_act)
        val_ds = torch.utils.data.TensorDataset(val_obs, val_act)
        
        train_ld = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=16)
        
        # Baseline (Concatenation)
        print(f"  Training Baseline (Concat)...")
        baseline = BaselineConcat()
        baseline_loss = train_model(baseline, train_ld, val_ld)
        
        # Standard Attention
        print(f"  Training Attention...")
        attention = AttentionFusion()
        attention_loss = train_model(attention, train_ld, val_ld)
        
        # Action-Gated Attention
        print(f"  Training Action-Gated Attention...")
        action_gated = ActionGatedAttention()
        action_gated_loss = train_model(action_gated, train_ld, val_ld)
        
        # Calculate improvements
        baseline_to_attn = (baseline_loss - attention_loss) / baseline_loss * 100
        baseline_to_action = (baseline_loss - action_gated_loss) / baseline_loss * 100
        
        results[seq_len] = {
            'baseline': baseline_loss,
            'attention': attention_loss,
            'action_gated': action_gated_loss,
            'attn_improvement': baseline_to_attn,
            'action_improvement': baseline_to_action
        }
        
        print(f"  Results:")
        print(f"    Baseline (Concat): {baseline_loss:.6f}")
        print(f"    Attention: {attention_loss:.6f} ({baseline_to_attn:+.1f}%)")
        print(f"    Action-Gated: {action_gated_loss:.6f} ({baseline_to_action:+.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_baseline = np.mean([r['baseline'] for r in results.values()])
    avg_attention = np.mean([r['attention'] for r in results.values()])
    avg_action = np.mean([r['action_gated'] for r in results.values()])
    
    avg_attn_improvement = (avg_baseline - avg_attention) / avg_baseline * 100
    avg_action_improvement = (avg_baseline - avg_action) / avg_baseline * 100
    
    print(f"Average Baseline: {avg_baseline:.6f}")
    print(f"Average Attention: {avg_attention:.6f} ({avg_attn_improvement:+.1f}%)")
    print(f"Average Action-Gated: {avg_action:.6f} ({avg_action_improvement:+.1f}%)")
    
    # Determine status
    if avg_attn_improvement > 50:
        status = "SUPPORTED"
    elif avg_attn_improvement > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"\nStatus: {status}")
    
    # Save results
    output = {
        'hypothesis': 'H1.150',
        'statement': 'Attention maintains advantage on 200-250 step ultra-extreme multi-step tasks',
        'status': status,
        'results': results,
        'avg_baseline': avg_baseline,
        'avg_attention': avg_attention,
        'avg_action_gated': avg_action,
        'avg_attn_improvement': avg_attn_improvement,
        'avg_action_improvement': avg_action_improvement
    }
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.150-attention-200-250-steps/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results.json")
    
    return output


if __name__ == "__main__":
    main()