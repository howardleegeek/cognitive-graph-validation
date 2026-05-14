#!/usr/bin/env python3
"""
H1.248: Hierarchical Attention on 80-100 Step Sequences

Based on findings:
- H1.247: Hierarchical attention +7.7% on 50-80 steps (extends beyond 45-step boundary)
- H1.246: Task decomposition +4.8% on 50-70 steps
- H3.144: Chunked attention makes things worse (-7.4%)

Hypothesis: Hierarchical attention can extend further beyond 80 steps with 
optimized segment size and regularization.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset


class HierarchicalAttentionModel(nn.Module):
    """Hierarchical attention: segment-level then global attention"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256, 
                 num_heads=8, segment_size=20, reg=0.35):
        super().__init__()
        self.segment_size = segment_size
        self.num_heads = num_heads
        self.reg = reg
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        self.segment_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.segment_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.global_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs, lang):
        batch_size, seq_len, obs_dim = obs.shape
        
        obs_emb = self.obs_encoder(obs)
        lang_emb = self.lang_encoder(lang)
        
        combined = obs_emb + lang_emb
        
        num_segments = (seq_len + self.segment_size - 1) // self.segment_size
        
        segment_reprs = []
        for i in range(num_segments):
            start = i * self.segment_size
            end = min(start + self.segment_size, seq_len)
            segment = combined[:, start:end, :]
            
            if segment.size(1) > 0:
                seg_attn_out, _ = self.segment_attn(segment, segment, segment)
                seg_summary = seg_attn_out.mean(dim=1)
                segment_reprs.append(seg_summary)
        
        if len(segment_reprs) == 0:
            segment_reprs = [combined.mean(dim=1)]
        
        segment_tensor = torch.stack(segment_reprs, dim=1)
        
        global_attn_out, _ = self.global_attn(segment_tensor, segment_tensor, segment_tensor)
        global_summary = global_attn_out.mean(dim=1)
        
        output = self.decoder(global_summary)
        return output


class StandardAttentionModel(nn.Module):
    """Standard attention for comparison"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256, num_heads=8, reg=0.35):
        super().__init__()
        self.reg = reg
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs, lang):
        obs_emb = self.obs_encoder(obs)
        lang_emb = self.lang_encoder(lang)
        
        combined = obs_emb + lang_emb
        
        attn_out, _ = self.attn(combined, combined, combined)
        summary = attn_out.mean(dim=1)
        
        return self.decoder(summary)


class BaselineModel(nn.Module):
    """Baseline concatenation model"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs, lang):
        obs_emb = self.obs_encoder(obs)
        lang_emb = self.lang_encoder(lang)
        
        combined = torch.cat([obs_emb, lang_emb], dim=-1)
        return self.decoder(combined.mean(dim=1))


def create_dataset(seq_len, n_samples, autocorr=0.95):
    """Create synthetic dataset with autocorrelation"""
    observations = []
    languages = []
    actions = []
    
    for _ in range(n_samples):
        base_obs = np.random.randn(seq_len, 8).astype(np.float32)
        for i in range(1, seq_len):
            base_obs[i] = base_obs[i-1] * autocorr + base_obs[i] * (1 - autocorr)
        
        lang = np.random.randn(seq_len, 32).astype(np.float32) * 0.5
        
        action = np.random.randn(seq_len, 7).astype(np.float32) * 0.3
        for i in range(1, seq_len):
            action[i] = action[i-1] * 0.8 + action[i] * 0.2
        
        observations.append(base_obs)
        languages.append(lang)
        actions.append(action[-1])
    
    class DummyDataset(Dataset):
        def __init__(self, obs, lang, act):
            self.obs = torch.tensor(obs, dtype=torch.float32)
            self.lang = torch.tensor(lang, dtype=torch.float32)
            self.act = torch.tensor(act, dtype=torch.float32)
        
        def __len__(self):
            return len(self.obs)
        
        def __getitem__(self, idx):
            return {'observation': self.obs[idx], 'language': self.lang[idx], 'action': self.act[idx]}
    
    return DummyDataset(observations, languages, actions)


def train_model(model, train_loader, val_loader, epochs=30):
    """Train model and return validation loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                val_losses.append(criterion(pred, batch['action']).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


def run_experiment():
    """Run the hierarchical attention experiment on 80-100 steps"""
    print("=" * 70)
    print("H1.248: Hierarchical Attention on 80-100 Step Sequences")
    print("=" * 70)
    
    results = {}
    
    seq_lengths = [80, 90, 100]
    autocorr = 0.95
    
    hier_improvements = []
    std_improvements = []
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing seq_len={seq_len} ---")
        
        train_data = create_dataset(seq_len, 100, autocorr)
        val_data = create_dataset(30, 30, autocorr)
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)
        
        print("  Training Baseline...")
        baseline = BaselineModel()
        base_loss = train_model(baseline, train_loader, val_loader)
        
        print("  Training Hierarchical Attention (segment=20, reg=0.35)...")
        hier_model = HierarchicalAttentionModel(segment_size=20, reg=0.35)
        hier_loss = train_model(hier_model, train_loader, val_loader)
        
        print("  Training Standard Attention (reg=0.35)...")
        std_model = StandardAttentionModel(reg=0.35)
        std_loss = train_model(std_model, train_loader, val_loader)
        
        hier_improvement = (base_loss - hier_loss) / base_loss * 100
        std_improvement = (base_loss - std_loss) / base_loss * 100
        
        hier_improvements.append(hier_improvement)
        std_improvements.append(std_improvement)
        
        print(f"  Baseline: {base_loss:.6f}")
        print(f"  Hierarchical: {hier_loss:.6f} ({hier_improvement:+.1f}%)")
        print(f"  Standard: {std_loss:.6f} ({std_improvement:+.1f}%)")
        
        results[f'seq_{seq_len}'] = {
            'baseline': base_loss,
            'hierarchical': hier_loss,
            'standard': std_loss,
            'hier_improvement': hier_improvement,
            'std_improvement': std_improvement
        }
    
    avg_hier = np.mean(hier_improvements)
    avg_std = np.mean(std_improvements)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Average Hierarchical Improvement: {avg_hier:+.1f}%")
    print(f"Average Standard Attention Improvement: {avg_std:+.1f}%")
    print(f"Hierarchical vs Standard: {avg_hier - avg_std:+.1f}%")
    
    if avg_hier > avg_std and avg_hier > 0:
        status = "SUPPORTED"
        print(f"\nStatus: {status} - Hierarchical attention extends beyond 80 steps")
    elif avg_std > avg_hier and avg_std > 0:
        status = "REFUTED"
        print(f"\nStatus: {status} - Standard attention still better on 80-100 steps")
    else:
        status = "INCONCLUSIVE"
        print(f"\nStatus: {status}")
    
    results['summary'] = {
        'avg_hier_improvement': avg_hier,
        'avg_std_improvement': avg_std,
        'status': status,
        'hier_wins': sum(1 for h in hier_improvements if h > 0),
        'std_wins': sum(1 for s in std_improvements if s > 0),
        'hier_better_than_std': sum(1 for i in range(len(seq_lengths)) if hier_improvements[i] > std_improvements[i])
    }
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.248-hierarchical-attention-80-100-steps/results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to metrics.json")
    return results


if __name__ == "__main__":
    run_experiment()