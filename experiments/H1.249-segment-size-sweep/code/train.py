#!/usr/bin/env python3
"""
H1.249: Segment Size Sweep for Hierarchical Attention

Based on findings:
- H1.247: Hierarchical attention +7.7% on 50-80 steps with segment_size=15
- H1.248: Hierarchical attention +5.8% on 80-100 steps with segment_size=20

Hypothesis: Different segment sizes may optimize hierarchical attention for different 
sequence lengths. Test segment sizes [10, 15, 20, 25, 30] to find optimal.
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
    """Hierarchical attention with configurable segment size"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256, 
                 num_heads=8, segment_size=15, reg=0.35):
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
        
        return self.decoder(global_summary)


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
    """Run the segment size sweep experiment"""
    print("=" * 70)
    print("H1.249: Segment Size Sweep for Hierarchical Attention")
    print("=" * 70)
    
    results = {}
    
    seq_lengths = [60, 80, 100]
    segment_sizes = [10, 15, 20, 25, 30]
    autocorr = 0.95
    
    best_per_seq_len = {}
    
    for seq_len in seq_lengths:
        print(f"\n=== Testing seq_len={seq_len} ===")
        
        train_data = create_dataset(seq_len, 100, autocorr)
        val_data = create_dataset(30, 30, autocorr)
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)
        
        print("  Training Baseline...")
        baseline = BaselineModel()
        base_loss = train_model(baseline, train_loader, val_loader)
        print(f"  Baseline: {base_loss:.6f}")
        
        segment_results = {}
        best_improvement = -float('inf')
        best_segment = None
        
        for seg_size in segment_sizes:
            print(f"  Testing segment_size={seg_size}...")
            model = HierarchicalAttentionModel(segment_size=seg_size, reg=0.35)
            loss = train_model(model, train_loader, val_loader)
            
            improvement = (base_loss - loss) / base_loss * 100
            segment_results[seg_size] = {'loss': loss, 'improvement': improvement}
            
            print(f"    MSE: {loss:.6f}, Improvement: {improvement:+.1f}%")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_segment = seg_size
        
        best_per_seq_len[seq_len] = {
            'baseline': base_loss,
            'best_segment': best_segment,
            'best_improvement': best_improvement,
            'all_segments': segment_results
        }
        
        print(f"  Best segment_size for seq_len={seq_len}: {best_segment} ({best_improvement:+.1f}%)")
        
        results[f'seq_{seq_len}'] = best_per_seq_len[seq_len]
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for seq_len, data in best_per_seq_len.items():
        print(f"seq_len={seq_len}: best segment={data['best_segment']}, improvement={data['best_improvement']:+.1f}%")
    
    avg_improvement = np.mean([d['best_improvement'] for d in best_per_seq_len.values()])
    print(f"\nAverage Best Improvement: {avg_improvement:+.1f}%")
    
    segment_wins = {}
    for seq_len, data in best_per_seq_len.items():
        for seg, res in data['all_segments'].items():
            if seg not in segment_wins:
                segment_wins[seg] = []
            segment_wins[seg].append(res['improvement'])
    
    print("\nSegment Size Performance:")
    for seg, imps in sorted(segment_wins.items()):
        print(f"  segment_size={seg}: avg improvement={np.mean(imps):+.1f}%")
    
    status = "SUPPORTED" if avg_improvement > 0 else "REFUTED"
    print(f"\nStatus: {status}")
    
    results['summary'] = {
        'avg_improvement': avg_improvement,
        'status': status,
        'best_per_seq_len': {k: {'best_segment': v['best_segment'], 'best_improvement': v['best_improvement']} for k, v in best_per_seq_len.items()}
    }
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.249-segment-size-sweep/results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to metrics.json")
    return results


if __name__ == "__main__":
    run_experiment()