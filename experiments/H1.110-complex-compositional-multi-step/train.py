#!/usr/bin/env python3
"""H1.110: Attention on Extreme Multi-Step Tasks (50-100 steps)"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    """Multi-head attention for temporal modeling"""
    def __init__(self, state_dim: int, hidden: int = 64, num_heads: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        
        self.q_proj = nn.Linear(state_dim, hidden)
        self.k_proj = nn.Linear(state_dim, hidden)
        self.v_proj = nn.Linear(state_dim, hidden)
        self.out_proj = nn.Linear(hidden, state_dim)
        
    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)

class UnifiedModel(nn.Module):
    """Unified architecture (32k dims from H1.20)"""
    def __init__(self, state_dim: int, hidden: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ConcatBaseline(nn.Module):
    """Simple concatenation baseline"""
    def __init__(self, state_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class AttentionUnified(nn.Module):
    """Attention + Unified combined"""
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.attn = AttentionModel(state_dim, hidden)
        self.unified = UnifiedModel(state_dim, hidden * 2)
        self.fuse = nn.Linear(state_dim * 3, state_dim)
        
    def forward(self, x):
        attn_out = self.attn(x)
        unified_out = self.unified(x)
        fused = torch.cat([x, attn_out, unified_out], dim=-1)
        return self.fuse(fused)

def run():
    print("H1.110: Attention on Extreme Multi-Step Tasks (50-100 steps)")
    print("=" * 55)
    
    state_dim = 16
    hidden = 64
    num_samples = 50
    seq_lengths = [50, 60, 70, 80, 90, 100]
    results = []
    
    for ns in seq_lengths:
        np.random.seed(ns * 100); torch.manual_seed(ns * 100)
        
        # Generate data: multi-step trajectories
        S = torch.randn(num_samples, ns, state_dim)
        T = S + torch.randn(num_samples, ns, state_dim) * 0.1
        for t in range(1, ns):
            T[:, t] = T[:, t-1] + (S[:, t] - S[:, t-1]) * 0.5
        
        # Baseline: Concatenation
        baseline = ConcatBaseline(state_dim, hidden * 4)
        opt = torch.optim.Adam(baseline.parameters(), lr=1e-3)
        for epoch in range(30):
            for i in range(num_samples):
                opt.zero_grad()
                pred = baseline(S[i:i+1])
                loss = ((pred - T[i:i+1])**2).mean()
                loss.backward(); opt.step()
        
        pred = baseline(S)
        base_loss = ((pred - T)**2).mean().item()
        
        # Unified 32k
        unified = UnifiedModel(state_dim, hidden * 4)
        opt = torch.optim.Adam(unified.parameters(), lr=1e-3)
        for epoch in range(30):
            for i in range(num_samples):
                opt.zero_grad()
                pred = unified(S[i:i+1])
                loss = ((pred - T[i:i+1])**2).mean()
                loss.backward(); opt.step()
        
        pred = unified(S)
        unified_loss = ((pred - T)**2).mean().item()
        
        # Attention
        attn = AttentionModel(state_dim, hidden * 4)
        opt = torch.optim.Adam(attn.parameters(), lr=1e-3)
        for epoch in range(30):
            for i in range(num_samples):
                opt.zero_grad()
                pred = attn(S[i:i+1])
                loss = ((pred - T[i:i+1])**2).mean()
                loss.backward(); opt.step()
        
        pred = attn(S)
        attn_loss = ((pred - T)**2).mean().item()
        
        # Combined
        combined = AttentionUnified(state_dim, hidden * 2)
        opt = torch.optim.Adam(combined.parameters(), lr=1e-3)
        for epoch in range(30):
            for i in range(num_samples):
                opt.zero_grad()
                pred = combined(S[i:i+1])
                loss = ((pred - T[i:i+1])**2).mean()
                loss.backward(); opt.step()
        
        pred = combined(S)
        combined_loss = ((pred - T)**2).mean().item()
        
        base_vs_unified = (base_loss - unified_loss) / base_loss * 100 if base_loss > 0 else 0
        base_vs_attn = (base_loss - attn_loss) / base_loss * 100 if base_loss > 0 else 0
        base_vs_comb = (base_loss - combined_loss) / base_loss * 100 if base_loss > 0 else 0
        
        print(f"{ns} steps: Base={base_loss:.4f}, Unif={unified_loss:.4f}, Attn={attn_loss:.4f}, Comb={combined_loss:.4f}")
        print(f"         Base→Unif: {base_vs_unified:+.1f}%, Base→Attn: {base_vs_attn:+.1f}%, Base→Comb: {base_vs_comb:+.1f}%")
        
        results.append({
            'steps': ns,
            'baseline': base_loss,
            'unified': unified_loss,
            'attention': attn_loss,
            'combined': combined_loss,
            'base_vs_unified': base_vs_unified,
            'base_vs_attn': base_vs_attn,
            'base_vs_comb': base_vs_comb
        })
    
    avg_unified = np.mean([r['base_vs_unified'] for r in results])
    avg_attn = np.mean([r['base_vs_attn'] for r in results])
    avg_comb = np.mean([r['base_vs_comb'] for r in results])
    
    print("\n" + "=" * 55)
    print(f"Summary:")
    print(f"  Baseline → Unified: {avg_unified:+.1f}% avg")
    print(f"  Baseline → Attention: {avg_attn:+.1f}% avg")
    print(f"  Baseline → Combined: {avg_comb:+.1f}% avg")
    
    status = "SUPPORTED" if avg_attn > 10 else "INCONCLUSIVE" if avg_attn > 0 else "REFUTED"
    print(f"\nStatus: {status}")
    
    return results, status

if __name__ == "__main__":
    results, status = run()