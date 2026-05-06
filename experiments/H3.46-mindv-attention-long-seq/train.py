#!/usr/bin/env python3
"""H3.46: MIND-V SRH + Attention on Long Sequences"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRHModel(nn.Module):
    """Semantic Reasoning Hub (from H3.45)"""
    def __init__(self, state_dim: int, lang_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.lang_dim = lang_dim
        self.hidden = hidden
        self.srh = nn.Sequential(
            nn.Linear(state_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.bsb = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mvg = nn.Sequential(
            nn.Linear(state_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, state, language):
        B, T, D = state.shape
        state_flat = state.reshape(-1, D)
        lang_flat = language.reshape(-1, self.lang_dim)
        
        task_emb = self.srh(torch.cat([state_flat, lang_flat], dim=-1))
        bsb = self.bsb(task_emb)
        out_flat = self.mvg(torch.cat([state_flat, bsb], dim=-1))
        return out_flat.reshape(B, T, D)

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

class SRHAttention(nn.Module):
    """SRH + Attention combined"""
    def __init__(self, state_dim: int, lang_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.lang_dim = lang_dim
        self.hidden = hidden
        self.srh = SRHModel(state_dim, lang_dim, hidden)
        self.attn = AttentionModel(state_dim, hidden)
        self.fuse = nn.Linear(state_dim * 2, state_dim)
        
    def forward(self, state, language):
        B, T, D = state.shape
        srh_out = self.srh(state, language)
        attn_out = self.attn(state)
        fused = torch.cat([srh_out, attn_out], dim=-1)
        return self.fuse(fused)

class ConcatBaseline(nn.Module):
    """Simple concatenation baseline"""
    def __init__(self, state_dim: int, lang_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.lang_dim = lang_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, state, language):
        B, T, D = state.shape
        state_flat = state.reshape(-1, D)
        lang_flat = language.reshape(-1, self.lang_dim)
        out_flat = self.net(torch.cat([state_flat, lang_flat], dim=-1))
        return out_flat.reshape(B, T, D)

class SRHConcat(nn.Module):
    """SRH with simple concatenation (no attention)"""
    def __init__(self, state_dim: int, lang_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.lang_dim = lang_dim
        self.srh = SRHModel(state_dim, lang_dim, hidden)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, state, language):
        srh_out = self.srh(state, language)
        return self.predictor(srh_out)

def run():
    print("H3.46: MIND-V SRH + Attention on Long Sequences")
    print("=" * 50)
    
    state_dim, lang_dim = 16, 32
    hidden = 64
    num_heads = 4
    num_samples = 50
    seq_lengths = [40, 50, 60, 80, 100]
    results = []
    
    for ns in seq_lengths:
        np.random.seed(ns * 100); torch.manual_seed(ns * 100)
        
        S = torch.randn(num_samples, ns, state_dim)
        L = torch.randn(num_samples, ns, lang_dim)
        T = S + torch.randn(num_samples, ns, state_dim) * 0.1
        for t in range(1, ns):
            T[:, t] = T[:, t-1] + (S[:, t] - S[:, t-1]) * 0.5
        
        baseline = ConcatBaseline(state_dim, lang_dim, hidden)
        opt = torch.optim.Adam(baseline.parameters(), lr=1e-3)
        for epoch in range(20):
            for i in range(num_samples):
                opt.zero_grad()
                pred = baseline(S[i:i+1], L[i:i+1])
                loss = ((pred - T[i:i+1])**2).mean()
                loss.backward(); opt.step()
        
        pred = baseline(S, L)
        base_loss = ((pred - T)**2).mean().item()
        
        srh_concat = SRHConcat(state_dim, lang_dim, hidden)
        opt = torch.optim.Adam(srh_concat.parameters(), lr=1e-3)
        for epoch in range(20):
            for i in range(num_samples):
                opt.zero_grad()
                pred = srh_concat(S[i:i+1], L[i:i+1])
                loss = ((pred - T[i:i+1])**2).mean()
                loss.backward(); opt.step()
        
        pred = srh_concat(S, L)
        srh_loss = ((pred - T)**2).mean().item()
        
        srh_attn = SRHAttention(state_dim, lang_dim, hidden)
        opt = torch.optim.Adam(srh_attn.parameters(), lr=1e-3)
        for epoch in range(20):
            for i in range(num_samples):
                opt.zero_grad()
                pred = srh_attn(S[i:i+1], L[i:i+1])
                loss = ((pred - T[i:i+1])**2).mean()
                loss.backward(); opt.step()
        
        pred = srh_attn(S, L)
        combined_loss = ((pred - T)**2).mean().item()
        
        base_vs_srh = (base_loss - srh_loss) / base_loss * 100 if base_loss > 0 else 0
        base_vs_comb = (base_loss - combined_loss) / base_loss * 100 if base_loss > 0 else 0
        srh_vs_comb = (srh_loss - combined_loss) / srh_loss * 100 if srh_loss > 0 else 0
        
        print(f"{ns} steps: Base={base_loss:.4f}, SRH={srh_loss:.4f}, Attn+SRH={combined_loss:.4f}")
        print(f"         Base→SRH: {base_vs_srh:+.1f}%, Base→Attn: {base_vs_comb:+.1f}%, SRH→Attn: {srh_vs_comb:+.1f}%")
        
        results.append({
            'steps': ns,
            'baseline': base_loss,
            'srh': srh_loss,
            'combined': combined_loss,
            'base_vs_srh': base_vs_srh,
            'base_vs_comb': base_vs_comb,
            'srh_vs_comb': srh_vs_comb
        })
    
    avg_base_vs_srh = np.mean([r['base_vs_srh'] for r in results])
    avg_base_vs_comb = np.mean([r['base_vs_comb'] for r in results])
    avg_srh_vs_comb = np.mean([r['srh_vs_comb'] for r in results])
    
    print("\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Baseline → SRH:  {avg_base_vs_srh:+.1f}% avg")
    print(f"  Baseline → Attn:  {avg_base_vs_comb:+.1f}% avg")
    print(f"  SRH → Combined:   {avg_srh_vs_comb:+.1f}% avg")
    
    status = "SUPPORTED" if avg_base_vs_comb > 10 else "INCONCLUSIVE" if avg_base_vs_comb > 0 else "REFUTED"
    print(f"\nStatus: {status}")
    
    return results, status

if __name__ == "__main__":
    results, status = run()