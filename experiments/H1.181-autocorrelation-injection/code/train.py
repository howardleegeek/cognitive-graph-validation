#!/usr/bin/env python3
"""
H1.181: Autocorrelation Injection - Can we unlock attention on synthetic data?

Key insight from H1.180: Real robot data has autocorrelation (0.7-0.95) which enables attention.
Synthetic data lacks this structure -> attention fails.

Hypothesis: If we inject temporal autocorrelation into synthetic data, attention should work.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

np.random.seed(42)
torch.manual_seed(42)

def generate_sequence_with_autocorr(T: int, autocorr: float) -> np.ndarray:
    """Generate sequence with specified autocorrelation."""
    x = np.zeros(T)
    x[0] = np.random.randn()
    for t in range(1, T):
        x[t] = autocorr * x[t-1] + np.sqrt(1 - autocorr**2) * np.random.randn()
    return x

def create_temporal_structure(data: np.ndarray, target_autocorr: float) -> np.ndarray:
    """Transform data to have target autocorrelation via filtering."""
    T = len(data)
    if target_autocorr == 0:
        return data
    
    x = data.copy()
    alpha = target_autocorr
    for t in range(1, T):
        x[t] = alpha * x[t-1] + np.sqrt(1 - alpha**2) * data[t]
    return x

class ConcatModel(nn.Module):
    def __init__(self, phys_dim=64, sem_dim=128):
        super().__init__()
        total_dim = phys_dim + sem_dim
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    
    def forward(self, phys, sem):
        x = torch.cat([phys, sem], dim=-1)
        x = x.mean(dim=1)
        return self.fc(x)

class AttentionModel(nn.Module):
    def __init__(self, phys_dim=64, sem_dim=64):
        super().__init__()
        total_dim = phys_dim + sem_dim
        self.qkv = nn.Linear(total_dim, total_dim * 3)
        self.proj = nn.Linear(total_dim, total_dim)
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
    
    def forward(self, phys, sem):
        B, T, D = phys.shape
        h = torch.cat([phys, sem], dim=-1)
        qkv = self.qkv(h)
        qkv = qkv.view(B, T, 3, -1)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn = torch.softmax(attn, dim=-2)
        h = torch.matmul(attn, v)
        h = h.mean(dim=1)
        return self.fc(h)

def evaluate(concat_mse, attn_mse):
    delta = (attn_mse - concat_mse) / concat_mse * 100
    return delta

def main():
    print("=" * 70)
    print("H1.181: Autocorrelation Injection Test")
    print("=" * 70)
    print("Hypothesis: Injecting temporal autocorrelation unlocks attention on synthetic data")
    print()
    
    T = 50  # Sequence length
    autocorr_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
    results = []
    
    for autocorr in autocorr_levels:
        print(f"\n{'='*60}")
        print(f"Testing autocorrelation = {autocorr}")
        print(f"{'='*60}")
        
        # Generate data
        raw_data = np.random.randn(500, T, 128).astype(np.float32) * 0.1
        physics = np.array([create_temporal_structure(raw_data[i, :, :64], autocorr) for i in range(500)])
        physics = physics + np.random.randn(500, T, 64).astype(np.float32) * 0.05
        semantics = raw_data[:, :, 64:]
        
        semantics_with_action = semantics[:, :, :64]  # 64 dims to match physics
        
        # Targets: simple position prediction
        targets = raw_data[:, :, :64].sum(axis=1) / T
        
        train_phys = torch.tensor(physics[:400])
        train_sem = torch.tensor(semantics_with_action[:400])
        train_tgt = torch.tensor(targets[:400])
        val_phys = torch.tensor(physics[400:])
        val_sem = torch.tensor(semantics_with_action[400:])
        val_tgt = torch.tensor(targets[400:])
        
        concat_model = ConcatModel(phys_dim=64, sem_dim=64)
        attn_model = AttentionModel(phys_dim=64, sem_dim=64)
        opt_c = torch.optim.Adam(concat_model.parameters(), lr=0.001)
        opt_a = torch.optim.Adam(attn_model.parameters(), lr=0.001)
        
        for epoch in range(200):
            opt_c.zero_grad()
            opt_a.zero_grad()
            
            pred_c = concat_model(train_phys, train_sem)
            pred_a = attn_model(train_phys, train_sem)
            
            loss_c = nn.MSELoss()(pred_c, train_tgt.mean(dim=0))
            loss_a = nn.MSELoss()(pred_a, train_tgt.mean(dim=0))
            
            loss_c.backward()
            loss_a.backward()
            opt_c.step()
            opt_a.step()
        
        with torch.no_grad():
            pred_c = concat_model(val_phys, val_sem)
            pred_a = attn_model(val_phys, val_sem)
            
            mse_c = nn.MSELoss()(pred_c, val_tgt.mean(dim=0)).item()
            mse_a = nn.MSELoss()(pred_a, val_tgt.mean(dim=0)).item()
        
        delta = evaluate(mse_c, mse_a)
        status = "ATTN WINS" if delta < 0 else "CONCAT WINS"
        
        print(f"  Autocorrelation: {autocorr}")
        print(f"  Concat MSE: {mse_c:.6f}")
        print(f"  Attn MSE: {mse_a:.6f}")
        print(f"  Delta: {delta:+.1f}% ({status})")
        
        results.append({
            'autocorr': autocorr,
            'concat_mse': mse_c,
            'attn_mse': mse_a,
            'delta': delta,
            'status': status
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY: Autocorrelation Effect on Attention")
    print("=" * 70)
    
    for r in results:
        print(f"  ρ={r['autocorr']:.2f}: {r['status']} ({r['delta']:+.1f}%)")
    
    # Find crossover
    crossover_found = False
    for i in range(len(results) - 1):
        if results[i]['delta'] > 0 and results[i+1]['delta'] < 0:
            crossover_found = True
            print(f"\nCROSSOVER: Attention starts winning between ρ={results[i]['autocorr']} and ρ={results[i+1]['autocorr']}")
            break
    
    if not crossover_found:
        all_positive = all(r['delta'] > 0 for r in results)
        all_negative = all(r['delta'] < 0 for r in results)
        if all_negative:
            print(f"\nAttention wins across ALL autocorrelation levels")
        else:
            print(f"\nNo clear crossover found - attention still loses at high autocorrelation")
    
    # Overall trend
    high_autocorr_results = [r for r in results if r['autocorr'] >= 0.7]
    if high_autocorr_results:
        avg_high = np.mean([r['delta'] for r in high_autocorr_results])
        print(f"\nAverage delta at high autocorrelation (ρ≥0.7): {avg_high:+.1f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if results[-1]['delta'] < 0:
        print("✓ ATTENTION UNLOCKED: High autocorrelation enables attention")
        print("  This validates H1.180's hypothesis about temporal structure")
    else:
        print("✗ ATTENTION STILL FAILS: Autocorrelation alone not sufficient")
        print("  Other factors (real robot dynamics) may be required")

if __name__ == "__main__":
    main()
