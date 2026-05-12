#!/usr/bin/env python3
"""
H3.117: Attention Death Zone + Autocorrelation

Hypothesis: The 30-50 step "death zone" where attention fails is due to synthetic data
lacking temporal autocorrelation. If we inject autocorrelation, attention should work.

Based on H1.181: Autocorrelation injection enables attention (+18.3% at ρ≥0.7)
Based on H3.116: Attention fails on 30-50 steps without autocorrelation (-11.2%)
"""

import numpy as np
import torch
import torch.nn as nn
import json
import os

np.random.seed(42)
torch.manual_seed(42)

def create_temporal_structure(data: np.ndarray, target_autocorr: float) -> np.ndarray:
    """Transform data to have target autocorrelation via AR(1) filter."""
    T = len(data)
    if target_autocorr == 0:
        return data
    
    x = data.copy()
    alpha = target_autocorr
    for t in range(1, T):
        x[t] = alpha * x[t-1] + np.sqrt(1 - alpha**2) * data[t]
    return x

class ConcatModel(nn.Module):
    def __init__(self, phys_dim=64, sem_dim=64):
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

class SSMModel(nn.Module):
    def __init__(self, phys_dim=64, sem_dim=64, state_dim=64):
        super().__init__()
        total_dim = phys_dim + sem_dim
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.eye(state_dim) * 0.9)
        self.B = nn.Linear(total_dim, state_dim, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
    
    def forward(self, phys, sem):
        B, T, D = phys.shape
        h = torch.cat([phys, sem], dim=-1)
        final_hidden = torch.tanh(self.B(h.mean(dim=1)) @ self.A.T)
        return self.fc(final_hidden)

def main():
    print("=" * 70)
    print("H3.117: Attention Death Zone + Autocorrelation")
    print("=" * 70)
    print("Hypothesis: Autocorrelation unlocks attention in the 30-50 step death zone")
    print()
    
    # Test lengths in the death zone
    test_lengths = [30, 35, 40, 45, 50]
    autocorr_levels = [0.0, 0.7, 0.9]  # None, medium, high
    
    all_results = {}
    
    for autocorr in autocorr_levels:
        print(f"\n{'='*60}")
        print(f"Testing autocorrelation = {autocorr}")
        print(f"{'='*60}")
        
        results = []
        
        for T in test_lengths:
            print(f"\n  Sequence length T={T}")
            
            # Generate base synthetic data
            raw_data = np.random.randn(500, T, 128).astype(np.float32) * 0.1
            
            # Physics has temporal structure based on autocorr
            physics = np.array([create_temporal_structure(raw_data[i, :, :64], autocorr) for i in range(500)])
            physics = physics + np.random.randn(500, T, 64).astype(np.float32) * 0.05
            
            semantics = raw_data[:, :, 64:]
            semantics_with_action = semantics[:, :, :64]
            
            # Goal conditioning: future state
            goals = raw_data[:, -1, :64]
            
            # Targets
            targets = raw_data[:, :, :64].sum(axis=1) / T
            
            train_phys = torch.tensor(physics[:400])
            train_sem = torch.tensor(semantics_with_action[:400])
            train_tgt = torch.tensor(targets[:400])
            val_phys = torch.tensor(physics[400:])
            val_sem = torch.tensor(semantics_with_action[400:])
            val_tgt = torch.tensor(targets[400:])
            
            # Train models
            concat_model = ConcatModel(phys_dim=64, sem_dim=64)
            attn_model = AttentionModel(phys_dim=64, sem_dim=64)
            ssm_model = SSMModel(phys_dim=64, sem_dim=64)
            
            opt_c = torch.optim.Adam(concat_model.parameters(), lr=0.001)
            opt_a = torch.optim.Adam(attn_model.parameters(), lr=0.001)
            opt_s = torch.optim.Adam(ssm_model.parameters(), lr=0.001)
            
            for epoch in range(200):
                opt_c.zero_grad()
                opt_a.zero_grad()
                opt_s.zero_grad()
                
                pred_c = concat_model(train_phys, train_sem)
                pred_a = attn_model(train_phys, train_sem)
                pred_s = ssm_model(train_phys, train_sem)
                
                loss_c = nn.MSELoss()(pred_c, train_tgt.mean(dim=0))
                loss_a = nn.MSELoss()(pred_a, train_tgt.mean(dim=0))
                loss_s = nn.MSELoss()(pred_s, train_tgt.mean(dim=0))
                
                loss_c.backward()
                loss_a.backward()
                loss_s.backward()
                
                opt_c.step()
                opt_a.step()
                opt_s.step()
            
            with torch.no_grad():
                pred_c = concat_model(val_phys, val_sem)
                pred_a = attn_model(val_phys, val_sem)
                pred_s = ssm_model(val_phys, val_sem)
                
                mse_c = nn.MSELoss()(pred_c, val_tgt.mean(dim=0)).item()
                mse_a = nn.MSELoss()(pred_a, val_tgt.mean(dim=0)).item()
                mse_s = nn.MSELoss()(pred_s, val_tgt.mean(dim=0)).item()
            
            attn_delta = (mse_a - mse_c) / mse_c * 100
            ssm_delta = (mse_s - mse_c) / mse_c * 100
            
            print(f"    Concat: {mse_c:.6f}")
            print(f"    Attention: {mse_a:.6f} ({attn_delta:+.1f}%)")
            print(f"    SSM: {mse_s:.6f} ({ssm_delta:+.1f}%)")
            
            results.append({
                'length': T,
                'concat_mse': mse_c,
                'attn_mse': mse_a,
                'ssm_mse': mse_s,
                'attn_delta': attn_delta,
                'ssm_delta': ssm_delta
            })
        
        all_results[f'autocorr_{autocorr}'] = results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Autocorrelation Effect on Death Zone")
    print("=" * 70)
    
    for autocorr_str, results in all_results.items():
        print(f"\n{autocorr_str.upper()}")
        attn_wins = sum(1 for r in results if r['attn_delta'] < 0)
        ssm_wins = sum(1 for r in results if r['ssm_delta'] < 0)
        avg_attn = np.mean([r['attn_delta'] for r in results])
        avg_ssm = np.mean([r['ssm_delta'] for r in results])
        print(f"  Attention wins: {attn_wins}/5, avg delta: {avg_attn:+.1f}%")
        print(f"  SSM wins: {ssm_wins}/5, avg delta: {avg_ssm:+.1f}%")
    
    # Calculate overall metrics
    autocorr_0 = all_results['autocorr_0.0']
    autocorr_07 = all_results['autocorr_0.7']
    autocorr_09 = all_results['autocorr_0.9']
    
    avg_attn_0 = np.mean([r['attn_delta'] for r in autocorr_0])
    avg_attn_07 = np.mean([r['attn_delta'] for r in autocorr_07])
    avg_attn_09 = np.mean([r['attn_delta'] for r in autocorr_09])
    
    avg_ssm_0 = np.mean([r['ssm_delta'] for r in autocorr_0])
    avg_ssm_07 = np.mean([r['ssm_delta'] for r in autocorr_07])
    avg_ssm_09 = np.mean([r['ssm_delta'] for r in autocorr_09])
    
    # Determine status
    if avg_attn_09 < avg_attn_0 and avg_attn_07 < avg_attn_0:
        status = "SUPPORTED"
        conclusion = "Autocorrelation unlocks attention in the death zone!"
    elif avg_attn_09 < 0:
        status = "SUPPORTED"
        conclusion = "High autocorrelation enables attention in death zone"
    else:
        status = "REFUTED"
        conclusion = "Autocorrelation alone does not unlock attention in death zone"
    
    print("\n" + "=" * 70)
    print(f"CONCLUSION: {status}")
    print("=" * 70)
    print(conclusion)
    
    print("\nKey findings:")
    print(f"  ρ=0.0: Attn avg={avg_attn_0:+.1f}%, SSM avg={avg_ssm_0:+.1f}%")
    print(f"  ρ=0.7: Attn avg={avg_attn_07:+.1f}%, SSM avg={avg_ssm_07:+.1f}%")
    print(f"  ρ=0.9: Attn avg={avg_attn_09:+.1f}%, SSM avg={avg_ssm_09:+.1f}%")
    
    # Save results
    metrics = {
        "hypothesis": "H3.117",
        "timestamp": "2026-05-12T16:30:00",
        "metrics": {
            "autocorr_0_attn_avg": avg_attn_0,
            "autocorr_07_attn_avg": avg_attn_07,
            "autocorr_09_attn_avg": avg_attn_09,
            "autocorr_0_ssm_avg": avg_ssm_0,
            "autocorr_07_ssm_avg": avg_ssm_07,
            "autocorr_09_ssm_avg": avg_ssm_09,
            "attn_improvement": avg_attn_09 - avg_attn_0,
            "ssm_improvement": avg_ssm_09 - avg_ssm_0,
        },
        "status": status,
        "detailed_results": all_results
    }
    
    results_dir = os.path.dirname(os.path.abspath(__file__)) + "/results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(results_dir + "/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/metrics.json")
    
    return metrics

if __name__ == "__main__":
    main()
