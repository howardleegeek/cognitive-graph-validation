#!/usr/bin/env python3
"""
H1.470.1: Representation Bottleneck - Dimension Sweep

Hypothesis: CG's advantage decreases with task complexity because the fixed 512-dim 
unified representation becomes a bottleneck when encoding both current state and 
task history. Increasing representation dimension should reduce this gap.

Prediction: Larger unified representations (768, 1024, 2048) will show:
1. Better absolute performance on multi-step tasks
2. Smaller single-to-multi performance gap
3. CG advantage maintained or increased on multi-step tasks

Test: Compare CG with dimensions [256, 512, 768, 1024] on single-step vs 3-step.
If representation bottleneck is the issue, larger dims should disproportionately help multi-step.

Falsification criteria:
- REFUTED if: Larger representations don't improve multi-step performance relative to single-step
- REFUTED if: Baseline also improves proportionally (general capacity issue, not CG-specific)
- SUPPORTED if: CG multi-step improvement increases disproportionately with larger representations
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang, history=None):
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        return self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))


class CognitiveGraph(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 unified_dim=512, dropout=0.4, n_gnn_layers=2):
        super().__init__()
        self.unified_dim = unified_dim
        
        ratio = unified_dim / 512.0
        self.scaled_physical = int(144 * ratio)
        self.scaled_semantic = unified_dim - self.scaled_physical
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, self.scaled_physical), nn.LayerNorm(self.scaled_physical)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, self.scaled_semantic), nn.LayerNorm(self.scaled_semantic)
        )
        
        self.history_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, self.scaled_physical), nn.LayerNorm(self.scaled_physical)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(unified_dim, unified_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(unified_dim)
            ) for _ in range(n_gnn_layers)
        ])
        
        n_heads = max(1, min(4, unified_dim // 64))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=unified_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(unified_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang, history=None):
        physical = self.obs_to_physical(obs)
        semantic = self.lang_to_semantic(lang)
        unified = torch.cat([physical, semantic], dim=-1)
        
        x = unified.unsqueeze(1)
        
        if history is not None:
            hist_physical = self.history_encoder(history)
            batch_size, n_steps, _ = hist_physical.shape
            hist_semantic_zeros = torch.zeros(batch_size, n_steps, self.scaled_semantic, device=hist_physical.device)
            hist_unified = torch.cat([hist_physical, hist_semantic_zeros], dim=-1)
            x = torch.cat([x, hist_unified], dim=1)
        
        for gnn in self.gnn_layers:
            x = gnn(x) + x
        
        attn_out, _ = self.cross_attention(x, x, x)
        x = x + attn_out
        
        pooled = x.mean(dim=1)
        return self.decoder(pooled)


# ============================================================
# Data Generation
# ============================================================

def generate_task_data(n_samples, obs_dim=8, lang_dim=32, action_dim=7, 
                       task_type="single_step", n_steps=1, seed=42):
    rng = np.random.RandomState(seed)
    
    obs = rng.randn(n_samples, obs_dim).astype(np.float32)
    lang = rng.randn(n_samples, lang_dim).astype(np.float32)
    
    W_obs = rng.randn(obs_dim, action_dim).astype(np.float32)
    W_lang = rng.randn(lang_dim, action_dim).astype(np.float32)
    
    if task_type == "single_step":
        actions = (0.3 * obs @ W_obs + 0.2 * lang @ W_lang +
                   0.1 * rng.randn(n_samples, action_dim).astype(np.float32))
        return obs, lang, actions, None
    else:
        history_states = [rng.randn(n_samples, obs_dim).astype(np.float32) for _ in range(n_steps)]
        history = np.stack(history_states, axis=1)
        W_hist = rng.randn(obs_dim, action_dim).astype(np.float32)
        actions = (0.2 * obs @ W_obs + 0.15 * lang @ W_lang +
                   0.1 * history.mean(axis=1) @ W_hist +
                   0.1 * rng.randn(n_samples, action_dim).astype(np.float32))
        return obs, lang, actions, history


def train_model(model, train_loader, n_epochs=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(n_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 4 and batch[3] is not None:
                obs, lang, actions, history = batch
                pred = model(obs, lang, history=history)
            else:
                obs, lang, actions = batch[0], batch[1], batch[2]
                pred = model(obs, lang)
            loss = F.mse_loss(pred, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4 and batch[3] is not None:
                obs, lang, actions, history = batch
                pred = model(obs, lang, history=history)
            else:
                obs, lang, actions = batch[0], batch[1], batch[2]
                pred = model(obs, lang)
            loss = F.mse_loss(pred, actions)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def make_loader(obs, lang, actions, history=None, batch_size=128, shuffle=True):
    if history is not None:
        dataset = TensorDataset(torch.tensor(obs), torch.tensor(lang), torch.tensor(actions), torch.tensor(history))
    else:
        dataset = TensorDataset(torch.tensor(obs), torch.tensor(lang), torch.tensor(actions))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 60)
    print("H1.470.1: Representation Bottleneck - Dimension Sweep")
    print("=" * 60)
    
    dimensions = [256, 512, 768, 1024]
    n_epochs = 15
    lr = 0.001
    n_train = 800
    n_test = 200
    
    results = {
        "hypothesis": "H1.470.1: Representation Bottleneck",
        "prediction": "Larger unified representations reduce single-to-multi performance gap",
        "dimensions": dimensions,
        "detailed_results": {},
        "analysis": {}
    }
    
    for dim in dimensions:
        print(f"\n--- Testing unified_dim={dim} ---")
        
        single_obs, single_lang, single_actions, _ = generate_task_data(n_train, task_type="single_step", seed=42)
        ts_obs, ts_lang, ts_actions, _ = generate_task_data(n_test, task_type="single_step", seed=123)
        
        m_obs, m_lang, m_actions, m_hist = generate_task_data(n_train, task_type="multi_step", n_steps=3, seed=42)
        tm_obs, tm_lang, tm_actions, tm_hist = generate_task_data(n_test, task_type="multi_step", n_steps=3, seed=123)
        
        single_train_loader = make_loader(single_obs, single_lang, single_actions)
        single_test_loader = make_loader(ts_obs, ts_lang, ts_actions, shuffle=False)
        multi_train_loader = make_loader(m_obs, m_lang, m_actions, m_hist)
        multi_test_loader = make_loader(tm_obs, tm_lang, tm_actions, tm_hist, shuffle=False)
        
        # Baseline single
        bl_s = BaselineArchitecture()
        train_model(bl_s, single_train_loader, n_epochs, lr)
        bl_loss_s = evaluate_model(bl_s, single_test_loader)
        
        # CG single
        cg_s = CognitiveGraph(unified_dim=dim)
        train_model(cg_s, single_train_loader, n_epochs, lr)
        cg_loss_s = evaluate_model(cg_s, single_test_loader)
        
        # Baseline multi
        bl_m = BaselineArchitecture()
        train_model(bl_m, multi_train_loader, n_epochs, lr)
        bl_loss_m = evaluate_model(bl_m, multi_test_loader)
        
        # CG multi
        cg_m = CognitiveGraph(unified_dim=dim)
        train_model(cg_m, multi_train_loader, n_epochs, lr)
        cg_loss_m = evaluate_model(cg_m, multi_test_loader)
        
        single_imp = ((bl_loss_s - cg_loss_s) / bl_loss_s) * 100
        multi_imp = ((bl_loss_m - cg_loss_m) / bl_loss_m) * 100
        imp_gap = multi_imp - single_imp
        bl_s2m = ((bl_loss_s - bl_loss_m) / bl_loss_s) * 100
        cg_s2m = ((cg_loss_s - cg_loss_m) / cg_loss_s) * 100
        
        print(f"  Single: bl={bl_loss_s:.6f}, cg={cg_loss_s:.6f}, imp={single_imp:+.2f}%")
        print(f"  Multi:  bl={bl_loss_m:.6f}, cg={cg_loss_m:.6f}, imp={multi_imp:+.2f}%")
        print(f"  Gap: {imp_gap:+.2f}% | bl_s2m={bl_s2m:+.2f}% | cg_s2m={cg_s2m:+.2f}%")
        
        results["detailed_results"][f"dim_{dim}"] = {
            "unified_dim": dim,
            "single_step": {"baseline_loss": round(bl_loss_s, 6), "cg_loss": round(cg_loss_s, 6), "improvement_percent": round(single_imp, 2)},
            "multi_step": {"baseline_loss": round(bl_loss_m, 6), "cg_loss": round(cg_loss_m, 6), "improvement_percent": round(multi_imp, 2)},
            "improvement_gap": round(imp_gap, 2),
            "baseline_single_to_multi_change": round(bl_s2m, 2),
            "cg_single_to_multi_change": round(cg_s2m, 2)
        }
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    gaps = [results["detailed_results"][f"dim_{d}"]["improvement_gap"] for d in dimensions]
    cg_s2m_vals = [results["detailed_results"][f"dim_{d}"]["cg_single_to_multi_change"] for d in dimensions]
    bl_s2m_vals = [results["detailed_results"][f"dim_{d}"]["baseline_single_to_multi_change"] for d in dimensions]
    
    print(f"Gaps by dim: {dict(zip(dimensions, [round(g,2) for g in gaps]))}")
    print(f"CG s2m by dim: {dict(zip(dimensions, [round(c,2) for c in cg_s2m_vals]))}")
    print(f"BL s2m by dim: {dict(zip(dimensions, [round(b,2) for b in bl_s2m_vals]))}")
    
    gap_512 = results["detailed_results"]["dim_512"]["improvement_gap"]
    gap_1024 = results["detailed_results"]["dim_1024"]["improvement_gap"]
    gap_trend = gap_1024 - gap_512
    
    cg_s2m_512 = results["detailed_results"]["dim_512"]["cg_single_to_multi_change"]
    cg_s2m_1024 = results["detailed_results"]["dim_1024"]["cg_single_to_multi_change"]
    cg_change_trend = cg_s2m_1024 - cg_s2m_512
    
    hypothesis_supported = gap_trend > 0
    
    results["analysis"] = {
        "improvement_gaps_by_dimension": dict(zip(dimensions, [round(g, 2) for g in gaps])),
        "cg_single_to_multi_change_by_dimension": dict(zip(dimensions, [round(c, 2) for c in cg_s2m_vals])),
        "baseline_single_to_multi_change_by_dimension": dict(zip(dimensions, [round(b, 2) for b in bl_s2m_vals])),
        "gap_512": round(gap_512, 2),
        "gap_1024": round(gap_1024, 2),
        "gap_trend_1024_vs_512": round(gap_trend, 2),
        "cg_change_trend_1024_vs_512": round(cg_change_trend, 2),
        "hypothesis_supported": hypothesis_supported,
        "key_insight": f"{'SUPPORTED' if hypothesis_supported else 'REFUTED'}: Representation bottleneck {'confirmed' if hypothesis_supported else 'not confirmed'} - larger dimensions {'reduce' if hypothesis_supported else 'do not reduce'} the single-to-multi performance gap (gap trend: {gap_trend:+.2f}%)"
    }
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/083-representation-dimension/results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nHypothesis supported: {hypothesis_supported}")
    print(f"Key insight: {results['analysis']['key_insight']}")


if __name__ == "__main__":
    main()
