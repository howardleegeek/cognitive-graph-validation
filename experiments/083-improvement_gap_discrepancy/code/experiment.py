#!/usr/bin/env python3
"""
H1.470.1.1.3: Improvement Gap Sign Discrepancy Investigation (Fast)

Hypothesis: The discrepancy in improvement gap sign (positive in simulation vs 
negative in real experiments) indicates that the simulation model doesn't capture 
the key mechanism that makes CG better on multi-step tasks in real data.

Prediction: Adding structured cross-modal relationships and temporal dependencies
will flip the gap sign from positive to negative, matching real experiments.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Lightweight Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=32):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraph(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=36, semantic_dim=92, dropout=0.1):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(2)
        ])
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=total_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, obs, lang):
        physical = self.obs_to_physical(obs)
        semantic = self.lang_to_semantic(lang)
        combined = torch.cat([physical, semantic], dim=-1)
        
        for gnn in self.gnn_layers:
            combined = gnn(combined) + combined
        
        combined = combined.unsqueeze(1)
        attended, _ = self.cross_attention(combined, combined, combined)
        combined = combined + attended
        combined = combined.squeeze(1)
        
        return self.action_head(combined)


# ============================================================
# Data Generation
# ============================================================

def generate_random_data(n_samples, obs_dim=8, lang_dim=32, action_dim=7, task_type="single_step"):
    observations = torch.randn(n_samples, obs_dim)
    language = torch.randn(n_samples, lang_dim)
    
    if task_type == "single_step":
        W = torch.randn(obs_dim + lang_dim, action_dim) * 0.3
        combined = torch.cat([observations, language], dim=-1)
        actions = combined @ W + torch.randn(n_samples, action_dim) * 0.01
    else:
        W1 = torch.randn(obs_dim + lang_dim, 16) * 0.3
        W2 = torch.randn(16, 16) * 0.3
        W3 = torch.randn(16, action_dim) * 0.3
        combined = torch.cat([observations, language], dim=-1)
        h1 = F.relu(combined @ W1)
        h2 = F.relu(h1 @ W2)
        actions = h2 @ W3 + torch.randn(n_samples, action_dim) * 0.02
    
    return observations, language, actions


def generate_structured_data(n_samples, obs_dim=8, lang_dim=32, action_dim=7, task_type="single_step"):
    observations = torch.randn(n_samples, obs_dim)
    
    language = torch.zeros(n_samples, lang_dim)
    language[:, :4] = observations[:, :4] * 0.8 + torch.randn(n_samples, 4) * 0.1
    language[:, 4:8] = observations[:, 4:] * 0.6 + torch.randn(n_samples, 4) * 0.15
    language[:, 8:] = torch.randn(n_samples, lang_dim - 8) * 0.5
    
    if task_type == "single_step":
        W_obs = torch.randn(obs_dim, action_dim) * 0.2
        W_lang = torch.randn(lang_dim, action_dim) * 0.2
        shared = torch.randn(min(obs_dim, lang_dim), action_dim) * 0.15
        W_obs[:min(obs_dim, lang_dim)] += shared
        W_lang[:min(obs_dim, lang_dim)] += shared
        actions = observations @ W_obs + language @ W_lang + torch.randn(n_samples, action_dim) * 0.01
    else:
        W1_obs = torch.randn(obs_dim, 16) * 0.2
        W1_lang = torch.randn(lang_dim, 16) * 0.2
        shared1 = torch.randn(min(obs_dim, lang_dim), 16) * 0.15
        W1_obs[:min(obs_dim, lang_dim)] += shared1
        W1_lang[:min(obs_dim, lang_dim)] += shared1
        h1 = F.relu(observations @ W1_obs + language @ W1_lang)
        W2 = torch.randn(16, 16) * 0.2
        h2 = F.relu(h1 @ W2)
        W3 = torch.randn(16, action_dim) * 0.2
        actions = h2 @ W3 + torch.randn(n_samples, action_dim) * 0.02
    
    return observations, language, actions


def generate_temporal_data(n_samples, obs_dim=8, lang_dim=32, action_dim=7, task_type="single_step"):
    observations = torch.randn(n_samples, obs_dim)
    language = torch.randn(n_samples, lang_dim)
    
    if task_type == "single_step":
        W = torch.randn(obs_dim + lang_dim, action_dim) * 0.3
        combined = torch.cat([observations, language], dim=-1)
        actions = combined @ W + torch.randn(n_samples, action_dim) * 0.01
    else:
        W1 = torch.randn(obs_dim + lang_dim, 16) * 0.3
        combined = torch.cat([observations, language], dim=-1)
        h1 = F.relu(combined @ W1)
        obs_modified = observations + h1[:, :obs_dim] * 0.1
        W2 = torch.randn(obs_dim + 16, 16) * 0.3
        h2_input = torch.cat([obs_modified, h1], dim=-1)
        h2 = F.relu(h2_input @ W2)
        W3 = torch.randn(16 + obs_dim, action_dim) * 0.3
        h3_input = torch.cat([h2, obs_modified], dim=-1)
        actions = h3_input @ W3 + torch.randn(n_samples, action_dim) * 0.02
    
    return observations, language, actions


# ============================================================
# Training
# ============================================================

def train_and_eval(model, train_obs, train_lang, train_actions,
                   val_obs, val_lang, val_actions,
                   epochs=20, lr=1e-3, batch_size=256):
    train_dataset = TensorDataset(train_obs, train_lang, train_actions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 8
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_obs, batch_lang, batch_actions in train_loader:
            optimizer.zero_grad()
            pred = model(batch_obs, batch_lang)
            loss = criterion(pred, batch_actions)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 4 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_obs, val_lang)
                val_loss = criterion(val_pred, val_actions).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(val_obs, val_lang)
        val_loss = criterion(val_pred, val_actions).item()
    
    return val_loss


def run_experiment(data_regime, task_type, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if data_regime == "random":
        gen_fn = generate_random_data
    elif data_regime == "structured":
        gen_fn = generate_structured_data
    elif data_regime == "temporal":
        gen_fn = generate_temporal_data
    else:
        raise ValueError(f"Unknown regime: {data_regime}")
    
    n_train = 500
    n_val = 100
    
    train_obs, train_lang, train_actions = gen_fn(n_train, task_type=task_type)
    val_obs, val_lang, val_actions = gen_fn(n_val, task_type=task_type)
    
    baseline = BaselineArchitecture()
    baseline_loss = train_and_eval(baseline, train_obs, train_lang, train_actions,
                                   val_obs, val_lang, val_actions)
    
    cg = CognitiveGraph()
    cg_loss = train_and_eval(cg, train_obs, train_lang, train_actions,
                             val_obs, val_lang, val_actions)
    
    improvement = (baseline_loss - cg_loss) / baseline_loss * 100
    
    return {
        "baseline_loss": round(baseline_loss, 6),
        "cg_loss": round(cg_loss, 6),
        "improvement_pct": round(improvement, 2)
    }


# ============================================================
# Main
# ============================================================

def main():
    results = {
        "experiment_id": "H1.470.1.1.3",
        "title": "Improvement Gap Sign Discrepancy Investigation",
        "date": datetime.now().isoformat(),
        "regimes": {},
        "summary": {}
    }
    
    regimes = ["random", "structured", "temporal"]
    task_types = ["single_step", "multi_step"]
    n_runs = 2
    
    print("=" * 60)
    print("H1.470.1.1.3: Improvement Gap Sign Discrepancy")
    print("=" * 60)
    
    for regime in regimes:
        regime_results = {"single_step": [], "multi_step": []}
        
        for task_type in task_types:
            print(f"\n--- {regime.upper()} / {task_type} ---")
            
            for run in range(n_runs):
                result = run_experiment(regime, task_type, seed=42 + run * 100)
                regime_results[task_type].append(result)
                print(f"  Run {run+1}: baseline={result['baseline_loss']:.6f}, "
                      f"cg={result['cg_loss']:.6f}, improvement={result['improvement_pct']:+.2f}%")
        
        avg_single = np.mean([r["improvement_pct"] for r in regime_results["single_step"]])
        avg_multi = np.mean([r["improvement_pct"] for r in regime_results["multi_step"]])
        gap = avg_multi - avg_single
        
        results["regimes"][regime] = {
            "single_step_avg": round(avg_single, 2),
            "multi_step_avg": round(avg_multi, 2),
            "improvement_gap": round(gap, 2),
            "runs": regime_results
        }
        
        print(f"\n{regime.upper()} SUMMARY:")
        print(f"  Single-step avg improvement: {avg_single:+.2f}%")
        print(f"  Multi-step avg improvement:  {avg_multi:+.2f}%")
        print(f"  Improvement gap (multi - single): {gap:+.2f}%")
        print(f"  Gap sign: {'POSITIVE (CG better on multi-step)' if gap > 0 else 'NEGATIVE (CG better on single-step)'}")
    
    results["summary"] = {
        "random_gap": results["regimes"]["random"]["improvement_gap"],
        "structured_gap": results["regimes"]["structured"]["improvement_gap"],
        "temporal_gap": results["regimes"]["temporal"]["improvement_gap"],
        "gap_sign_flip": (
            results["regimes"]["random"]["improvement_gap"] * 
            results["regimes"]["temporal"]["improvement_gap"] < 0
        )
    }
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Random regime gap:     {results['summary']['random_gap']:+.2f}%")
    print(f"Structured regime gap: {results['summary']['structured_gap']:+.2f}%")
    print(f"Temporal regime gap:   {results['summary']['temporal_gap']:+.2f}%")
    print(f"Gap sign flip detected: {results['summary']['gap_sign_flip']}")
    
    os.makedirs("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/083-improvement_gap_discrepancy/results", exist_ok=True)
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/083-improvement_gap_discrepancy/results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved.")
    return results


if __name__ == "__main__":
    main()
