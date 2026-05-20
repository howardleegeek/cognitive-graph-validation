#!/usr/bin/env python3
"""
H1.471: Gradient Variance Analysis - Error Accumulation Mechanism

Hypothesis: CG's unified representation causes higher gradient variance across 
modalities on multi-step tasks, leading to error accumulation.

Prediction: CG will show higher gradient variance on multi-step vs single-step tasks,
while baseline (separated encoders) will show more stable gradients.

Test: Measure per-modality gradient norms during training on single-step vs multi-step.
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
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, dropout=0.4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_physical(obs)
        z_sem = self.lang_to_semantic(lang)
        
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


# ============================================================
# Data Generation
# ============================================================

def generate_data(n_samples, task_type="single_step", obs_dim=8, lang_dim=32, action_dim=7):
    np.random.seed(42)
    torch.manual_seed(42)
    
    observations = torch.randn(n_samples, obs_dim)
    language = torch.randn(n_samples, lang_dim)
    
    if task_type == "single_step":
        W = torch.randn(obs_dim + lang_dim, action_dim) * 0.3
        combined = torch.cat([observations, language], dim=-1)
        actions = combined @ W + torch.randn(n_samples, action_dim) * 0.01
    elif task_type == "multi_step":
        W1 = torch.randn(obs_dim + lang_dim, 16) * 0.3
        W2 = torch.randn(16, 16) * 0.3
        W3 = torch.randn(16, action_dim) * 0.3
        combined = torch.cat([observations, language], dim=-1)
        h1 = F.relu(combined @ W1)
        h2 = F.relu(h1 @ W2)
        actions = h2 @ W3 + torch.randn(n_samples, action_dim) * 0.02
    
    return observations, language, actions


def measure_gradient_variance(model, train_obs, train_lang, train_actions,
                               epochs=30, lr=3e-4, batch_size=64):
    """
    Train model and measure gradient variance across epochs.
    Returns gradient statistics per epoch.
    """
    train_dataset = TensorDataset(train_obs, train_lang, train_actions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    gradient_norms = []
    gradient_variances = []
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_grad_norms = []
        
        for batch_obs, batch_lang, batch_actions in train_loader:
            optimizer.zero_grad()
            pred = model(batch_obs, batch_lang)
            loss = criterion(pred, batch_actions)
            loss.backward()
            
            # Record gradient norms
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grad_norms.append(total_norm)
            
            optimizer.step()
        
        # Epoch statistics
        mean_grad = np.mean(epoch_grad_norms)
        var_grad = np.var(epoch_grad_norms)
        gradient_norms.append(mean_grad)
        gradient_variances.append(var_grad)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(train_obs[:100], train_lang[:100])
            val_loss = criterion(val_pred, train_actions[:100]).item()
        losses.append(val_loss)
    
    return {
        "gradient_norms": gradient_norms,
        "gradient_variances": gradient_variances,
        "losses": losses,
        "mean_gradient_norm": float(np.mean(gradient_norms)),
        "std_gradient_norm": float(np.std(gradient_norms)),
        "mean_gradient_variance": float(np.mean(gradient_variances)),
        "final_loss": losses[-1]
    }


# ============================================================
# Experiment
# ============================================================

def run_experiment():
    print("=" * 60)
    print("H1.471: Gradient Variance Analysis")
    print("=" * 60)
    
    n_train = 400
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    results = {
        "hypothesis": "H1.471: Gradient Variance - Error Accumulation Mechanism",
        "prediction": "CG shows higher gradient variance on multi-step vs single-step",
        "task_types": ["single_step", "multi_step"],
        "models": ["baseline", "cg"],
        "detailed_results": {}
    }
    
    for task_type in ["single_step", "multi_step"]:
        print(f"\n--- Task: {task_type} ---")
        
        train_obs, train_lang, train_actions = generate_data(
            n_train, task_type=task_type, obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim
        )
        
        task_results = {}
        
        # Baseline
        baseline = BaselineArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim)
        baseline_stats = measure_gradient_variance(baseline, train_obs, train_lang, train_actions)
        task_results["baseline"] = baseline_stats
        print(f"  Baseline: mean_grad={baseline_stats['mean_gradient_norm']:.4f}, "
              f"std_grad={baseline_stats['std_gradient_norm']:.4f}, "
              f"final_loss={baseline_stats['final_loss']:.6f}")
        
        # CG
        cg = CognitiveGraphArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, dropout=0.4)
        cg_stats = measure_gradient_variance(cg, train_obs, train_lang, train_actions)
        task_results["cg"] = cg_stats
        print(f"  CG: mean_grad={cg_stats['mean_gradient_norm']:.4f}, "
              f"std_grad={cg_stats['std_gradient_norm']:.4f}, "
              f"final_loss={cg_stats['final_loss']:.6f}")
        
        results["detailed_results"][task_type] = task_results
    
    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    single = results["detailed_results"]["single_step"]
    multi = results["detailed_results"]["multi_step"]
    
    # Gradient variance ratio: CG vs Baseline
    cg_single_var = single["cg"]["mean_gradient_variance"]
    baseline_single_var = single["baseline"]["mean_gradient_variance"]
    cg_multi_var = multi["cg"]["mean_gradient_variance"]
    baseline_multi_var = multi["baseline"]["mean_gradient_variance"]
    
    # Does CG gradient variance increase more from single to multi?
    cg_var_increase = cg_multi_var - cg_single_var
    baseline_var_increase = baseline_multi_var - baseline_single_var
    
    # Hypothesis: CG gradient variance increases MORE on multi-step
    hypothesis_supported = cg_var_increase > baseline_var_increase
    
    results["analysis"] = {
        "cg_single_gradient_variance": cg_single_var,
        "cg_multi_gradient_variance": cg_multi_var,
        "cg_variance_increase": cg_var_increase,
        "baseline_single_gradient_variance": baseline_single_var,
        "baseline_multi_gradient_variance": baseline_multi_var,
        "baseline_variance_increase": baseline_var_increase,
        "cg_vs_baseline_variance_ratio_single": cg_single_var / baseline_single_var if baseline_single_var > 0 else 0,
        "cg_vs_baseline_variance_ratio_multi": cg_multi_var / baseline_multi_var if baseline_multi_var > 0 else 0,
        "hypothesis_supported": hypothesis_supported,
        "key_insight": (
            f"CG gradient variance increases by {cg_var_increase:.4f} from single to multi-step. "
            f"Baseline increases by {baseline_var_increase:.4f}. "
            f"{'Supports error accumulation hypothesis' if hypothesis_supported else 'Does not support error accumulation hypothesis'}"
        )
    }
    
    print(f"\nCG gradient variance: single={cg_single_var:.4f}, multi={cg_multi_var:.4f}, increase={cg_var_increase:+.4f}")
    print(f"Baseline gradient variance: single={baseline_single_var:.4f}, multi={baseline_multi_var:.4f}, increase={baseline_var_increase:+.4f}")
    print(f"Hypothesis supported: {hypothesis_supported}")
    
    output_path = "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/083-gradient_analysis/results/metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment()
