#!/usr/bin/env python3
"""
H1.470: Error Accumulation in Unified Representations

Hypothesis: CG's advantage decreases with task complexity because errors in the 
unified representation space accumulate across steps. In single-step tasks, the 
cross-modal grounding helps. In multi-step tasks, small errors in the unified 
space compound, degrading performance.

Prediction: Adding explicit error correction (residual connections between steps)
will reduce the performance gap between single-step and multi-step tasks for CG.

Test: Compare standard CG vs CG-with-residual-correction on single-step vs 3-step.
If error accumulation is the issue, residual correction should help multi-step more.
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


class CognitiveGraphStandard(nn.Module):
    """Standard CG architecture (from H1.469)."""
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


class CognitiveGraphResidual(nn.Module):
    """CG with residual error correction for multi-step tasks."""
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
        
        # Residual correction layer
        self.residual_correction = nn.Sequential(
            nn.Linear(total_dim, total_dim), nn.ReLU(),
            nn.Linear(total_dim, total_dim)
        )
        
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
        
        # Apply residual correction
        corrected = attn_out + self.residual_correction(attn_out)
        
        return self.decoder(corrected.mean(dim=1))


# ============================================================
# Data Generation - Matching H1.469 setup
# ============================================================

def generate_libero_style_data(n_samples, task_type="single_step", obs_dim=8, lang_dim=32, action_dim=7):
    """
    Generate data matching LIBERO-style task structure.
    
    Single-step: pick-and-place (one action)
    Multi-step: pick-then-place-then-adjust (3 actions chained)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    observations = torch.randn(n_samples, obs_dim)
    language = torch.randn(n_samples, lang_dim)
    
    if task_type == "single_step":
        # Simple pick-and-place: direct mapping
        W = torch.randn(obs_dim + lang_dim, action_dim) * 0.3
        combined = torch.cat([observations, language], dim=-1)
        actions = combined @ W + torch.randn(n_samples, action_dim) * 0.01
        
    elif task_type == "multi_step":
        # 3-step task: each step depends on previous
        # This creates error accumulation potential
        W1 = torch.randn(obs_dim + lang_dim, 16) * 0.3
        W2 = torch.randn(16, 16) * 0.3
        W3 = torch.randn(16, action_dim) * 0.3
        
        combined = torch.cat([observations, language], dim=-1)
        h1 = F.relu(combined @ W1)
        h2 = F.relu(h1 @ W2)
        actions = h2 @ W3 + torch.randn(n_samples, action_dim) * 0.02
        
    return observations, language, actions


def train_and_eval(model, train_obs, train_lang, train_actions,
                   val_obs, val_lang, val_actions,
                   epochs=80, lr=3e-4, batch_size=64):
    """Train model and return validation loss."""
    train_dataset = TensorDataset(train_obs, train_lang, train_actions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_obs, batch_lang, batch_actions in train_loader:
            optimizer.zero_grad()
            pred = model(batch_obs, batch_lang)
            loss = criterion(pred, batch_actions)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
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
    
    # Load best and evaluate
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(val_obs, val_lang)
        val_loss = criterion(val_pred, val_actions).item()
    
    return val_loss


# ============================================================
# Experiment
# ============================================================

def run_experiment():
    print("=" * 60)
    print("H1.470: Error Accumulation in Unified Representations")
    print("=" * 60)
    
    n_train = 400
    n_val = 100
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    results = {
        "hypothesis": "H1.470: Error Accumulation",
        "prediction": "Residual correction reduces the single-to-multi performance gap for CG",
        "task_types": ["single_step", "multi_step"],
        "models": ["baseline", "cg_standard", "cg_residual"],
        "detailed_results": {}
    }
    
    for task_type in ["single_step", "multi_step"]:
        print(f"\n--- Task: {task_type} ---")
        
        train_obs, train_lang, train_actions = generate_libero_style_data(
            n_train, task_type=task_type, obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim
        )
        val_obs, val_lang, val_actions = generate_libero_style_data(
            n_val, task_type=task_type, obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim
        )
        
        task_results = {}
        
        # 1. Baseline
        baseline = BaselineArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim)
        baseline_loss = train_and_eval(baseline, train_obs, train_lang, train_actions,
                                       val_obs, val_lang, val_actions)
        task_results["baseline"] = {"loss": baseline_loss}
        print(f"  Baseline loss: {baseline_loss:.6f}")
        
        # 2. CG Standard
        cg_std = CognitiveGraphStandard(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, dropout=0.4)
        cg_std_loss = train_and_eval(cg_std, train_obs, train_lang, train_actions,
                                     val_obs, val_lang, val_actions)
        cg_std_imp = (baseline_loss - cg_std_loss) / baseline_loss * 100
        task_results["cg_standard"] = {
            "loss": cg_std_loss,
            "improvement_vs_baseline": round(cg_std_imp, 2)
        }
        print(f"  CG Standard loss: {cg_std_loss:.6f} ({cg_std_imp:+.2f}%)")
        
        # 3. CG Residual
        cg_res = CognitiveGraphResidual(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, dropout=0.4)
        cg_res_loss = train_and_eval(cg_res, train_obs, train_lang, train_actions,
                                     val_obs, val_lang, val_actions)
        cg_res_imp = (baseline_loss - cg_res_loss) / baseline_loss * 100
        task_results["cg_residual"] = {
            "loss": cg_res_loss,
            "improvement_vs_baseline": round(cg_res_imp, 2)
        }
        print(f"  CG Residual loss: {cg_res_loss:.6f} ({cg_res_imp:+.2f}%)")
        
        results["detailed_results"][task_type] = task_results
    
    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    single = results["detailed_results"]["single_step"]
    multi = results["detailed_results"]["multi_step"]
    
    cg_std_single_imp = single["cg_standard"]["improvement_vs_baseline"]
    cg_std_multi_imp = multi["cg_standard"]["improvement_vs_baseline"]
    cg_std_drop = cg_std_multi_imp - cg_std_single_imp
    
    cg_res_single_imp = single["cg_residual"]["improvement_vs_baseline"]
    cg_res_multi_imp = multi["cg_residual"]["improvement_vs_baseline"]
    cg_res_drop = cg_res_multi_imp - cg_res_single_imp
    
    # Does residual correction help multi-step more than single-step?
    error_accumulation_evidence = cg_res_drop > cg_std_drop
    
    results["analysis"] = {
        "cg_standard_single_step_improvement": cg_std_single_imp,
        "cg_standard_multi_step_improvement": cg_std_multi_imp,
        "cg_standard_improvement_drop": round(cg_std_drop, 2),
        "cg_residual_single_step_improvement": cg_res_single_imp,
        "cg_residual_multi_step_improvement": cg_res_multi_imp,
        "cg_residual_improvement_drop": round(cg_res_drop, 2),
        "error_accumulation_evidence": error_accumulation_evidence,
        "hypothesis_supported": error_accumulation_evidence,
        "key_insight": (
            f"Residual correction changes improvement drop from {cg_std_drop:.2f}% to {cg_res_drop:.2f}%. "
            f"{'Supports error accumulation hypothesis' if error_accumulation_evidence else 'Does not support error accumulation hypothesis'}"
        )
    }
    
    print(f"\nCG Standard: single={cg_std_single_imp:+.2f}%, multi={cg_std_multi_imp:+.2f}%, drop={cg_std_drop:+.2f}%")
    print(f"CG Residual: single={cg_res_single_imp:+.2f}%, multi={cg_res_multi_imp:+.2f}%, drop={cg_res_drop:+.2f}%")
    print(f"Error accumulation evidence: {error_accumulation_evidence}")
    print(f"Hypothesis supported: {error_accumulation_evidence}")
    
    output_path = "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/082-representation_interference/results/metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment()
