#!/usr/bin/env python3
"""
H1.470.1.1.13: Why Does CG Underperform? Lightweight CG Variants

Hypothesis: CG's poor performance is due to parameter budget mismatch and 
architectural complexity, not the unified representation concept itself.

Prediction: Lightweight CG variants with reduced dimensions (matching LSTM's 
~358K parameter budget) will perform better than the bloated 1.995M param CG.

Test: Compare multiple lightweight CG variants against baseline and LSTM:
1. CG-tiny: 64-dim unified space (32 physical + 32 semantic), 1 GNN layer
2. CG-small: 128-dim unified space (64 physical + 64 semantic), 2 GNN layers  
3. CG-medium: 256-dim unified space (128 physical + 128 semantic), 2 GNN layers
4. CG-noGNN: 128-dim unified space, no GNN layers (just projection + fusion)
5. CG-attention: 128-dim unified space, cross-attention instead of GNN
6. LSTM: standard LSTM (control, ~358K params)
7. Baseline: separate encoders + concatenation (~61K params)

Test on 3 task types: temporal-only, cross-modal-only, combined
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation
# ============================================================

def generate_task_data(n_samples=2000, seq_len=10, task_type="combined"):
    """
    Generate synthetic robot manipulation data.
    
    task_type: "temporal_only", "crossmodal_only", "combined"
    """
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    # Generate observations (physical state)
    obs = torch.randn(n_samples, seq_len, obs_dim)
    
    # Generate language embeddings (semantic)
    lang = torch.randn(n_samples, lang_dim)
    
    # Generate actions based on task type
    if task_type == "temporal_only":
        # Actions depend on temporal patterns in observations
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            actions[:, t, :] = (
                0.5 * obs[:, t, :7] + 
                0.3 * obs[:, max(0,t-1), :7] +
                0.2 * torch.randn(n_samples, action_dim) * 0.1
            )
    elif task_type == "crossmodal_only":
        # Actions depend on language-observation alignment
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            lang_expanded = lang.unsqueeze(1).expand(-1, seq_len, -1)
            actions[:, t, :] = (
                0.6 * torch.matmul(obs[:, t, :7], torch.randn(7, action_dim) * 0.3) +
                0.4 * lang[:, :action_dim] +
                torch.randn(n_samples, action_dim) * 0.1
            )
    else:  # combined
        # Actions depend on both temporal patterns AND language grounding
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            actions[:, t, :] = (
                0.3 * obs[:, t, :7] +
                0.2 * obs[:, max(0,t-1), :7] +
                0.3 * lang[:, :action_dim] +
                0.2 * torch.randn(n_samples, action_dim) * 0.1
            )
    
    return obs, lang, actions


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Separate encoders + concatenation."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=64):
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
        # obs: (batch, seq, obs_dim), lang: (batch, lang_dim)
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang_expanded)
        fused = torch.cat([obs_enc, lang_enc], dim=-1)
        return self.fusion(fused)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LSTMArchitecture(nn.Module):
    """Standard LSTM for temporal processing."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang_expanded)
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.output(lstm_out)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class CGTiny(nn.Module):
    """CG-tiny: 64-dim unified space (32+32), 1 GNN layer."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=32, semantic_dim=32):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            )
        ])
        
        self.output = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        
        physical = self.obs_to_physical(obs)
        semantic = self.lang_to_semantic(lang_expanded)
        
        unified = torch.cat([physical, semantic], dim=-1)
        
        for gnn in self.gnn_layers:
            unified = unified + gnn(unified)
        
        return self.output(unified)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class CGSmall(nn.Module):
    """CG-small: 128-dim unified space (64+64), 2 GNN layers."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=64, semantic_dim=64):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(2)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        
        physical = self.obs_to_physical(obs)
        semantic = self.lang_to_semantic(lang_expanded)
        
        unified = torch.cat([physical, semantic], dim=-1)
        
        for gnn in self.gnn_layers:
            unified = unified + gnn(unified)
        
        return self.output(unified)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class CGMedium(nn.Module):
    """CG-medium: 256-dim unified space (128+128), 2 GNN layers."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=128, semantic_dim=128):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(2)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        
        physical = self.obs_to_physical(obs)
        semantic = self.lang_to_semantic(lang_expanded)
        
        unified = torch.cat([physical, semantic], dim=-1)
        
        for gnn in self.gnn_layers:
            unified = unified + gnn(unified)
        
        return self.output(unified)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class CGNoGNN(nn.Module):
    """CG-noGNN: 128-dim unified space, no GNN layers (just projection + fusion)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=64, semantic_dim=64):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # No GNN - just direct fusion
        self.output = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        
        physical = self.obs_to_physical(obs)
        semantic = self.lang_to_semantic(lang_expanded)
        
        unified = torch.cat([physical, semantic], dim=-1)
        
        return self.output(unified)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class CGAttention(nn.Module):
    """CG-attention: 128-dim unified space, cross-attention instead of GNN."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=64, semantic_dim=64, n_heads=4):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Cross-attention: physical attends to semantic and vice versa
        self.physical_attn = nn.MultiheadAttention(
            embed_dim=physical_dim, num_heads=n_heads, batch_first=True
        )
        self.semantic_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim, num_heads=n_heads, batch_first=True
        )
        
        self.output = nn.Sequential(
            nn.Linear(physical_dim + semantic_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        
        physical = self.obs_to_physical(obs)  # (batch, seq, physical_dim)
        semantic = self.lang_to_semantic(lang_expanded)  # (batch, seq, semantic_dim)
        
        # Cross-attention
        physical_attn_out, _ = self.physical_attn(
            physical, semantic, semantic
        )
        semantic_attn_out, _ = self.semantic_attn(
            semantic, physical, physical
        )
        
        physical = physical + physical_attn_out
        semantic = semantic + semantic_attn_out
        
        unified = torch.cat([physical, semantic], dim=-1)
        
        return self.output(unified)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Training Loop
# ============================================================

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Train model and return final validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for obs_batch, lang_batch, action_batch in train_loader:
            optimizer.zero_grad()
            pred = model(obs_batch, lang_batch)
            loss = F.mse_loss(pred, action_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs_batch, lang_batch, action_batch in val_loader:
                pred = model(obs_batch, lang_batch)
                loss = F.mse_loss(pred, action_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_experiment():
    """Run the full experiment."""
    print("=" * 60)
    print("H1.470.1.1.13: Lightweight CG Variants")
    print("=" * 60)
    
    task_types = ["temporal_only", "crossmodal_only", "combined"]
    
    # Define architectures to test
    architectures = {
        "baseline": lambda: BaselineArchitecture(),
        "lstm": lambda: LSTMArchitecture(),
        "cg_tiny": lambda: CGTiny(),
        "cg_small": lambda: CGSmall(),
        "cg_medium": lambda: CGMedium(),
        "cg_noGNN": lambda: CGNoGNN(),
        "cg_attention": lambda: CGAttention(),
    }
    
    results = {
        "hypothesis": "H1.470.1.1.13: Lightweight CG variants match LSTM performance",
        "prediction": "Reduced-dimension CG variants will perform better than bloated CG",
        "task_types": task_types,
        "architectures": list(architectures.keys()),
        "detailed_results": {},
        "parameter_counts": {},
        "analysis": {}
    }
    
    # First, count parameters for all architectures
    print("\nParameter counts:")
    for name, arch_fn in architectures.items():
        model = arch_fn()
        n_params = model.count_params()
        results["parameter_counts"][name] = n_params
        print(f"  {name}: {n_params:,} params")
    
    # Run experiments for each task type
    for task_type in task_types:
        print(f"\n{'='*40}")
        print(f"Task: {task_type}")
        print(f"{'='*40}")
        
        # Generate data
        obs, lang, actions = generate_task_data(n_samples=2000, seq_len=10, task_type=task_type)
        
        # Split into train/val
        n_train = 1600
        n_val = 400
        
        train_obs = obs[:n_train]
        train_lang = lang[:n_train]
        train_actions = actions[:n_train]
        
        val_obs = obs[n_train:]
        val_lang = lang[n_train:]
        val_actions = actions[n_train:]
        
        train_dataset = TensorDataset(train_obs, train_lang, train_actions)
        val_dataset = TensorDataset(val_obs, val_lang, val_actions)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        results["detailed_results"][task_type] = {}
        
        for name, arch_fn in architectures.items():
            print(f"\n  Training {name}...")
            model = arch_fn()
            n_params = model.count_params()
            
            # Adjust learning rate based on model size
            if n_params > 500000:
                lr = 0.0005
            elif n_params > 100000:
                lr = 0.001
            else:
                lr = 0.002
            
            val_loss = train_model(model, train_loader, val_loader, epochs=30, lr=lr)
            
            results["detailed_results"][task_type][name] = {
                "loss": val_loss,
                "params": n_params
            }
            
            print(f"    {name}: val_loss={val_loss:.6f}, params={n_params:,}")
    
    # Compute improvements vs baseline
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    
    improvements = {}
    for task_type in task_types:
        baseline_loss = results["detailed_results"][task_type]["baseline"]["loss"]
        improvements[task_type] = {}
        
        print(f"\n{task_type} (baseline loss: {baseline_loss:.6f}):")
        for name in architectures.keys():
            loss = results["detailed_results"][task_type][name]["loss"]
            improvement = ((baseline_loss - loss) / baseline_loss) * 100
            improvements[task_type][name] = improvement
            params = results["detailed_results"][task_type][name]["params"]
            print(f"  {name:15s}: loss={loss:.6f}, improvement={improvement:+.2f}%, params={params:,}")
    
    results["improvements"] = improvements
    
    # Analysis
    lstm_params = results["parameter_counts"]["lstm"]
    
    # Find best lightweight CG variant
    best_lightweight = None
    best_lightweight_score = -float('inf')
    
    lightweight_variants = ["cg_tiny", "cg_small", "cg_medium", "cg_noGNN", "cg_attention"]
    
    for variant in lightweight_variants:
        # Average improvement across all tasks
        avg_improvement = np.mean([improvements[t][variant] for t in task_types])
        if avg_improvement > best_lightweight_score:
            best_lightweight_score = avg_improvement
            best_lightweight = variant
    
    # Compare best lightweight CG to LSTM
    lstm_avg_improvement = np.mean([improvements[t]["lstm"] for t in task_types])
    
    results["analysis"] = {
        "best_lightweight_cg": best_lightweight,
        "best_lightweight_avg_improvement": best_lightweight_score,
        "lstm_avg_improvement": lstm_avg_improvement,
        "lightweight_vs_lstm_gap": best_lightweight_score - lstm_avg_improvement,
        "lstm_params": lstm_params,
        "key_insight": "",
        "hypothesis_supported": False
    }
    
    # Determine if hypothesis is supported
    if best_lightweight_score > lstm_avg_improvement * 0.8:  # Within 80% of LSTM
        results["analysis"]["hypothesis_supported"] = True
        results["analysis"]["key_insight"] = (
            f"Lightweight CG ({best_lightweight}) achieves {best_lightweight_score:.2f}% avg improvement, "
            f"within {abs(best_lightweight_score - lstm_avg_improvement):.2f}% of LSTM ({lstm_avg_improvement:.2f}%). "
            f"CG concept is viable when parameter budget is controlled."
        )
    else:
        results["analysis"]["key_insight"] = (
            f"Even lightweight CG variants underperform LSTM significantly. "
            f"Best lightweight CG ({best_lightweight}): {best_lightweight_score:.2f}%, "
            f"LSTM: {lstm_avg_improvement:.2f}%. "
            f"The unified representation concept itself may be fundamentally flawed for these tasks."
        )
    
    # Save results
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/083-lightweight_cg/results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Experiment complete. Results saved.")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
