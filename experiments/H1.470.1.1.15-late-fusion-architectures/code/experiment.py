#!/usr/bin/env python3
"""
H1.470.1.1.15: Late-Fusion Architecture Test (Fast Version)

Hypothesis: Based on H1.470.1.1.14 findings, the optimal architecture should be:
  separated encoders → temporal processing → late concatenation

This "late fusion" approach should:
1. Preserve the benefits of separated encoding (no cross-modal interference)
2. Add temporal processing to each modality independently
3. Fuse at the final stage for action prediction

Prediction: Late-fusion will outperform both:
- Baseline (no temporal processing)
- LSTM (early fusion with recurrence)
- CG (unified representation)

Test architectures:
1. Baseline: separate encoders → concat → output (no temporal)
2. LSTM-early: separate encoders → concat → LSTM → output (early fusion)
3. LSTM-late: separate encoders → LSTM each → concat → output (late fusion)
4. Temporal-conv-early: separate encoders → concat → 1D conv → output
5. Temporal-conv-late: separate encoders → 1D conv each → concat → output
6. CG: unified encoder → GNN → output (reference)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Architectures
# ============================================================

class Baseline(nn.Module):
    """Separate encoders → concat → output (no temporal processing)"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        B, T, _ = obs.shape
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, T, -1)
        fused = torch.cat([obs_encoded, lang_encoded], dim=-1)
        return self.head(fused)


class LSTMEarlyFusion(nn.Module):
    """Separate encoders → concat → LSTM → output (early fusion)"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, lang):
        B, T, _ = obs.shape
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, T, -1)
        fused = torch.cat([obs_encoded, lang_encoded], dim=-1)
        lstm_out, _ = self.lstm(fused)
        return self.head(lstm_out)


class LSTMLateFusion(nn.Module):
    """Separate encoders → LSTM each → concat → output (late fusion)"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.obs_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lang_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        B, T, _ = obs.shape
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, T, -1)
        
        obs_temporal, _ = self.obs_lstm(obs_encoded)
        lang_temporal, _ = self.lang_lstm(lang_encoded)
        
        fused = torch.cat([obs_temporal, lang_temporal], dim=-1)
        return self.head(fused)


class TemporalConvEarlyFusion(nn.Module):
    """Separate encoders → concat → 1D conv → output (early fusion with parallel temporal)"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.temporal_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, lang):
        B, T, _ = obs.shape
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, T, -1)
        fused = torch.cat([obs_encoded, lang_encoded], dim=-1)
        
        fused_t = fused.transpose(1, 2)
        temporal_out = self.temporal_conv(fused_t).transpose(1, 2)
        
        return self.head(temporal_out)


class TemporalConvLateFusion(nn.Module):
    """Separate encoders → 1D conv each → concat → output (late fusion with parallel temporal)"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.obs_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.lang_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        B, T, _ = obs.shape
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, T, -1)
        
        obs_temporal = self.obs_conv(obs_encoded.transpose(1, 2)).transpose(1, 2)
        lang_temporal = self.lang_conv(lang_encoded.transpose(1, 2)).transpose(1, 2)
        
        fused = torch.cat([obs_temporal, lang_temporal], dim=-1)
        return self.head(fused)


class CognitiveGraph(nn.Module):
    """Unified encoder → GNN → output (reference architecture)"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        total_dim = hidden_dim * 2
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(2)
        ])
        self.head = nn.Linear(total_dim, action_dim)
    
    def forward(self, obs, lang):
        B, T, _ = obs.shape
        obs_unified = self.obs_to_unified(obs)
        lang_unified = self.lang_to_unified(lang).unsqueeze(1).expand(-1, T, -1)
        
        unified = torch.cat([obs_unified, lang_unified], dim=-1)
        
        for gnn_layer in self.gnn_layers:
            unified = unified + gnn_layer(unified)
        
        return self.head(unified)


# ============================================================
# Data Generation
# ============================================================

def generate_task_data(n_samples=1000, seq_len=10, task_type="combined"):
    """Generate synthetic robot manipulation data."""
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    obs = torch.randn(n_samples, seq_len, obs_dim)
    lang = torch.randn(n_samples, lang_dim)
    
    if task_type == "temporal_only":
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            actions[:, t, :] = (
                0.5 * obs[:, t, :7] + 
                0.3 * obs[:, max(0,t-1), :7] +
                0.2 * torch.randn(n_samples, action_dim) * 0.1
            )
    elif task_type == "crossmodal_only":
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            actions[:, t, :] = (
                0.6 * torch.matmul(obs[:, t, :7], torch.randn(7, action_dim) * 0.3) +
                0.4 * lang[:, :action_dim] +
                torch.randn(n_samples, action_dim) * 0.1
            )
    else:  # combined
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            actions[:, t, :] = (
                0.3 * obs[:, t, :7] +
                0.2 * obs[:, max(0,t-1), :7] +
                0.3 * lang[:, :action_dim] +
                0.2 * torch.randn(n_samples, action_dim) * 0.1
            )
    
    return obs, lang, actions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, train_loader, val_loader, epochs=30, lr=1e-3):
    """Train model and return best validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for obs, lang, actions in train_loader:
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, actions)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, lang, actions in val_loader:
                pred = model(obs, lang)
                loss = criterion(pred, actions)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    return best_val_loss


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 60)
    print("H1.470.1.1.15: Late-Fusion Architecture Test")
    print("=" * 60)
    
    architectures = {
        "baseline": Baseline,
        "lstm_early": LSTMEarlyFusion,
        "lstm_late": LSTMLateFusion,
        "tempconv_early": TemporalConvEarlyFusion,
        "tempconv_late": TemporalConvLateFusion,
        "cognitive_graph": CognitiveGraph,
    }
    
    task_types = ["temporal_only", "crossmodal_only", "combined"]
    
    results = {
        "hypothesis": "H1.470.1.1.15: Late-fusion (separate temporal processing per modality) outperforms early fusion",
        "prediction": "Late-fusion architectures will match or exceed early fusion on all task types",
        "task_types": task_types,
        "architectures": list(architectures.keys()),
        "detailed_results": {},
        "parameter_counts": {},
    }
    
    for task_type in task_types:
        print(f"\n--- Task Type: {task_type} ---")
        results["detailed_results"][task_type] = {}
        
        obs, lang, actions = generate_task_data(n_samples=1000, seq_len=10, task_type=task_type)
        dataset = TensorDataset(obs, lang, actions)
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        for arch_name, ArchClass in architectures.items():
            print(f"  Training {arch_name}...", end=" ", flush=True)
            model = ArchClass()
            results["parameter_counts"][arch_name] = count_parameters(model)
            
            val_loss = train_and_evaluate(model, train_loader, val_loader, epochs=30)
            results["detailed_results"][task_type][arch_name] = {
                "loss": val_loss,
                "params": count_parameters(model)
            }
            print(f"Loss: {val_loss:.6f}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    baseline_losses = {tt: results["detailed_results"][tt]["baseline"]["loss"] for tt in task_types}
    
    improvements = {}
    for arch_name in architectures:
        improvements[arch_name] = {}
        for tt in task_types:
            arch_loss = results["detailed_results"][tt][arch_name]["loss"]
            baseline_loss = baseline_losses[tt]
            improvement = (baseline_loss - arch_loss) / baseline_loss * 100
            improvements[arch_name][tt] = improvement
    
    results["improvements_vs_baseline"] = improvements
    
    print("\nImprovement vs Baseline (%):")
    print(f"{'Architecture':<20} {'Temporal':>12} {'Crossmodal':>12} {'Combined':>12}")
    print("-" * 60)
    for arch_name in architectures:
        imp = improvements[arch_name]
        print(f"{arch_name:<20} {imp['temporal_only']:>11.2f}% {imp['crossmodal_only']:>11.2f}% {imp['combined']:>11.2f}%")
    
    print("\n--- Late Fusion vs Early Fusion ---")
    for task_type in task_types:
        lstm_late_imp = improvements["lstm_late"][task_type]
        lstm_early_imp = improvements["lstm_early"][task_type]
        tempconv_late_imp = improvements["tempconv_late"][task_type]
        tempconv_early_imp = improvements["tempconv_early"][task_type]
        
        print(f"\n{task_type}:")
        print(f"  LSTM: Late={lstm_late_imp:.2f}% vs Early={lstm_early_imp:.2f}% (diff={lstm_late_imp - lstm_early_imp:.2f}%)")
        print(f"  TempConv: Late={tempconv_late_imp:.2f}% vs Early={tempconv_early_imp:.2f}% (diff={tempconv_late_imp - tempconv_early_imp:.2f}%)")
    
    results["analysis"] = {
        "lstm_late_vs_early": {
            tt: improvements["lstm_late"][tt] - improvements["lstm_early"][tt] for tt in task_types
        },
        "tempconv_late_vs_early": {
            tt: improvements["tempconv_late"][tt] - improvements["tempconv_early"][tt] for tt in task_types
        },
        "best_architecture_per_task": {
            tt: max(architectures.keys(), key=lambda a: improvements[a][tt]) for tt in task_types
        }
    }
    
    # Save results
    with open("experiments/H1.470.1.1.15-late-fusion-architectures/results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/H1.470.1.1.15-late-fusion-architectures/results/metrics.json")
    
    return results


if __name__ == "__main__":
    main()