#!/usr/bin/env python3
"""
H1.470.1.1.27: Ensemble Disagreement on Complex Hierarchical Tasks (4-5 phases)
+ Adaptive Uncertainty Estimation

Context:
- H1.470.1.1.26: Ensemble disagreement REFUTED on 3-phase tasks (-4.05% vs baseline)
- Oracle noise still worked (+4.62%) suggesting true noise IS informative
- Hypothesis: 3-phase structure too simple; 4-5 phases may provide richer uncertainty signal
- Alternative: Learned uncertainty head may outperform ensemble disagreement
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"[H1.470.1.1.27] Device: {device}")

# ============================================================
# Data Generation
# ============================================================

def generate_hierarchical_task_data(n_samples=3000, n_phases=4, seq_len=20, obs_dim=128, action_dim=7, noise_level=0.1):
    observations = np.zeros((n_samples, seq_len, obs_dim), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)
    noise_levels = np.zeros((n_samples, seq_len), dtype=np.float32)
    phase_labels = np.zeros((n_samples, seq_len), dtype=np.int32)
    
    samples_per_phase = seq_len // n_phases
    remainder = seq_len % n_phases
    
    for i in range(n_samples):
        phase_dynamics = []
        for p in range(n_phases):
            phase_mean = np.random.randn(action_dim) * 0.5
            phase_std = 0.1 + np.random.rand() * 0.2
            phase_dynamics.append((phase_mean, phase_std))
        
        for t in range(seq_len):
            if t < samples_per_phase * n_phases + remainder:
                phase_idx = min(t // (samples_per_phase + (1 if t < remainder else 0)), n_phases - 1)
            else:
                phase_idx = n_phases - 1
            
            phase_labels[i, t] = phase_idx
            phase_mean, phase_std = phase_dynamics[phase_idx]
            
            obs = np.random.randn(obs_dim) * 0.3
            obs[:n_phases] = 0
            obs[phase_idx] = 1.0
            progress = (t % (samples_per_phase + (1 if t < remainder else 0))) / max(samples_per_phase, 1)
            obs[n_phases:n_phases+1] = progress
            observations[i, t] = obs
            
            action = phase_mean + np.random.randn(action_dim) * phase_std
            is_transition = (t % (samples_per_phase + (1 if t < remainder else 0)) == 0) and t > 0
            if is_transition:
                action += np.random.randn(action_dim) * noise_level * 2
                noise_levels[i, t] = noise_level * 2
            else:
                action += np.random.randn(action_dim) * noise_level * 0.5
                noise_levels[i, t] = noise_level * 0.5
            
            actions[i, t] = action
    
    return observations, actions, noise_levels, phase_labels


# ============================================================
# Models
# ============================================================

class BaselineModel(nn.Module):
    def __init__(self, obs_dim=128, action_dim=7, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.processor = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim))
    def forward(self, x):
        encoded = self.encoder(x)
        out, _ = self.processor(encoded)
        return self.decoder(out)

class EnsembleModel(nn.Module):
    def __init__(self, obs_dim=128, action_dim=7, hidden_dim=128, dropout_rate=0.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.processor = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim // 2, action_dim))
    def forward(self, x):
        encoded = self.encoder(x)
        out, _ = self.processor(encoded)
        return self.decoder(out)

class AdaptiveUncertaintyModel(nn.Module):
    def __init__(self, obs_dim=128, action_dim=7, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.processor = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim))
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim), nn.Softplus())
    def forward(self, x):
        encoded = self.encoder(x)
        out, _ = self.processor(encoded)
        return self.action_head(out), self.uncertainty_head(out)

class OracleNoiseModel(nn.Module):
    def __init__(self, obs_dim=128, action_dim=7, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.processor = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim))
    def forward(self, x):
        encoded = self.encoder(x)
        out, _ = self.processor(encoded)
        return self.decoder(out)


# ============================================================
# Training
# ============================================================

def train_baseline(model, train_loader, val_loader, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        for obs, actions, _, _ in train_loader:
            obs, actions = obs.to(device), actions.to(device)
            optimizer.zero_grad()
            pred = model(obs)
            loss = criterion(pred, actions)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, actions, _, _ in val_loader:
                obs, actions = obs.to(device), actions.to(device)
                pred = model(obs)
                val_loss += criterion(pred, actions).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return best_val_loss

def train_oracle_noise(model, train_loader, val_loader, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        for obs, actions, noise_levels, _ in train_loader:
            obs, actions = obs.to(device), actions.to(device)
            noise_levels = noise_levels.to(device)
            optimizer.zero_grad()
            pred = model(obs)
            weights = 1.0 / (1.0 + noise_levels.unsqueeze(-1) * 10)
            weights = weights / weights.mean()
            loss = torch.mean(weights * (pred - actions) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, actions, _, _ in val_loader:
                obs, actions = obs.to(device), actions.to(device)
                pred = model(obs)
                val_loss += torch.mean((pred - actions) ** 2).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return best_val_loss

def train_adaptive_uncertainty(model, train_loader, val_loader, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        for obs, actions, _, _ in train_loader:
            obs, actions = obs.to(device), actions.to(device)
            optimizer.zero_grad()
            pred, log_var = model(obs)
            loss = 0.5 * torch.mean(log_var + torch.exp(-log_var) * (pred - actions) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, actions, _, _ in val_loader:
                obs, actions = obs.to(device), actions.to(device)
                pred, log_var = model(obs)
                val_loss += (0.5 * torch.mean(log_var + torch.exp(-log_var) * (pred - actions) ** 2)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return best_val_loss

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for obs, actions, _, _ in test_loader:
            obs, actions = obs.to(device), actions.to(device)
            pred = model(obs)
            all_preds.append(pred.cpu())
            all_targets.append(actions.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return torch.mean((all_preds - all_targets) ** 2).item()

def evaluate_adaptive(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for obs, actions, _, _ in test_loader:
            obs, actions = obs.to(device), actions.to(device)
            pred, _ = model(obs)
            all_preds.append(pred.cpu())
            all_targets.append(actions.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return torch.mean((all_preds - all_targets) ** 2).item()

def train_and_evaluate_ensemble(n_models=3, train_loader=None, val_loader=None, test_loader=None, obs_dim=128, action_dim=7, hidden_dim=128, epochs=15, lr=1e-3):
    models = []
    val_losses = []
    for m in range(n_models):
        torch.manual_seed(SEED + m * 100)
        np.random.seed(SEED + m * 100)
        random.seed(SEED + m * 100)
        model = EnsembleModel(obs_dim, action_dim, hidden_dim, dropout_rate=0.1).to(device)
        vloss = train_baseline(model, train_loader, val_loader, epochs=epochs, lr=lr)
        models.append(model)
        val_losses.append(vloss)
    
    # Evaluate ensemble (simple average)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for obs, actions, _, _ in test_loader:
            obs = obs.to(device)
            preds = []
            for model in models:
                model.eval()
                preds.append(model(obs).cpu())
            ensemble_pred = torch.stack(preds, dim=0).mean(dim=0)
            all_preds.append(ensemble_pred)
            all_targets.append(actions)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    test_loss = torch.mean((all_preds - all_targets) ** 2).item()
    return models, np.mean(val_losses), test_loss


# ============================================================
# Dataset
# ============================================================

class HierarchicalTaskDataset(torch.utils.data.Dataset):
    def __init__(self, observations, actions, noise_levels, phase_labels):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        self.noise_levels = torch.FloatTensor(noise_levels)
        self.phase_labels = torch.LongTensor(phase_labels)
    def __len__(self):
        return len(self.observations)
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.noise_levels[idx], self.phase_labels[idx]


# ============================================================
# Main
# ============================================================

def run_experiment():
    print("=" * 70)
    print("H1.470.1.1.27: Ensemble Disagreement on Complex Hierarchical Tasks")
    print("=" * 70)
    
    results = {
        "experiment_id": "H1.470.1.1.27",
        "description": "Ensemble disagreement on 4-5 phase hierarchical tasks + adaptive uncertainty",
        "timestamp": datetime.now().isoformat(),
        "configurations": {},
        "key_metrics": {},
        "key_insights": [],
    }
    
    configs = [
        {"n_phases": 4, "label": "4_phase"},
        {"n_phases": 5, "label": "5_phase"},
    ]
    
    all_results = {}
    
    for config in configs:
        n_phases = config["n_phases"]
        label = config["label"]
        
        print(f"\n{'='*50}")
        print(f"Testing {label} hierarchical task")
        print(f"{'='*50}")
        
        print(f"Generating data with {n_phases} phases...")
        obs, actions, noise_levels, phase_labels = generate_hierarchical_task_data(
            n_samples=3000, n_phases=n_phases, seq_len=20, obs_dim=128, noise_level=0.1,
        )
        
        n_train, n_val, n_test = 2000, 500, 500
        train_ds = HierarchicalTaskDataset(obs[:n_train], actions[:n_train], noise_levels[:n_train], phase_labels[:n_train])
        val_ds = HierarchicalTaskDataset(obs[n_train:n_train+n_val], actions[n_train:n_train+n_val], noise_levels[n_train:n_train+n_val], phase_labels[n_train:n_train+n_val])
        test_ds = HierarchicalTaskDataset(obs[n_train+n_val:], actions[n_train+n_val:], noise_levels[n_train+n_val:], phase_labels[n_train+n_val:])
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
        
        # 1. Baseline
        print(f"  Training baseline...")
        baseline_model = BaselineModel().to(device)
        train_baseline(baseline_model, train_loader, val_loader, epochs=15, lr=1e-3)
        baseline_test_loss = evaluate_model(baseline_model, test_loader)
        print(f"  Baseline test loss: {baseline_test_loss:.6f}")
        
        # 2. Oracle noise
        print(f"  Training oracle noise model...")
        oracle_model = OracleNoiseModel().to(device)
        train_oracle_noise(oracle_model, train_loader, val_loader, epochs=15, lr=1e-3)
        oracle_test_loss = evaluate_model(oracle_model, test_loader)
        oracle_improvement = (baseline_test_loss - oracle_test_loss) / baseline_test_loss * 100
        print(f"  Oracle test loss: {oracle_test_loss:.6f} ({oracle_improvement:+.2f}%)")
        
        # 3. Ensemble disagreement
        print(f"  Training ensemble (3 models)...")
        ensemble_models, _, ensemble_test_loss = train_and_evaluate_ensemble(
            n_models=3, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            epochs=15, lr=1e-3,
        )
        ensemble_improvement = (baseline_test_loss - ensemble_test_loss) / baseline_test_loss * 100
        print(f"  Ensemble test loss: {ensemble_test_loss:.6f} ({ensemble_improvement:+.2f}%)")
        
        # 4. Adaptive uncertainty
        print(f"  Training adaptive uncertainty model...")
        adaptive_model = AdaptiveUncertaintyModel().to(device)
        train_adaptive_uncertainty(adaptive_model, train_loader, val_loader, epochs=15, lr=1e-3)
        adaptive_test_loss = evaluate_adaptive(adaptive_model, test_loader)
        adaptive_improvement = (baseline_test_loss - adaptive_test_loss) / baseline_test_loss * 100
        print(f"  Adaptive uncertainty test loss: {adaptive_test_loss:.6f} ({adaptive_improvement:+.2f}%)")
        
        ensemble_oracle_ratio = ensemble_improvement / oracle_improvement * 100 if oracle_improvement != 0 else float('inf')
        
        all_results[label] = {
            "n_phases": n_phases,
            "baseline_test_loss": baseline_test_loss,
            "oracle_test_loss": oracle_test_loss,
            "oracle_improvement": oracle_improvement,
            "ensemble_test_loss": ensemble_test_loss,
            "ensemble_improvement": ensemble_improvement,
            "ensemble_oracle_ratio": ensemble_oracle_ratio,
            "adaptive_test_loss": adaptive_test_loss,
            "adaptive_improvement": adaptive_improvement,
        }
        
        print(f"\n  Summary for {label}:")
        print(f"    Baseline: {baseline_test_loss:.6f}")
        print(f"    Oracle:   {oracle_test_loss:.6f} ({oracle_improvement:+.2f}%)")
        print(f"    Ensemble: {ensemble_test_loss:.6f} ({ensemble_improvement:+.2f}%)")
        print(f"    Adaptive: {adaptive_test_loss:.6f} ({adaptive_improvement:+.2f}%)")
    
    # ============================================================
    # Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    ensemble_4ph = all_results["4_phase"]["ensemble_improvement"]
    ensemble_5ph = all_results["5_phase"]["ensemble_improvement"]
    adaptive_4ph = all_results["4_phase"]["adaptive_improvement"]
    adaptive_5ph = all_results["5_phase"]["adaptive_improvement"]
    prior_ensemble_improvement = -4.05
    
    print(f"\nEnsemble disagreement improvements:")
    print(f"  3-phase (prior): {prior_ensemble_improvement:+.2f}%")
    print(f"  4-phase: {ensemble_4ph:+.2f}%")
    print(f"  5-phase: {ensemble_5ph:+.2f}%")
    
    print(f"\nAdaptive uncertainty improvements:")
    print(f"  4-phase: {adaptive_4ph:+.2f}%")
    print(f"  5-phase: {adaptive_5ph:+.2f}%")
    
    insights = []
    
    if ensemble_5ph > ensemble_4ph > prior_ensemble_improvement:
        insights.append("Ensemble disagreement improves monotonically with phase complexity")
        ensemble_trend = "improving"
    elif ensemble_5ph > prior_ensemble_improvement:
        insights.append("Ensemble disagreement improves at 5 phases vs 3-phase prior")
        ensemble_trend = "improving_at_5"
    else:
        insights.append("Ensemble disagreement does not improve with more phases")
        ensemble_trend = "not_improving"
    
    if adaptive_4ph > ensemble_4ph and adaptive_5ph > ensemble_5ph:
        insights.append("Adaptive uncertainty head consistently outperforms ensemble disagreement")
        adaptive_wins = True
    else:
        insights.append("Adaptive uncertainty does not consistently outperform ensemble")
        adaptive_wins = False
    
    oracle_4ph = all_results["4_phase"]["oracle_improvement"]
    oracle_5ph = all_results["5_phase"]["oracle_improvement"]
    insights.append(f"Oracle noise: 4-phase {oracle_4ph:+.2f}%, 5-phase {oracle_5ph:+.2f}%")
    
    if ensemble_5ph > 0 and ensemble_5ph > prior_ensemble_improvement:
        conclusion = "SUPPORTED"
        insights.append("H1.470.1.1.27 SUPPORTED: 5-phase tasks show positive ensemble disagreement improvement")
    elif ensemble_5ph > prior_ensemble_improvement:
        conclusion = "PARTIAL"
        insights.append("H1.470.1.1.27 PARTIAL: 5-phase tasks show improvement over 3-phase but still negative")
    else:
        conclusion = "REFUTED"
        insights.append("H1.470.1.1.27 REFUTED: More phases do not help ensemble disagreement")
    
    key_metrics = {
        "prior_3phase_ensemble_improvement": prior_ensemble_improvement,
        "4phase_baseline_loss": all_results["4_phase"]["baseline_test_loss"],
        "4phase_oracle_improvement": all_results["4_phase"]["oracle_improvement"],
        "4phase_ensemble_improvement": all_results["4_phase"]["ensemble_improvement"],
        "4phase_adaptive_improvement": all_results["4_phase"]["adaptive_improvement"],
        "5phase_baseline_loss": all_results["5_phase"]["baseline_test_loss"],
        "5phase_oracle_improvement": all_results["5_phase"]["oracle_improvement"],
        "5phase_ensemble_improvement": all_results["5_phase"]["ensemble_improvement"],
        "5phase_adaptive_improvement": all_results["5_phase"]["adaptive_improvement"],
        "ensemble_trend": ensemble_trend,
        "adaptive_wins": adaptive_wins,
        "best_config": max(
            [("4ph_ensemble", ensemble_4ph), ("5ph_ensemble", ensemble_5ph),
             ("4ph_adaptive", adaptive_4ph), ("5ph_adaptive", adaptive_5ph)],
            key=lambda x: x[1]
        )[0],
        "best_improvement": max(ensemble_4ph, ensemble_5ph, adaptive_4ph, adaptive_5ph),
    }
    
    results["conclusion"] = conclusion
    results["key_metrics"] = key_metrics
    results["key_insights"] = insights
    results["configurations"] = all_results
    
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Conclusion: {conclusion}")
    print(f"Best config: {key_metrics['best_config']} ({key_metrics['best_improvement']:+.2f}%)")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
