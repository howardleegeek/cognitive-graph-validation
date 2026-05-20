#!/usr/bin/env python3
"""
H1.470.1.1.1: Finer sweep around 832 dimensions

Hypothesis: 832 is the optimal representation dimension for CG on multi-step tasks.
Prediction: A finer sweep [800, 816, 832, 848, 864] will confirm 832 as the peak.

Fast version: reduced epochs, smaller data, 2 runs per dimension.
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
    def __init__(self, obs_dim=8, lang_dim=32, output_dim=7, latent_dim=128):
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
            nn.Linear(64, output_dim)
        )

    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, output_dim=7, total_dim=832, dropout=0.2):
        super().__init__()
        self.physical_dim = int(total_dim * 0.28)
        self.semantic_dim = total_dim - self.physical_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, self.physical_dim), nn.LayerNorm(self.physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, self.semantic_dim), nn.LayerNorm(self.semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(2)
        ])
        
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, obs, lang):
        z_phys = self.obs_to_physical(obs)
        z_sem = self.lang_to_semantic(lang)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


# ============================================================
# Data generation
# ============================================================

def generate_multi_step_data(n_samples, n_steps=3, obs_dim=8, lang_dim=32, action_dim=7, noise=0.1):
    observations = torch.randn(n_samples, obs_dim)
    language = torch.randn(n_samples, lang_dim)
    
    actions = torch.zeros(n_samples, n_steps, action_dim)
    current_obs = observations.clone()
    
    for step in range(n_steps):
        combined = torch.cat([current_obs, language], dim=-1)
        actions[:, step, :] = torch.tanh(
            torch.sin(combined[:, :action_dim]) * 0.5 +
            torch.cos(combined[:, action_dim:action_dim*2] if combined.size(-1) >= action_dim*2 else combined[:, :action_dim]) * 0.3 +
            torch.randn(n_samples, action_dim) * noise
        )
        action_update = torch.zeros(n_samples, obs_dim)
        action_update[:, :action_dim] = actions[:, step, :]
        current_obs = current_obs + action_update * 0.1 + torch.randn(n_samples, obs_dim) * 0.05
    
    # Flatten: (n_samples, n_steps * action_dim)
    actions = actions.view(n_samples, -1)
    
    return observations, language, actions


def generate_single_step_data(n_samples, obs_dim=8, lang_dim=32, action_dim=7, noise=0.1):
    observations = torch.randn(n_samples, obs_dim)
    language = torch.randn(n_samples, lang_dim)
    
    combined = torch.cat([observations, language], dim=-1)
    actions = torch.tanh(
        torch.sin(combined[:, :action_dim]) * 0.5 +
        torch.cos(combined[:, action_dim:action_dim*2] if combined.size(-1) >= action_dim*2 else combined[:, :action_dim]) * 0.3 +
        torch.randn(n_samples, action_dim) * noise
    )
    
    return observations, language, actions


class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, observations, language, actions):
        self.observations = observations
        self.language = language
        self.actions = actions
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observation': self.observations[idx],
            'language': self.language[idx],
            'action': self.actions[idx]
        }


# ============================================================
# Training and evaluation
# ============================================================

def train_model(model, train_loader, epochs=15, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
    
    return epoch_loss / n_batches


def evaluate_model(model, val_loader):
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            losses.append(loss.item())
    
    return np.mean(losses), np.std(losses)


def run_experiment_for_dimension(total_dim, n_runs=2, epochs=15, batch_size=64):
    n_train = 400
    n_val = 100
    multi_action_dim = 3 * 7  # n_steps * action_dim
    
    # Multi-step data
    obs_train, lang_train, actions_train = generate_multi_step_data(n_train, n_steps=3)
    obs_val, lang_val, actions_val = generate_multi_step_data(n_val, n_steps=3)
    
    train_loader = DataLoader(TaskDataset(obs_train, lang_train, actions_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TaskDataset(obs_val, lang_val, actions_val), batch_size=batch_size)
    
    # Single-step data
    obs_ss_train, lang_ss_train, actions_ss_train = generate_single_step_data(n_train)
    obs_ss_val, lang_ss_val, actions_ss_val = generate_single_step_data(n_val)
    
    train_ss_loader = DataLoader(TaskDataset(obs_ss_train, lang_ss_train, actions_ss_train), batch_size=batch_size, shuffle=True)
    val_ss_loader = DataLoader(TaskDataset(obs_ss_val, lang_ss_val, actions_ss_val), batch_size=batch_size)
    
    baseline_losses = []
    cg_losses = []
    cg_ss_losses = []
    baseline_ss_losses = []
    
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # Baseline multi-step
        baseline = BaselineArchitecture(output_dim=multi_action_dim)
        train_model(baseline, train_loader, epochs=epochs)
        bl_loss, _ = evaluate_model(baseline, val_loader)
        baseline_losses.append(bl_loss)
        
        # CG multi-step
        cg = CognitiveGraphArchitecture(total_dim=total_dim, output_dim=multi_action_dim)
        train_model(cg, train_loader, epochs=epochs)
        cg_loss, _ = evaluate_model(cg, val_loader)
        cg_losses.append(cg_loss)
        
        # Baseline single-step
        baseline_ss = BaselineArchitecture()
        train_model(baseline_ss, train_ss_loader, epochs=epochs)
        bl_ss_loss, _ = evaluate_model(baseline_ss, val_ss_loader)
        baseline_ss_losses.append(bl_ss_loss)
        
        # CG single-step
        cg_ss = CognitiveGraphArchitecture(total_dim=total_dim)
        train_model(cg_ss, train_ss_loader, epochs=epochs)
        cg_ss_loss, _ = evaluate_model(cg_ss, val_ss_loader)
        cg_ss_losses.append(cg_ss_loss)
    
    baseline_mean = np.mean(baseline_losses)
    cg_mean = np.mean(cg_losses)
    cg_ss_mean = np.mean(cg_ss_losses)
    baseline_ss_mean = np.mean(baseline_ss_losses)
    
    multi_step_improvement = (baseline_mean - cg_mean) / baseline_mean * 100
    single_step_improvement = (baseline_ss_mean - cg_ss_mean) / baseline_ss_mean * 100
    improvement_gap = single_step_improvement - multi_step_improvement
    
    baseline_s2m_change = (baseline_ss_mean - baseline_mean) / baseline_ss_mean * 100
    cg_s2m_change = (cg_ss_mean - cg_mean) / cg_ss_mean * 100
    
    return {
        'dimension': total_dim,
        'physical_dim': int(total_dim * 0.28),
        'semantic_dim': total_dim - int(total_dim * 0.28),
        'baseline_multi_loss': float(baseline_mean),
        'cg_multi_loss': float(cg_mean),
        'baseline_single_loss': float(baseline_ss_mean),
        'cg_single_loss': float(cg_ss_mean),
        'multi_step_improvement': float(multi_step_improvement),
        'single_step_improvement': float(single_step_improvement),
        'improvement_gap': float(improvement_gap),
        'baseline_s2m_change': float(baseline_s2m_change),
        'cg_s2m_change': float(cg_s2m_change),
        'cg_wins_multi': bool(cg_mean < baseline_mean),
        'cg_wins_single': bool(cg_ss_mean < baseline_ss_mean),
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    SWEEP_DIMENSIONS = [800, 816, 832, 848, 864]
    N_RUNS = 2
    EPOCHS = 15
    
    print("=" * 70)
    print("H1.470.1.1.1: Finer dimension sweep around 832")
    print("Dimensions:", SWEEP_DIMENSIONS)
    print("Runs:", N_RUNS, "Epochs:", EPOCHS)
    print("=" * 70)
    
    all_results = []
    
    for dim in SWEEP_DIMENSIONS:
        print("\n--- Testing dimension", dim, "---")
        result = run_experiment_for_dimension(dim, n_runs=N_RUNS, epochs=EPOCHS)
        all_results.append(result)
        
        print("  Physical:", result['physical_dim'], "Semantic:", result['semantic_dim'])
        print("  Baseline multi: %.4f" % result['baseline_multi_loss'])
        print("  CG multi:       %.4f" % result['cg_multi_loss'])
        print("  Multi-step improvement: %+.2f%%" % result['multi_step_improvement'])
        print("  Single-step improvement: %+.2f%%" % result['single_step_improvement'])
        print("  Improvement gap: %+.2f%%" % result['improvement_gap'])
    
    best = max(all_results, key=lambda r: r['multi_step_improvement'])
    print("\n" + "=" * 70)
    print("BEST DIMENSION:", best['dimension'])
    print("  Multi-step improvement: %+.2f%%" % best['multi_step_improvement'])
    print("=" * 70)
    
    print("\n%6s | %12s | %12s | %10s | %10s | %8s" % ('Dim', 'Base Multi', 'CG Multi', 'Multi Imp', 'Single Imp', 'Gap'))
    print("-" * 75)
    for r in all_results:
        print("%6d | %12.4f | %12.4f | %+.2f%% | %+.2f%% | %+.2f%%" % (
            r['dimension'], r['baseline_multi_loss'], r['cg_multi_loss'],
            r['multi_step_improvement'], r['single_step_improvement'],
            r['improvement_gap']))
    
    conclusion_status = 'CONFIRMED' if best['dimension'] == 832 else 'REFUTED'
    if best['dimension'] == 832:
        conclusion_detail = '832 confirmed as optimal dimension for CG on multi-step tasks'
    else:
        conclusion_detail = '%d is optimal, not 832' % best['dimension']
    
    output = {
        'hypothesis': 'H1.470.1.1.1',
        'description': 'Finer dimension sweep around 832 [800, 816, 832, 848, 864]',
        'prediction': '832 will be confirmed as optimal dimension with peak multi-step performance',
        'sweep_dimensions': SWEEP_DIMENSIONS,
        'n_runs': N_RUNS,
        'epochs': EPOCHS,
        'results': all_results,
        'best_dimension': best['dimension'],
        'best_multi_step_improvement': best['multi_step_improvement'],
        'best_improvement_gap': best['improvement_gap'],
        'conclusion': conclusion_status + ': ' + conclusion_detail
    }
    
    with open('experiments/083-finer_sweep_832/results/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to metrics.json")
