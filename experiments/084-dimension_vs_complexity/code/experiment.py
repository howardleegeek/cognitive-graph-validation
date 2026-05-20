#!/usr/bin/env python3
"""
H1.470.1.1.2: Does optimal dimension shift with task complexity?

Hypothesis: The optimal representation dimension for CG depends on task complexity.
More complex tasks (more steps) require higher dimensions to encode the additional
state history and planning information.

Prediction: 
- 2-step tasks: optimal dimension ~768
- 3-step tasks: optimal dimension ~816 (confirmed in H1.470.1.1.1)
- 4-step tasks: optimal dimension ~864
- 5-step tasks: optimal dimension ~912

Test: Run CG at dimensions [768, 816, 864, 912] across 2-step, 3-step, 4-step, 5-step tasks.
If the hypothesis is correct, the optimal dimension should increase monotonically with steps.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader

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
    def __init__(self, obs_dim=8, lang_dim=32, output_dim=7, total_dim=816, dropout=0.2):
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
    
    actions = actions.view(n_samples, -1)
    
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


def run_experiment(n_steps, total_dim, n_runs=2, epochs=15, batch_size=64):
    n_train = 400
    n_val = 100
    action_dim = 7
    output_dim = n_steps * action_dim
    
    obs_train, lang_train, actions_train = generate_multi_step_data(n_train, n_steps=n_steps)
    obs_val, lang_val, actions_val = generate_multi_step_data(n_val, n_steps=n_steps)
    
    train_loader = DataLoader(TaskDataset(obs_train, lang_train, actions_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TaskDataset(obs_val, lang_val, actions_val), batch_size=batch_size)
    
    baseline_losses = []
    cg_losses = []
    
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        baseline = BaselineArchitecture(output_dim=output_dim)
        train_model(baseline, train_loader, epochs=epochs)
        bl_loss, _ = evaluate_model(baseline, val_loader)
        baseline_losses.append(bl_loss)
        
        cg = CognitiveGraphArchitecture(total_dim=total_dim, output_dim=output_dim)
        train_model(cg, train_loader, epochs=epochs)
        cg_loss, _ = evaluate_model(cg, val_loader)
        cg_losses.append(cg_loss)
    
    return {
        'baseline_loss': float(np.mean(baseline_losses)),
        'cg_loss': float(np.mean(cg_losses)),
        'improvement': float((np.mean(baseline_losses) - np.mean(cg_losses)) / np.mean(baseline_losses) * 100),
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    DIMENSIONS = [768, 816, 864, 912]
    STEP_COUNTS = [2, 3, 4, 5]
    N_RUNS = 2
    EPOCHS = 15
    
    print("=" * 70)
    print("H1.470.1.1.2: Dimension vs Task Complexity")
    print("Dimensions:", DIMENSIONS)
    print("Step counts:", STEP_COUNTS)
    print("=" * 70)
    
    all_results = {}
    
    for n_steps in STEP_COUNTS:
        print("\n=== %d-step tasks ===" % n_steps)
        step_results = []
        
        for dim in DIMENSIONS:
            print("  Testing dim %d..." % dim, end=" ")
            result = run_experiment(n_steps, dim, n_runs=N_RUNS, epochs=EPOCHS)
            result['dimension'] = dim
            result['n_steps'] = n_steps
            step_results.append(result)
            print("CG improvement: %+.2f%%" % result['improvement'])
        
        all_results[n_steps] = step_results
        
        # Find best dimension for this step count
        best = max(step_results, key=lambda r: r['improvement'])
        print("  Best dimension for %d steps: %d (%+.2f%%)" % (n_steps, best['dimension'], best['improvement']))
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: CG improvement (%) by dimension and step count")
    print("=" * 70)
    
    header = "%6s |" % "Steps"
    for dim in DIMENSIONS:
        header += " %6d |" % dim
    header += " Best"
    print(header)
    print("-" * (10 + 9 * len(DIMENSIONS)))
    
    optimal_dims = {}
    for n_steps in STEP_COUNTS:
        row = "%6d |" % n_steps
        best_dim = None
        best_imp = -999
        for r in all_results[n_steps]:
            row += " %+.2f%% |" % r['improvement']
            if r['improvement'] > best_imp:
                best_imp = r['improvement']
                best_dim = r['dimension']
        row += " %d" % best_dim
        optimal_dims[n_steps] = best_dim
        print(row)
    
    print("\nOptimal dimensions by step count:")
    for n_steps in STEP_COUNTS:
        print("  %d steps -> %d dimensions" % (n_steps, optimal_dims[n_steps]))
    
    # Check if optimal dimension increases with steps
    dims_list = [optimal_dims[s] for s in STEP_COUNTS]
    is_monotonic = all(dims_list[i] <= dims_list[i+1] for i in range(len(dims_list)-1))
    
    print("\nMonotonic increase:", is_monotonic)
    print("Hypothesis:", "SUPPORTED" if is_monotonic else "REFUTED")
    
    output = {
        'hypothesis': 'H1.470.1.1.2',
        'description': 'Optimal dimension shifts with task complexity',
        'prediction': 'Optimal dimension increases monotonically with number of steps',
        'dimensions': DIMENSIONS,
        'step_counts': STEP_COUNTS,
        'n_runs': N_RUNS,
        'epochs': EPOCHS,
        'results': all_results,
        'optimal_dimensions': {str(k): v for k, v in optimal_dims.items()},
        'is_monotonic': is_monotonic,
        'conclusion': 'SUPPORTED: Optimal dimension increases with task complexity' if is_monotonic else 'REFUTED: Optimal dimension does not increase monotonically with task complexity'
    }
    
    with open('experiments/084-dimension_vs_complexity/results/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to metrics.json")
