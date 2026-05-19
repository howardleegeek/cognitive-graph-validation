"""
H1.459: Task Complexity Investigation
Test whether CG advantage emerges only on complex multi-step tasks.
Previous experiments showed CG underperforms on simple tasks.
Hypothesis: CG might help on complex tasks requiring multi-step reasoning.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


class SimpleTaskDataset(Dataset):
    """Single-step task: map observation to action directly."""
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        # Simple linear relationship: action = W @ obs + noise
        obs = np.random.randn(8).astype(np.float32)
        action = obs[:7] * 0.5 + np.random.randn(7) * 0.1
        lang = np.random.randn(32).astype(np.float32)
        return {'observation': torch.from_numpy(obs), 'language': torch.from_numpy(lang), 'action': torch.from_numpy(action)}


class ComplexTaskDataset(Dataset):
    """Multi-step task: requires reasoning about intermediate states."""
    def __init__(self, n_samples=1000, n_steps=3):
        self.n_samples = n_samples
        self.n_steps = n_steps
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        # Complex relationship: action depends on history and future planning
        obs = np.random.randn(8).astype(np.float32)
        # Action depends on obs AND requires understanding of multi-step process
        action = np.zeros(7, dtype=np.float32)
        for step in range(self.n_steps):
            action += obs[:7] * (0.5 ** (step + 1))
        action += np.random.randn(7) * 0.05
        lang = np.random.randn(32).astype(np.float32)
        return {'observation': torch.from_numpy(obs), 'language': torch.from_numpy(lang), 'action': torch.from_numpy(action)}


class CompositionalTaskDataset(Dataset):
    """Compositional task: combine multiple concepts."""
    def __init__(self, n_samples=1000, n_concepts=4):
        self.n_samples = n_samples
        self.n_concepts = n_concepts
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        obs = np.random.randn(8).astype(np.float32)
        # Action is composition of multiple independent factors
        action = np.zeros(7, dtype=np.float32)
        for i in range(self.n_concepts):
            action += obs[i % 7] * np.random.randn(7) * 0.3
        action += np.random.randn(7) * 0.05
        lang = np.random.randn(32).astype(np.float32)
        return {'observation': torch.from_numpy(obs), 'language': torch.from_numpy(lang), 'action': torch.from_numpy(action)}


# Architectures
class BaselineArchitecture(nn.Module):
    """Simple concatenation baseline."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Cognitive Graph with unified representation."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)) 
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


def train_and_eval(model, train_loader, val_loader, epochs=50):
    """Train model and return validation loss."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_experiment(task_name, task_dataset_class, n_steps=None, n_concepts=None):
    """Run experiment on a specific task type."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    
    # Create datasets
    if n_steps is not None:
        train_data = task_dataset_class(n_samples=1000, n_steps=n_steps)
        val_data = task_dataset_class(n_samples=200, n_steps=n_steps)
    elif n_concepts is not None:
        train_data = task_dataset_class(n_samples=1000, n_concepts=n_concepts)
        val_data = task_dataset_class(n_samples=200, n_concepts=n_concepts)
    else:
        train_data = task_dataset_class(n_samples=1000)
        val_data = task_dataset_class(n_samples=200)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    
    # Test both architectures
    results = {}
    
    # Baseline
    baseline = BaselineArchitecture()
    baseline_loss = train_and_eval(baseline, train_loader, val_loader)
    results['baseline_loss'] = baseline_loss
    print(f"Baseline loss: {baseline_loss:.6f}")
    
    # Cognitive Graph
    cg = CognitiveGraphArchitecture()
    cg_loss = train_and_eval(cg, train_loader, val_loader)
    results['cg_loss'] = cg_loss
    print(f"Cognitive Graph loss: {cg_loss:.6f}")
    
    # Calculate improvement
    improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
    results['improvement_pct'] = improvement
    results['cg_wins'] = improvement > 0
    
    print(f"Improvement: {improvement:+.2f}%")
    print(f"CG wins: {results['cg_wins']}")
    
    return results


def main():
    results = {}
    
    # Task 1: Simple single-step (baseline)
    results['simple_single_step'] = run_experiment(
        "Simple Single-Step Task", 
        SimpleTaskDataset
    )
    
    # Task 2: Multi-step (2 steps)
    results['multi_step_2'] = run_experiment(
        "Multi-Step Task (2 steps)", 
        ComplexTaskDataset,
        n_steps=2
    )
    
    # Task 3: Multi-step (3 steps)
    results['multi_step_3'] = run_experiment(
        "Multi-Step Task (3 steps)", 
        ComplexTaskDataset,
        n_steps=3
    )
    
    # Task 4: Multi-step (5 steps)
    results['multi_step_5'] = run_experiment(
        "Multi-Step Task (5 steps)", 
        ComplexTaskDataset,
        n_steps=5
    )
    
    # Task 5: Compositional (2 concepts)
    results['compositional_2'] = run_experiment(
        "Compositional Task (2 concepts)", 
        CompositionalTaskDataset,
        n_concepts=2
    )
    
    # Task 6: Compositional (4 concepts)
    results['compositional_4'] = run_experiment(
        "Compositional Task (4 concepts)", 
        CompositionalTaskDataset,
        n_concepts=4
    )
    
    # Task 7: Compositional (8 concepts)
    results['compositional_8'] = run_experiment(
        "Compositional Task (8 concepts)", 
        CompositionalTaskDataset,
        n_concepts=8
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Task Complexity vs CG Performance")
    print("="*60)
    
    print(f"\n{'Task':<30} {'Baseline':<12} {'CG':<12} {'Improvement':<12} {'CG Wins'}")
    print("-" * 80)
    
    for task_name, res in results.items():
        print(f"{task_name:<30} {res['baseline_loss']:<12.6f} {res['cg_loss']:<12.6f} {res['improvement_pct']:+.2f}%     {res['cg_wins']}")
    
    # Save results
    output = {
        'experiment_id': 'H1.459',
        'description': 'Task complexity investigation',
        'results': results,
        'conclusion': analyze_results(results)
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nConclusion: {output['conclusion']}")
    
    return output


def analyze_results(results):
    """Analyze results to determine if complexity affects CG performance."""
    simple = results['simple_single_step']['improvement_pct']
    multi_2 = results['multi_step_2']['improvement_pct']
    multi_3 = results['multi_step_3']['improvement_pct']
    multi_5 = results['multi_step_5']['improvement_pct']
    
    comp_2 = results['compositional_2']['improvement_pct']
    comp_4 = results['compositional_4']['improvement_pct']
    comp_8 = results['compositional_8']['improvement_pct']
    
    # Check if CG improves with complexity
    multi_trend = (multi_2 + multi_3 + multi_5) / 3
    comp_trend = (comp_2 + comp_4 + comp_8) / 3
    
    if multi_trend > simple and comp_trend > simple:
        return f"CG improves with task complexity. Simple: {simple:.2f}%, Multi-step avg: {multi_trend:.2f}%, Compositional avg: {comp_trend:.2f}%"
    elif multi_trend > simple:
        return f"CG improves on multi-step but not compositional. Simple: {simple:.2f}%, Multi-step avg: {multi_trend:.2f}%"
    elif comp_trend > simple:
        return f"CG improves on compositional but not multi-step. Simple: {simple:.2f}%, Compositional avg: {comp_trend:.2f}%"
    else:
        return f"CG does NOT improve with task complexity. Simple: {simple:.2f}%, Multi-step avg: {multi_trend:.2f}%, Compositional avg: {comp_trend:.2f}%"


if __name__ == '__main__':
    main()
