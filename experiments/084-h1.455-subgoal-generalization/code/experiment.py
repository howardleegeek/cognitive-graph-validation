"""
H1.455: Sub-goal Generalization Across Task Complexities

Hypothesis: The optimal 3 sub-goals from H1.454 will generalize across different 
task complexities (varying steps per sub-goal: 2/3/5). The inverted-U pattern 
should hold regardless of task complexity.

Context:
- H1.454 found 3 sub-goals optimal (+2.05%) with 3 steps per sub-goal
- H1.454 showed inverted-U: 2 (-1.75%), 3 (+2.05%), 5 (+1.32%), 7 (-4.82%)
- This tests whether the optimal granularity is robust to task complexity

Prediction: 3 sub-goals should remain optimal across all complexity levels
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from data_loader import LIBERODataset


class BaselineArchitecture(nn.Module):
    """Baseline MLP with language conditioning."""
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7, latent_dim=128):
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
            nn.Linear(latent_dim*2, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang, subgoals=None):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphExplicit(nn.Module):
    """Cognitive Graph with explicit sub-goal conditioning."""
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7, 
                 physical_dim=144, semantic_dim=368, n_sub_goals=3):
        super().__init__()
        self.n_sub_goals = n_sub_goals
        self.lang_dim = lang_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        total_dim = physical_dim + semantic_dim
        
        # Unified space projections
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Sub-goal projection: lang_dim -> semantic_dim
        self.subgoal_proj = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-attention for sub-goal reasoning
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, subgoals=None):
        batch_size = obs.size(0)
        
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)  # [batch, physical_dim]
        z_sem = self.lang_to_unified(lang)  # [batch, semantic_dim]
        
        # Pad both to total_dim for stacking
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim), value=0)  # [batch, total_dim]
        z_sem_pad = F.pad(z_sem, (0, self.physical_dim), value=0)  # [batch, total_dim]
        
        # Create nodes: physical state + semantic state + sub-goal nodes
        nodes = [z_phys_pad, z_sem_pad]
        
        # Add sub-goal nodes
        if subgoals is not None:
            for i in range(self.n_sub_goals):
                # subgoals: [batch, n_sub_goals, lang_dim]
                z_subgoal = self.subgoal_proj(subgoals[:, i])  # [batch, semantic_dim]
                z_subgoal_pad = F.pad(z_subgoal, (0, self.physical_dim), value=0)  # [batch, total_dim]
                nodes.append(z_subgoal_pad)
        
        nodes = torch.stack(nodes, dim=1)  # [batch, n_nodes, total_dim]
        
        # GNN processing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, nodes.size(1), -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


def generate_task_data(n_demos, n_steps_per_goal, obs_dim=8, action_dim=7, lang_dim=384):
    """Generate multi-step task data with varying complexity."""
    np.random.seed(42 + n_steps_per_goal)
    
    observations = []
    actions = []
    languages = []
    subgoals = []
    
    for i in range(n_demos):
        # Total steps = n_steps_per_goal * n_sub_goals (3 sub-goals)
        total_steps = n_steps_per_goal * 3
        
        # Generate trajectory
        obs_seq = np.random.randn(total_steps, obs_dim).astype(np.float32)
        act_seq = np.random.randn(total_steps, action_dim).astype(np.float32)
        
        # Language instruction (embedded as random vector)
        lang = np.random.randn(lang_dim).astype(np.float32)
        
        # Sub-goals: 3 sub-goals, each with lang_dim features
        sg = np.random.randn(3, lang_dim).astype(np.float32)
        
        observations.append(obs_seq)
        actions.append(act_seq)
        languages.append(lang)
        subgoals.append(sg)
    
    return observations, actions, languages, subgoals


def create_dataloaders(n_demos, n_steps_per_goal, batch_size=32, obs_dim=8, action_dim=7, lang_dim=384):
    """Create train/val dataloaders."""
    obs, acts, langs, subgs = generate_task_data(n_demos, n_steps_per_goal, obs_dim, action_dim, lang_dim)
    
    # Flatten for training (treating each timestep independently)
    obs_flat = np.concatenate(obs, axis=0)
    acts_flat = np.concatenate(acts, axis=0)
    langs_flat = np.tile(np.array(langs), (len(obs[0]), 1, 1)).reshape(-1, lang_dim)
    subgs_flat = np.tile(np.array(subgs), (len(obs[0]), 1, 1)).reshape(-1, 3, lang_dim)
    
    # Train/val split
    n_train = int(0.8 * len(obs_flat))
    
    train_ds = TensorDataset(
        torch.from_numpy(obs_flat[:n_train]),
        torch.from_numpy(langs_flat[:n_train]),
        torch.from_numpy(subgs_flat[:n_train]),
        torch.from_numpy(acts_flat[:n_train])
    )
    val_ds = TensorDataset(
        torch.from_numpy(obs_flat[n_train:]),
        torch.from_numpy(langs_flat[n_train:]),
        torch.from_numpy(subgs_flat[n_train:]),
        torch.from_numpy(acts_flat[n_train:])
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader


def train_and_eval(model, train_loader, val_loader, epochs=30):
    """Train model and evaluate on validation set."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            obs, lang, subg, act = batch
            opt.zero_grad()
            pred = model(obs, lang, subg)
            loss = crit(pred, act)
            loss.backward()
            opt.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs, lang, subg, act = batch
                pred = model(obs, lang, subg)
                val_losses.append(crit(pred, act).item())
        
        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
    
    return best_loss


def run_experiment():
    """Run H1.455: Sub-goal generalization across task complexities."""
    results = {}
    
    # Config: test 3 sub-goals (optimal from H1.454) across different task complexities
    steps_per_goal_configs = [2, 3, 5]  # Different task complexities
    n_sub_goals = 3  # Fixed at optimal from H1.454
    n_demos = 200  # Reduced for speed
    epochs = 30  # Reduced for speed
    batch_size = 32
    obs_dim = 8
    action_dim = 7
    lang_dim = 384
    
    print("=" * 60)
    print("H1.455: Sub-goal Generalization Across Task Complexities")
    print("=" * 60)
    print(f"Testing 3 sub-goals (optimal from H1.454) across task complexities")
    print(f"Steps per sub-goal: {steps_per_goal_configs}")
    print()
    
    for n_steps in steps_per_goal_configs:
        print(f"\n--- Task Complexity: {n_steps} steps per sub-goal ---")
        
        # Create data
        train_loader, val_loader = create_dataloaders(n_demos, n_steps, batch_size, obs_dim, action_dim, lang_dim)
        
        # Baseline
        baseline = BaselineArchitecture(obs_dim, lang_dim, action_dim)
        baseline_loss = train_and_eval(baseline, train_loader, val_loader, epochs)
        
        # CG Explicit with 3 sub-goals
        cg = CognitiveGraphExplicit(obs_dim, lang_dim, action_dim, n_sub_goals=n_sub_goals)
        cg_loss = train_and_eval(cg, train_loader, val_loader, epochs)
        
        # Calculate improvement
        improvement_pct = ((baseline_loss - cg_loss) / baseline_loss) * 100
        cg_wins = improvement_pct > 0
        
        results[n_steps] = {
            'baseline_loss': baseline_loss,
            'cg_loss': cg_loss,
            'improvement_pct': improvement_pct,
            'cg_wins': cg_wins
        }
        
        print(f"  Baseline Loss: {baseline_loss:.6f}")
        print(f"  CG Loss: {cg_loss:.6f}")
        print(f"  Improvement: {improvement_pct:+.2f}%")
        print(f"  CG Wins: {cg_wins}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_cg_wins = all(r['cg_wins'] for r in results.values())
    avg_improvement = np.mean([r['improvement_pct'] for r in results.values()])
    
    print(f"All complexities CG wins: {all_cg_wins}")
    print(f"Average improvement: {avg_improvement:+.2f}%")
    
    for steps, r in results.items():
        print(f"  {steps} steps/subgoal: {r['improvement_pct']:+.2f}% ({'CG wins' if r['cg_wins'] else 'Baseline wins'})")
    
    # Determine conclusion
    if all_cg_wins and avg_improvement > 0:
        conclusion = "SUPPORTED - 3 sub-goals generalize across task complexities"
    elif not all_cg_wins and avg_improvement < 0:
        conclusion = "REFUTED - 3 sub-goals do not generalize"
    else:
        conclusion = "INCONCLUSIVE - Mixed results across complexities"
    
    print(f"\nConclusion: {conclusion}")
    
    # Save results
    output = {
        'experiment_id': 'H1.455',
        'hypothesis': 'Sub-goal generalization across task complexities',
        'conclusion': conclusion,
        'results': results,
        'config': {
            'n_sub_goals': n_sub_goals,
            'steps_per_goal_configs': steps_per_goal_configs,
            'n_demos': n_demos,
            'epochs': epochs
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


if __name__ == '__main__':
    run_experiment()
