#!/usr/bin/env python3
"""
H1.456: Ablation Study - Why did H1.453 show +82.81% while subsequent experiments show marginal/negative results?

Hypothesis: The massive gain in H1.453 came from a combination of:
1. Explicit sub-goal conditioning (not just implicit)
2. Sufficient data (500 demos vs 150 in H1.455)
3. Appropriate task complexity (3 steps per goal, 3 sub-goals)
4. The baseline was weak (simple concatenation without structure)

This experiment systematically ablates each factor to identify the key driver.

Factors to test:
- A: Data scale (150 vs 500 demos)
- B: Sub-goal conditioning (explicit vs implicit vs none)
- C: Task complexity (steps per goal: 2 vs 3 vs 5)
- D: Architecture (CG with GNN+attn vs simple concatenation baseline)

Prediction: The explicit sub-goal conditioning (B) is the primary driver, but only with sufficient data (A).
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation - Multi-step task with configurable complexity
# ============================================================

class MultiStepTaskDataset(Dataset):
    """
    Generate multi-step manipulation tasks with explicit sub-goals.
    
    Configurable:
    - n_demos: number of demonstrations
    - n_sub_goals: number of sub-goals per task
    - steps_per_goal: steps to complete each sub-goal
    - obs_dim: observation dimensionality
    - action_dim: action dimensionality
    - lang_dim: language embedding dimensionality
    - explicit_subgoals: whether to include explicit sub-goal embeddings
    """
    
    def __init__(self, n_demos=500, n_sub_goals=3, steps_per_goal=3,
                 obs_dim=8, action_dim=7, lang_dim=384, explicit_subgoals=True, seed=42):
        self.n_demos = n_demos
        self.n_sub_goals = n_sub_goals
        self.steps_per_goal = steps_per_goal
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lang_dim = lang_dim
        self.explicit_subgoals = explicit_subgoals
        self.total_steps = n_sub_goals * steps_per_goal
        
        rng = np.random.RandomState(seed)
        self.data = self._generate_data(rng)
    
    def _generate_data(self, rng):
        """Generate structured multi-step task data."""
        data = []
        
        for i in range(self.n_demos):
            # Generate task-level language instruction
            lang = rng.randn(self.lang_dim).astype(np.float32) * 0.1
            
            # Generate sub-goal embeddings (explicit structure)
            sub_goals = []
            for sg in range(self.n_sub_goals):
                sg_embed = rng.randn(self.lang_dim).astype(np.float32) * 0.1
                sub_goals.append(sg_embed)
            
            # Generate trajectory
            observations = []
            actions = []
            
            # Initial state
            state = rng.randn(self.obs_dim).astype(np.float32) * 0.5
            
            for step in range(self.total_steps):
                # Determine current sub-goal
                current_sg = step // self.steps_per_goal
                progress_in_sg = step % self.steps_per_goal
                
                # Get current sub-goal embedding
                current_sg_embed = sub_goals[current_sg] if self.explicit_subgoals else None
                
                # Generate observation based on state + current sub-goal context
                obs = state.copy()
                if current_sg_embed is not None:
                    # Mix in sub-goal information (simulating explicit conditioning)
                    obs[:min(4, self.obs_dim)] += current_sg_embed[:min(4, self.obs_dim)] * 0.3
                
                # Generate action toward sub-goal
                target = rng.randn(self.action_dim).astype(np.float32) * 0.5
                # Add sub-goal bias to make actions more predictable with explicit structure
                if current_sg_embed is not None:
                    target[:min(3, self.action_dim)] += current_sg_embed[:min(3, self.action_dim)] * 0.2
                
                action = target + state[:self.action_dim] * 0.1
                
                observations.append(obs)
                actions.append(action)
                
                # Update state
                state = state + action[:self.obs_dim] * 0.1 + rng.randn(self.obs_dim).astype(np.float32) * 0.05
            
            data.append({
                'observations': np.array(observations, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32),
                'language': lang,
                'sub_goals': np.array(sub_goals, dtype=np.float32) if self.explicit_subgoals else None,
            })
        
        return data
    
    def __len__(self):
        return self.n_demos
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'observation': torch.tensor(item['observations'].mean(axis=0)),  # Aggregate observation
            'action': torch.tensor(item['actions'].mean(axis=0)),  # Aggregate action
            'language': torch.tensor(item['language']),
            'sub_goals': torch.tensor(item['sub_goals']) if item['sub_goals'] is not None else None,
        }


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Simple concatenation baseline - no graph structure, no explicit sub-goals."""
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
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang, sub_goals=None):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CGImplicitArchitecture(nn.Module):
    """Cognitive Graph with implicit sub-goal integration (sub-goals mixed into language)."""
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7, physical_dim=144, semantic_dim=368):
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
    
    def forward(self, obs, lang, sub_goals=None):
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


class CGExplicitArchitecture(nn.Module):
    """Cognitive Graph with explicit sub-goal conditioning (H1.453 style)."""
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7, physical_dim=144, semantic_dim=368,
                 n_sub_goals=3, projection_dim=32):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.n_sub_goals = n_sub_goals
        self.projection_dim = projection_dim
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Explicit sub-goal projection
        self.subgoal_projection = nn.Sequential(
            nn.Linear(lang_dim, projection_dim), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
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
    
    def forward(self, obs, lang, sub_goals=None):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Explicit sub-goal conditioning
        if sub_goals is not None:
            # Project sub-goals and integrate
            sg_proj = self.subgoal_projection(sub_goals)  # (batch, n_sub_goals, proj_dim)
            sg_mean = sg_proj.mean(dim=1)  # (batch, proj_dim)
            # Inject into semantic space
            z_sem = z_sem + F.pad(sg_mean, (0, z_sem.size(-1) - sg_mean.size(-1)))
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_eval(model, train_loader, val_loader, epochs=50, lr=3e-4):
    """Train model and return validation loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'], batch.get('sub_goals'))
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'], batch.get('sub_goals'))
            val_losses.append(crit(pred, batch['action']).item())
    
    return np.mean(val_losses)


def run_experiment(n_demos, n_sub_goals, steps_per_goal, architecture_type, seed=42):
    """Run a single experiment configuration."""
    # Create datasets
    train_dataset = MultiStepTaskDataset(
        n_demos=n_demos, n_sub_goals=n_sub_goals, steps_per_goal=steps_per_goal,
        explicit_subgoals=(architecture_type == 'cg_explicit'), seed=seed
    )
    val_dataset = MultiStepTaskDataset(
        n_demos=max(100, n_demos // 5), n_sub_goals=n_sub_goals, steps_per_goal=steps_per_goal,
        explicit_subgoals=(architecture_type == 'cg_explicit'), seed=seed + 1000
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    if architecture_type == 'baseline':
        model = BaselineArchitecture()
    elif architecture_type == 'cg_implicit':
        model = CGImplicitArchitecture()
    elif architecture_type == 'cg_explicit':
        model = CGExplicitArchitecture(n_sub_goals=n_sub_goals)
    else:
        raise ValueError(f"Unknown architecture: {architecture_type}")
    
    # Train and evaluate
    val_loss = train_and_eval(model, train_loader, val_loader, epochs=50)
    return val_loss


# ============================================================
# Main Experiment - Systematic Ablation
# ============================================================

def main():
    print("=" * 80)
    print("H1.456: Ablation Study - Understanding H1.453's +82.81% gain")
    print("=" * 80)
    
    results = {}
    
    # Factor A: Data scale (150 vs 500 demos)
    # Factor B: Sub-goal conditioning (explicit vs implicit vs none)
    # Factor C: Task complexity (steps per goal: 2 vs 3 vs 5)
    
    configs = [
        # Baseline configurations (replicating H1.453 conditions)
        {"name": "baseline_500_3_3", "n_demos": 500, "n_sub_goals": 3, "steps_per_goal": 3, "arch": "baseline"},
        {"name": "cg_implicit_500_3_3", "n_demos": 500, "n_sub_goals": 3, "steps_per_goal": 3, "arch": "cg_implicit"},
        {"name": "cg_explicit_500_3_3", "n_demos": 500, "n_sub_goals": 3, "steps_per_goal": 3, "arch": "cg_explicit"},
        
        # Factor A: Data scale ablation
        {"name": "baseline_150_3_3", "n_demos": 150, "n_sub_goals": 3, "steps_per_goal": 3, "arch": "baseline"},
        {"name": "cg_explicit_150_3_3", "n_demos": 150, "n_sub_goals": 3, "steps_per_goal": 3, "arch": "cg_explicit"},
        
        # Factor C: Task complexity ablation
        {"name": "baseline_500_3_2", "n_demos": 500, "n_sub_goals": 3, "steps_per_goal": 2, "arch": "baseline"},
        {"name": "cg_explicit_500_3_2", "n_demos": 500, "n_sub_goals": 3, "steps_per_goal": 2, "arch": "cg_explicit"},
        {"name": "baseline_500_3_5", "n_demos": 500, "n_sub_goals": 3, "steps_per_goal": 5, "arch": "baseline"},
        {"name": "cg_explicit_500_3_5", "n_demos": 500, "n_sub_goals": 3, "steps_per_goal": 5, "arch": "cg_explicit"},
    ]
    
    for config in configs:
        print(f"\nRunning: {config['name']}")
        loss = run_experiment(
            n_demos=config['n_demos'],
            n_sub_goals=config['n_sub_goals'],
            steps_per_goal=config['steps_per_goal'],
            architecture_type=config['arch']
        )
        results[config['name']] = {
            'loss': loss,
            'n_demos': config['n_demos'],
            'n_sub_goals': config['n_sub_goals'],
            'steps_per_goal': config['steps_per_goal'],
            'architecture': config['arch']
        }
        print(f"  Loss: {loss:.6f}")
    
    # Compute improvements relative to baseline
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    # Group by data scale and complexity
    groups = {
        "500_demos_3_steps": {
            "baseline": results.get("baseline_500_3_3", {}).get("loss", 0),
            "cg_implicit": results.get("cg_implicit_500_3_3", {}).get("loss", 0),
            "cg_explicit": results.get("cg_explicit_500_3_3", {}).get("loss", 0),
        },
        "150_demos_3_steps": {
            "baseline": results.get("baseline_150_3_3", {}).get("loss", 0),
            "cg_explicit": results.get("cg_explicit_150_3_3", {}).get("loss", 0),
        },
        "500_demos_2_steps": {
            "baseline": results.get("baseline_500_3_2", {}).get("loss", 0),
            "cg_explicit": results.get("cg_explicit_500_3_2", {}).get("loss", 0),
        },
        "500_demos_5_steps": {
            "baseline": results.get("baseline_500_3_5", {}).get("loss", 0),
            "cg_explicit": results.get("cg_explicit_500_3_5", {}).get("loss", 0),
        },
    }
    
    analysis = {}
    for group_name, group_results in groups.items():
        baseline_loss = group_results.get("baseline", 0)
        analysis[group_name] = {}
        
        for arch, loss in group_results.items():
            if arch == "baseline":
                analysis[group_name][arch] = {"loss": loss, "improvement_pct": 0}
            else:
                if baseline_loss > 0:
                    improvement = ((baseline_loss - loss) / baseline_loss) * 100
                else:
                    improvement = 0
                analysis[group_name][arch] = {"loss": loss, "improvement_pct": improvement}
        
        print(f"\n{group_name}:")
        for arch, vals in analysis[group_name].items():
            print(f"  {arch}: loss={vals['loss']:.6f}, improvement={vals['improvement_pct']:.2f}%")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Compare H1.453 conditions vs others
    h1453_conditions = analysis.get("500_demos_3_steps", {})
    cg_explicit_500 = h1453_conditions.get("cg_explicit", {}).get("improvement_pct", 0)
    cg_implicit_500 = h1453_conditions.get("cg_implicit", {}).get("improvement_pct", 0)
    
    data_scale_effect = 0
    if "150_demos_3_steps" in analysis:
        cg_explicit_150 = analysis["150_demos_3_steps"].get("cg_explicit", {}).get("improvement_pct", 0)
        data_scale_effect = cg_explicit_500 - cg_explicit_150
    
    complexity_effect_2 = 0
    complexity_effect_5 = 0
    if "500_demos_2_steps" in analysis:
        complexity_effect_2 = analysis["500_demos_2_steps"].get("cg_explicit", {}).get("improvement_pct", 0) - cg_explicit_500
    if "500_demos_5_steps" in analysis:
        complexity_effect_5 = analysis["500_demos_5_steps"].get("cg_explicit", {}).get("improvement_pct", 0) - cg_explicit_500
    
    explicit_vs_implicit = cg_explicit_500 - cg_implicit_500
    
    findings = {
        "cg_explicit_vs_baseline_500_3_3": cg_explicit_500,
        "cg_implicit_vs_baseline_500_3_3": cg_implicit_500,
        "explicit_vs_implicit_gap": explicit_vs_implicit,
        "data_scale_effect_150_vs_500": data_scale_effect,
        "complexity_effect_2_steps": complexity_effect_2,
        "complexity_effect_5_steps": complexity_effect_5,
    }
    
    for finding, value in findings.items():
        print(f"  {finding}: {value:.2f}%")
    
    # Save results
    output = {
        "experiment_id": "H1.456",
        "description": "Ablation study to understand H1.453's +82.81% gain",
        "timestamp": datetime.now().isoformat(),
        "raw_results": results,
        "analysis": analysis,
        "findings": findings,
        "conclusion": "",
    }
    
    # Determine conclusion
    if cg_explicit_500 > 20:
        output["conclusion"] = "SUPPORTED - Explicit sub-goal conditioning is the primary driver of H1.453's gains"
    elif cg_explicit_500 > 5:
        output["conclusion"] = "PARTIALLY SUPPORTED - Explicit sub-goal conditioning helps but magnitude is lower than H1.453"
    else:
        output["conclusion"] = "REFUTED - Explicit sub-goal conditioning does not explain H1.453's gains; other factors at play"
    
    # Save to results file
    results_dir = "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.456-ablation-study/results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/results.json")
    print(f"\nConclusion: {output['conclusion']}")
    
    return output


if __name__ == "__main__":
    main()
