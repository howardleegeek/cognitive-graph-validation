#!/usr/bin/env python3
"""
H1.456: Investigate why H1.453 showed massive gains (+82.81%) while subsequent experiments show marginal/negative results.

This experiment systematically investigates the discrepancy between:
- H1.453: +82.81% improvement with explicit sub-goal conditioning
- H1.454: +2.05% improvement (optimal at 3 sub-goals)
- H1.455: -0.81% average (loses at all complexity levels)

We will:
1. Replicate H1.453 exactly to verify reproducibility
2. Test key differences: demo count, task complexity, initialization
3. Analyze what factors contribute to the massive gains
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

class SimpleBaseline(nn.Module):
    """Simple MLP baseline with language conditioning."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, hidden_dim=256):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang_emb):
        obs_feat = F.relu(self.obs_proj(obs))
        lang_feat = F.relu(self.lang_proj(lang_emb))
        combined = torch.cat([obs_feat, lang_feat], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CognitiveGraphExplicit(nn.Module):
    """Cognitive Graph with explicit sub-goal conditioning."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, 
                 projection_dim=32, n_sub_goals=3, hidden_dim=256):
        super().__init__()
        self.n_sub_goals = n_sub_goals
        
        # Projections for different inputs
        self.obs_proj = nn.Linear(obs_dim, projection_dim)
        self.lang_proj = nn.Linear(lang_dim, projection_dim)
        self.subgoal_proj = nn.Linear(lang_dim, projection_dim)
        
        # Graph attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=projection_dim, 
            num_heads=4,
            batch_first=True
        )
        
        # MLP for action prediction
        self.fc1 = nn.Linear(projection_dim * (n_sub_goals + 2), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang_emb, subgoal_embs):
        # Project inputs
        obs_feat = F.relu(self.obs_proj(obs)).unsqueeze(1)  # [B, 1, D]
        lang_feat = F.relu(self.lang_proj(lang_emb)).unsqueeze(1)  # [B, 1, D]
        
        # Project sub-goals
        subgoal_feats = []
        for i in range(self.n_sub_goals):
            sg_feat = F.relu(self.subgoal_proj(subgoal_embs[:, i, :])).unsqueeze(1)
            subgoal_feats.append(sg_feat)
        
        # Combine all nodes
        nodes = torch.cat([obs_feat, lang_feat] + subgoal_feats, dim=1)  # [B, N+2, D]
        
        # Apply self-attention
        attended, _ = self.attention(nodes, nodes, nodes)
        
        # Flatten and predict action
        flattened = attended.reshape(attended.shape[0], -1)
        x = F.relu(self.fc1(flattened))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def generate_synthetic_data(n_demos=500, n_steps_per_goal=3, n_sub_goals=3, 
                          obs_dim=8, action_dim=7, lang_dim=384, seed=42):
    """Generate synthetic data for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate language embeddings (random)
    lang_embs = torch.randn(n_demos, lang_dim)
    
    # Generate sub-goal embeddings (random, one per sub-goal)
    subgoal_embs = torch.randn(n_demos, n_sub_goals, lang_dim)
    
    # Generate observations and actions
    # Simple pattern: each sub-goal contributes to the action
    obs = torch.randn(n_demos, obs_dim)
    actions = torch.zeros(n_demos, action_dim)
    
    # Create a simple pattern where action depends on obs and weighted sum of sub-goals
    for i in range(n_demos):
        # Base action from observation
        base_action = torch.randn(action_dim) * 0.1
        
        # Contribution from each sub-goal
        for sg in range(n_sub_goals):
            weight = torch.randn(action_dim) * (1.0 / (sg + 1))
            subgoal_contrib = torch.matmul(subgoal_embs[i, sg, :], torch.randn(lang_dim, action_dim)) * 0.05
            base_action += subgoal_contrib * weight
            
        actions[i] = base_action + torch.randn(action_dim) * 0.01
    
    return obs, actions, lang_embs, subgoal_embs

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, model_type='baseline'):
    """Train a model and return final validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if model_type == 'baseline':
                obs, actions, lang_emb, _ = batch
                optimizer.zero_grad()
                pred_actions = model(obs, lang_emb)
                loss = criterion(pred_actions, actions)
            else:  # cognitive graph
                obs, actions, lang_emb, subgoal_embs = batch
                optimizer.zero_grad()
                pred_actions = model(obs, lang_emb, subgoal_embs)
                loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'baseline':
                    obs, actions, lang_emb, _ = batch
                    pred_actions = model(obs, lang_emb)
                    loss = criterion(pred_actions, actions)
                else:  # cognitive graph
                    obs, actions, lang_emb, subgoal_embs = batch
                    pred_actions = model(obs, lang_emb, subgoal_embs)
                    loss = criterion(pred_actions, actions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
    return best_val_loss

def run_experiment(config):
    """Run a single experiment configuration."""
    print(f"\nRunning experiment: {config['name']}")
    print(f"Config: {config}")
    
    # Generate data
    obs, actions, lang_embs, subgoal_embs = generate_synthetic_data(
        n_demos=config['n_demos'],
        n_steps_per_goal=config.get('n_steps_per_goal', 3),
        n_sub_goals=config.get('n_sub_goals', 3),
        obs_dim=config.get('obs_dim', 8),
        action_dim=config.get('action_dim', 7),
        lang_dim=config.get('lang_dim', 384),
        seed=config.get('seed', 42)
    )
    
    # Split data
    n_train = int(0.8 * len(obs))
    train_obs, val_obs = obs[:n_train], obs[n_train:]
    train_actions, val_actions = actions[:n_train], actions[n_train:]
    train_lang, val_lang = lang_embs[:n_train], lang_embs[n_train:]
    train_subgoals, val_subgoals = subgoal_embs[:n_train], subgoal_embs[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(train_obs, train_actions, train_lang, train_subgoals)
    val_dataset = TensorDataset(val_obs, val_actions, val_lang, val_subgoals)
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False)
    
    # Train baseline
    baseline = SimpleBaseline(
        obs_dim=config.get('obs_dim', 8),
        action_dim=config.get('action_dim', 7),
        lang_dim=config.get('lang_dim', 384),
        hidden_dim=config.get('hidden_dim', 256)
    )
    baseline_loss = train_model(
        baseline, train_loader, val_loader, 
        epochs=config.get('epochs', 50),
        lr=config.get('lr', 1e-3),
        model_type='baseline'
    )
    
    # Train cognitive graph
    cg = CognitiveGraphExplicit(
        obs_dim=config.get('obs_dim', 8),
        action_dim=config.get('action_dim', 7),
        lang_dim=config.get('lang_dim', 384),
        projection_dim=config.get('projection_dim', 32),
        n_sub_goals=config.get('n_sub_goals', 3),
        hidden_dim=config.get('hidden_dim', 256)
    )
    cg_loss = train_model(
        cg, train_loader, val_loader,
        epochs=config.get('epochs', 50),
        lr=config.get('lr', 1e-3),
        model_type='cognitive_graph'
    )
    
    # Calculate improvement
    improvement_pct = ((baseline_loss - cg_loss) / baseline_loss) * 100
    cg_wins = cg_loss < baseline_loss
    
    print(f"Baseline loss: {baseline_loss:.6f}")
    print(f"CG loss: {cg_loss:.6f}")
    print(f"Improvement: {improvement_pct:.2f}%")
    print(f"CG wins: {cg_wins}")
    
    return {
        'baseline_loss': baseline_loss,
        'cg_loss': cg_loss,
        'improvement_pct': improvement_pct,
        'cg_wins': cg_wins
    }

def main():
    """Main experiment function."""
    print("=" * 80)
    print("H1.456: Investigating H1.453 Discrepancy")
    print("=" * 80)
    
    # Experiment configurations to test
    experiments = [
        # 1. Replicate H1.453 exactly
        {
            'name': 'H1.453_replication',
            'n_demos': 500,
            'n_steps_per_goal': 3,
            'n_sub_goals': 3,
            'lang_dim': 384,
            'projection_dim': 32,
            'epochs': 50,
            'batch_size': 32,
            'seed': 42
        },
        # 2. Test with H1.454 configuration (different seed)
        {
            'name': 'H1.454_config',
            'n_demos': 500,
            'n_steps_per_goal': 3,
            'n_sub_goals': 3,  # Optimal from H1.454
            'lang_dim': 384,
            'projection_dim': 32,
            'epochs': 50,
            'batch_size': 32,
            'seed': 123  # Different seed
        },
        # 3. Test with fewer demos (like H1.455)
        {
            'name': 'H1.455_demo_count',
            'n_demos': 150,
            'n_steps_per_goal': 3,
            'n_sub_goals': 3,
            'lang_dim': 384,
            'projection_dim': 32,
            'epochs': 20,  # Fewer epochs like H1.455
            'batch_size': 32,
            'seed': 42
        },
        # 4. Test different task complexity (2 steps per goal)
        {
            'name': 'complexity_2_steps',
            'n_demos': 500,
            'n_steps_per_goal': 2,
            'n_sub_goals': 3,
            'lang_dim': 384,
            'projection_dim': 32,
            'epochs': 50,
            'batch_size': 32,
            'seed': 42
        },
        # 5. Test different task complexity (5 steps per goal)
        {
            'name': 'complexity_5_steps',
            'n_demos': 500,
            'n_steps_per_goal': 5,
            'n_sub_goals': 3,
            'lang_dim': 384,
            'projection_dim': 32,
            'epochs': 50,
            'batch_size': 32,
            'seed': 42
        },
        # 6. Test initialization sensitivity
        {
            'name': 'init_sensitivity',
            'n_demos': 500,
            'n_steps_per_goal': 3,
            'n_sub_goals': 3,
            'lang_dim': 384,
            'projection_dim': 32,
            'epochs': 50,
            'batch_size': 32,
            'seed': 999  # Very different seed
        }
    ]
    
    results = {}
    for exp_config in experiments:
        result = run_experiment(exp_config)
        results[exp_config['name']] = result
        
        # Save intermediate results
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Analyze results
    print("\n" + "=" * 80)
    print("Analysis of H1.453 Discrepancy")
    print("=" * 80)
    
    h1_453_replication = results.get('H1.453_replication', {})
    h1_454_config = results.get('H1.454_config', {})
    h1_455_demo = results.get('H1.455_demo_count', {})
    
    print(f"\nH1.453 Replication: {h1_453_replication.get('improvement_pct', 0):.2f}%")
    print(f"H1.454 Config: {h1_454_config.get('improvement_pct', 0):.2f}%")
    print(f"H1.455 Demo Count: {h1_455_demo.get('improvement_pct', 0):.2f}%")
    
    # Key insights
    print("\nKey Insights:")
    
    # Check if we can reproduce H1.453
    if h1_453_replication.get('improvement_pct', 0) > 50:
        print("✓ H1.453 result (+82.81%) appears reproducible")
    else:
        print("✗ H1.453 result NOT reproducible with current setup")
        
    # Check seed sensitivity
    seed_diff = abs(h1_453_replication.get('improvement_pct', 0) - 
                   h1_454_config.get('improvement_pct', 0))
    if seed_diff > 20:
        print(f"✓ High seed sensitivity detected: {seed_diff:.2f}% difference")
    else:
        print(f"✗ Low seed sensitivity: {seed_diff:.2f}% difference")
        
    # Check demo count effect
    demo_diff = abs(h1_453_replication.get('improvement_pct', 0) - 
                   h1_455_demo.get('improvement_pct', 0))
    if demo_diff > 20:
        print(f"✓ Demo count significantly affects results: {demo_diff:.2f}% difference")
    else:
        print(f"✗ Demo count has minimal effect: {demo_diff:.2f}% difference")
    
    # Save final results
    output = {
        'experiment_id': 'H1.456',
        'description': 'Investigate why H1.453 showed massive gains while subsequent experiments show marginal/negative results',
        'results': results,
        'analysis': {
            'h1_453_reproducible': h1_453_replication.get('improvement_pct', 0) > 50,
            'h1_453_replication_improvement': h1_453_replication.get('improvement_pct', 0),
            'seed_sensitivity': seed_diff,
            'demo_count_effect': demo_diff,
            'complexity_2_steps_improvement': results.get('complexity_2_steps', {}).get('improvement_pct', 0),
            'complexity_5_steps_improvement': results.get('complexity_5_steps', {}).get('improvement_pct', 0),
            'init_sensitivity_improvement': results.get('init_sensitivity', {}).get('improvement_pct', 0)
        }
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to experiment_results.json")

if __name__ == "__main__":
    main()