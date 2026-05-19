import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
from data_loader import prepare_datasets

# ============================================================
# H1.454: Test varying numbers of sub-goals (2/3/5/7) to find
# optimal granularity for explicit sub-goal conditioning.
#
# Builds on H1.453 breakthrough: explicit sub-goal conditioning
# achieved +82.81% over baseline. Now we test if there's a
# sweet spot in sub-goal granularity.
#
# Hypothesis: Moderate granularity (3-5 sub-goals) will be optimal.
# Too few (2) = insufficient structure. Too many (7) = overfitting
# and dilution of signal per sub-goal.
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

# ---- Data Generation ----
class MultiStepSubgoalDataset(Dataset):
    """
    Generate multi-step manipulation tasks with explicit sub-goals.
    Each task has N sub-goals, each requiring a sequence of actions.
    """
    def __init__(self, n_demos=500, n_steps_per_goal=3, n_sub_goals=3,
                 obs_dim=8, action_dim=7, lang_dim=384, split='train'):
        self.n_demos = n_demos
        self.n_steps_per_goal = n_steps_per_goal
        self.n_sub_goals = n_sub_goals
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lang_dim = lang_dim
        
        np.random.seed(42 + hash(split) % 1000)
        torch.manual_seed(42 + hash(split) % 1000)
        
        self.data = self._generate_data()
        
        # Split
        n = len(self.data)
        if split == 'train':
            self.data = self.data[:int(0.8*n)]
        else:
            self.data = self.data[int(0.8*n):]
    
    def _generate_data(self):
        data = []
        total_steps = self.n_steps_per_goal * self.n_sub_goals
        
        for i in range(self.n_demos):
            # Generate language embedding (simulated sentence-transformer)
            lang_embed = np.random.randn(self.lang_dim).astype(np.float32)
            lang_embed = lang_embed / np.linalg.norm(lang_embed) * 2.0
            
            # Generate sub-goal embeddings (explicit structure)
            subgoal_embeds = []
            for sg in range(self.n_sub_goals):
                # Each sub-goal has a distinct embedding
                sg_embed = np.random.randn(self.lang_dim).astype(np.float32)
                sg_embed = sg_embed / np.linalg.norm(sg_embed) * 1.5
                subgoal_embeds.append(sg_embed)
            
            # Generate trajectory with sub-goal structure
            observations = []
            actions = []
            subgoal_labels = []
            
            current_state = np.random.randn(self.obs_dim).astype(np.float32) * 0.5
            
            for sg_idx in range(self.n_sub_goals):
                # Target state for this sub-goal
                target_state = current_state + np.random.randn(self.obs_dim).astype(np.float32) * 0.8
                
                for step in range(self.n_steps_per_goal):
                    # Progress toward sub-goal target
                    progress = (step + 1) / self.n_steps_per_goal
                    target = current_state + (target_state - current_state) * progress
                    
                    # Observation with noise
                    obs = target + np.random.randn(self.obs_dim).astype(np.float32) * 0.1
                    
                    # Action to reach next state
                    next_state = target + np.random.randn(self.obs_dim).astype(np.float32) * 0.05
                    action = (next_state[:self.action_dim] - obs[:self.action_dim]) + \
                             np.random.randn(self.action_dim).astype(np.float32) * 0.05
                    
                    observations.append(obs)
                    actions.append(action)
                    subgoal_labels.append(sg_idx)
                    
                    current_state = next_state
            
            data.append({
                'observations': np.array(observations, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32),
                'language': lang_embed,
                'subgoal_embeds': np.array(subgoal_embeds, dtype=np.float32),
                'subgoal_labels': np.array(subgoal_labels, dtype=np.int64),
                'n_sub_goals': self.n_sub_goals,
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        # Return first step of trajectory for single-step prediction
        return {
            'observation': torch.tensor(d['observations'][0]),
            'action': torch.tensor(d['actions'][0]),
            'language': torch.tensor(d['language']),
            'subgoal_embeds': torch.tensor(d['subgoal_embeds']),
            'subgoal_labels': torch.tensor(d['subgoal_labels']),
            'n_sub_goals': d['n_sub_goals'],
        }


# ---- Architectures ----

class BaselineArchitecture(nn.Module):
    """Simple MLP baseline with language conditioning."""
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
    
    def forward(self, obs, lang, subgoal_embeds=None):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CGExplicitArchitecture(nn.Module):
    """
    Cognitive Graph with explicit sub-goal conditioning.
    From H1.453 breakthrough: +82.81% over baseline.
    """
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7,
                 physical_dim=144, semantic_dim=368, n_sub_goals=3):
        super().__init__()
        self.n_sub_goals = n_sub_goals
        total_dim = physical_dim + semantic_dim
        
        # Observation to physical space
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        
        # Language to semantic space
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Sub-goal projector
        self.subgoal_projector = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Sub-goal attention (attend over sub-goals)
        self.subgoal_attn = nn.MultiheadAttention(total_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, subgoal_embeds=None):
        batch_size = obs.size(0)
        
        # Project to unified space
        z_phys = self.obs_to_physical(obs)  # [B, physical_dim]
        z_sem = self.lang_to_semantic(lang)  # [B, semantic_dim]
        
        # Create graph nodes
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))  # [B, total_dim]
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)  # [B, total_dim]
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [B, 2, total_dim]
        
        # Add sub-goal nodes if provided
        if subgoal_embeds is not None:
            z_subgoals = self.subgoal_projector(subgoal_embeds)  # [B, n_sg, semantic_dim]
            z_subgoals_pad = F.pad(z_subgoals, (z_phys.size(-1), 0), value=0)  # [B, n_sg, total_dim]
            nodes = torch.cat([nodes, z_subgoals_pad], dim=1)  # [B, 2+n_sg, total_dim]
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, nodes.size(1), -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention over all nodes
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # If sub-goals present, do sub-goal attention
        if subgoal_embeds is not None:
            n_base = 2
            base_nodes = attn_out[:, :n_base, :]  # [B, 2, total_dim]
            sg_nodes = attn_out[:, n_base:, :]  # [B, n_sg, total_dim]
            
            # Attend from base nodes to sub-goal nodes
            sg_attn_out, _ = self.subgoal_attn(base_nodes, sg_nodes, sg_nodes)
            nodes = base_nodes + sg_attn_out
        else:
            nodes = attn_out
        
        return self.decoder(nodes.mean(dim=1))


# ---- Training ----

def train_model(model, train_loader, val_loader, epochs=50, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'], batch.get('subgoal_embeds'))
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        
        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'], batch.get('subgoal_embeds'))
                loss = criterion(pred, batch['action'])
                val_loss += loss.item()
                n_val += 1
        
        val_loss /= max(n_val, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_experiment():
    """Run H1.454: sub-goal granularity sweep."""
    
    # Config
    n_demos = 500
    n_steps_per_goal = 3
    obs_dim = 8
    action_dim = 7
    lang_dim = 384
    epochs = 50
    batch_size = 32
    
    sub_goal_configs = [2, 3, 5, 7]
    
    results = {}
    
    print("=" * 60)
    print("H1.454: Sub-goal Granularity Sweep")
    print("=" * 60)
    print(f"Configs: {sub_goal_configs}")
    print(f"Demos: {n_demos}, Steps/goal: {n_steps_per_goal}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print()
    
    for n_sg in sub_goal_configs:
        print(f"\n--- Testing {n_sg} sub-goals ---")
        
        # Create datasets
        train_ds = MultiStepSubgoalDataset(
            n_demos=n_demos, n_steps_per_goal=n_steps_per_goal,
            n_sub_goals=n_sg, obs_dim=obs_dim, action_dim=action_dim,
            lang_dim=lang_dim, split='train'
        )
        val_ds = MultiStepSubgoalDataset(
            n_demos=n_demos, n_steps_per_goal=n_steps_per_goal,
            n_sub_goals=n_sg, obs_dim=obs_dim, action_dim=action_dim,
            lang_dim=lang_dim, split='val'
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Baseline
        print(f"  Training Baseline ({n_sg} sub-goals)...")
        baseline = BaselineArchitecture(
            obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim
        )
        baseline_loss = train_model(baseline, train_loader, val_loader, epochs=epochs)
        print(f"  Baseline loss: {baseline_loss:.6f}")
        
        # CG Explicit
        print(f"  Training CG Explicit ({n_sg} sub-goals)...")
        cg_explicit = CGExplicitArchitecture(
            obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim,
            n_sub_goals=n_sg
        )
        cg_explicit_loss = train_model(cg_explicit, train_loader, val_loader, epochs=epochs)
        print(f"  CG Explicit loss: {cg_explicit_loss:.6f}")
        
        # Compute improvement
        improvement_pct = ((baseline_loss - cg_explicit_loss) / baseline_loss) * 100
        
        print(f"  CG Explicit vs Baseline: {improvement_pct:+.2f}%")
        
        results[n_sg] = {
            'baseline_loss': round(baseline_loss, 6),
            'cg_explicit_loss': round(cg_explicit_loss, 6),
            'improvement_pct': round(improvement_pct, 2),
            'cg_wins': cg_explicit_loss < baseline_loss,
        }
    
    # Find optimal
    best_sg = max(results.keys(), key=lambda k: results[k]['improvement_pct'])
    best_improvement = results[best_sg]['improvement_pct']
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Sub-goals':<12} {'Baseline':<12} {'CG Explicit':<14} {'Improvement':<12} {'Wins'}")
    print("-" * 60)
    for n_sg in sub_goal_configs:
        r = results[n_sg]
        print(f"{n_sg:<12} {r['baseline_loss']:<12.6f} {r['cg_explicit_loss']:<14.6f} {r['improvement_pct']:+>11.2f}% {'✓' if r['cg_wins'] else '✗'}")
    print("-" * 60)
    print(f"Optimal: {best_sg} sub-goals ({best_improvement:+.2f}% improvement)")
    
    # Output JSON
    output = {
        'experiment_id': 'H1.454',
        'description': 'Test varying numbers of sub-goals (2/3/5/7) to find optimal granularity',
        'results': results,
        'optimal_sub_goals': best_sg,
        'best_improvement_pct': best_improvement,
        'config': {
            'n_demos': n_demos,
            'n_steps_per_goal': n_steps_per_goal,
            'sub_goal_configs': sub_goal_configs,
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'lang_dim': lang_dim,
            'epochs': epochs,
            'batch_size': batch_size,
        },
        'conclusion': f"Optimal sub-goal granularity is {best_sg} sub-goals with {best_improvement:+.2f}% improvement over baseline.",
    }
    
    print("\n" + json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    result = run_experiment()
    
    # Save results
    os.makedirs('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-subgoal_granularity/results', exist_ok=True)
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-subgoal_granularity/results/metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nResults saved to experiments/084-subgoal_granularity/results/metrics.json")
