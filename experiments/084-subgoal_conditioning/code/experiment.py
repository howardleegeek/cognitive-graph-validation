"""
H1.453: Test CG with explicit sub-goal conditioning on multi-step tasks

Hypothesis: Explicit sub-goal conditioning (providing intermediate goal representations)
will improve CG performance on multi-step tasks compared to implicit learning from
sequence structure alone.

Method:
1. Generate multi-step tasks with explicit sub-goal labels
2. Compare 4 conditions:
   - Baseline: Simple MLP (no language, no sub-goals)
   - Simple Language: Cross-attention with language only
   - CG Implicit: CG with language, no explicit sub-goals (learns from sequence)
   - CG Explicit: CG with language + explicit sub-goal embeddings

Prediction: CG Explicit > CG Implicit > Simple Language > Baseline
Key metric: Improvement on multi-step tasks (3+ sub-goals)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from data_loader import prepare_datasets
import pickle
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============== Model Architectures ==============

class BaselineMLP(nn.Module):
    """Simple baseline with no language conditioning."""
    def __init__(self, obs_dim=8, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang=None, sub_goal=None):
        return self.net(obs)


class SimpleLanguageModel(nn.Module):
    """Cross-attention model with language conditioning."""
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        self.cross_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang, sub_goal=None):
        z_obs = self.obs_encoder(obs).unsqueeze(1)  # [B, 1, 128]
        z_lang = self.lang_encoder(lang).unsqueeze(1)  # [B, 1, 128]
        # Cross-attention: query from obs, key/value from lang
        attn_out, _ = self.cross_attn(z_obs, z_lang, z_lang)
        return self.decoder(attn_out.squeeze(1))


class CognitiveGraphImplicit(nn.Module):
    """CG with language but no explicit sub-goal conditioning."""
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7, 
                 physical_dim=144, semantic_dim=368, projection_dim=32):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        total_dim = physical_dim + semantic_dim
        
        # Project language to smaller dimension first
        self.lang_projection = nn.Sequential(
            nn.Linear(lang_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Encode to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers for graph processing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, sub_goal=None):
        # Project language
        lang_proj = self.lang_projection(lang)
        
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang_proj)
        
        # Create nodes: [state, goal]
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [B, 2, total_dim]
        
        # GNN message passing
        for layer in self.gnn_layers:
            # Mean aggregation for message passing
            messages = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(messages)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


class CognitiveGraphExplicit(nn.Module):
    """CG with language AND explicit sub-goal conditioning."""
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7, 
                 physical_dim=144, semantic_dim=368, projection_dim=32,
                 n_sub_goals=5):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = physical_dim + semantic_dim
        self.n_sub_goals = n_sub_goals
        
        # Project language to smaller dimension
        self.lang_projection = nn.Sequential(
            nn.Linear(lang_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Sub-goal embedding
        self.sub_goal_embedding = nn.Embedding(n_sub_goals, projection_dim)
        
        # Encode to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        self.subgoal_to_unified = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_dim, self.total_dim),
                nn.ReLU(),
                nn.LayerNorm(self.total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(self.total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, sub_goal):
        # Project language
        lang_proj = self.lang_projection(lang)
        
        # Get sub-goal embedding
        sub_goal_emb = self.sub_goal_embedding(sub_goal)  # [B, projection_dim]
        
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_lang = self.lang_to_unified(lang_proj)
        z_subgoal = self.subgoal_to_unified(sub_goal_emb)
        
        # Create nodes: [state, goal, sub_goal]
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_lang_pad = F.pad(z_lang, (self.physical_dim, 0), value=0)
        z_subgoal_pad = F.pad(z_subgoal, (self.physical_dim, 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_lang_pad, z_subgoal_pad], dim=1)  # [B, 3, total_dim]
        
        # GNN message passing
        for layer in self.gnn_layers:
            messages = nodes.mean(dim=1, keepdim=True).expand(-1, 3, -1)
            nodes = nodes + layer(messages)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


# ============== Data Generation ==============

def generate_multi_step_data_with_subgoals(n_demos=500, n_steps_per_goal=3, n_sub_goals=3):
    """Generate multi-step task data with explicit sub-goal labels."""
    
    # Real language embeddings from sentence transformer
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        use_real_embeddings = True
    except:
        print("[Warning] sentence-transformers not available, using random embeddings")
        use_real_embeddings = False
    
    # Generate task instructions
    instructions = [
        "pick up the red block and place it in the green box",
        "move the blue cube to the target area",
        "stack the yellow block on top of the red block",
        "push the green cylinder to the corner",
        "pick the orange sphere and drop it in the container",
        "arrange the blocks in a row",
        "sort the objects by color",
        "build a tower with three blocks",
        "clear the table by moving all objects",
        "transfer items from left to right"
    ]
    
    # Sub-goal descriptions for each task
    sub_goal_descriptions = {
        0: ["approach red block", "grasp red block", "move to green box", "release in box"],
        1: ["approach blue cube", "grasp blue cube", "move to target", "release"],
        2: ["approach yellow block", "grasp yellow block", "move over red block", "stack"],
        3: ["approach green cylinder", "push toward corner", "continue pushing", "reach corner"],
        4: ["approach orange sphere", "grasp sphere", "move to container", "drop"],
        5: ["approach first block", "move to position", "approach second block", "move to position"],
        6: ["identify red objects", "move red objects", "identify blue objects", "move blue objects"],
        7: ["pick first block", "place as base", "pick second block", "stack on base"],
        8: ["approach first object", "move to edge", "approach second object", "move to edge"],
        9: ["approach left item", "grasp item", "move to right", "release item"]
    }
    
    demos = []
    obs_dim = 8
    action_dim = 7
    lang_dim = 384
    
    for demo_idx in range(n_demos):
        task_idx = demo_idx % len(instructions)
        instruction = instructions[task_idx]
        
        # Get language embedding
        if use_real_embeddings:
            lang_emb = model.encode([instruction])[0]
        else:
            lang_emb = np.random.randn(lang_dim).astype(np.float32)
        
        # Generate trajectory with sub-goals
        total_steps = n_steps_per_goal * n_sub_goals
        
        for sub_goal_idx in range(n_sub_goals):
            for step_in_goal in range(n_steps_per_goal):
                # Observation: [x, y, z, gripper, obj_x, obj_y, obj_z, task_id_normalized]
                obs = np.random.randn(obs_dim).astype(np.float32)
                
                # Action: [dx, dy, dz, dgrip, target_x, target_y, target_z]
                # Action depends on current sub-goal
                if sub_goal_idx == 0:  # Approach
                    action = np.array([0.1, 0.05, 0.0, 0.0, 0.5, 0.3, 0.1], dtype=np.float32)
                elif sub_goal_idx == 1:  # Grasp/Manipulate
                    action = np.array([0.0, 0.0, 0.05, -0.5, 0.5, 0.3, 0.15], dtype=np.float32)
                else:  # Move to goal
                    action = np.array([-0.05, 0.1, 0.0, 0.0, 0.2, 0.6, 0.1], dtype=np.float32)
                
                # Add noise
                action += np.random.randn(action_dim).astype(np.float32) * 0.05
                
                demos.append({
                    'observation': obs,
                    'action': action,
                    'language': lang_emb,
                    'instruction': instruction,
                    'task_id': task_idx,
                    'sub_goal': sub_goal_idx,  # Explicit sub-goal label
                    'step_in_goal': step_in_goal,
                    'demo_idx': demo_idx
                })
    
    print(f"[Data] Generated {len(demos)} demonstrations with {n_sub_goals} sub-goals each")
    return demos, lang_dim


class SubGoalDataset(Dataset):
    def __init__(self, demos):
        self.demos = demos
    
    def __len__(self):
        return len(self.demos)
    
    def __getitem__(self, idx):
        d = self.demos[idx]
        return {
            'observation': torch.tensor(d['observation'], dtype=torch.float32),
            'action': torch.tensor(d['action'], dtype=torch.float32),
            'language': torch.tensor(d['language'], dtype=torch.float32),
            'sub_goal': torch.tensor(d['sub_goal'], dtype=torch.long)
        }


def train_and_eval(model, train_loader, val_loader, epochs=50, use_sub_goal=False):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            if use_sub_goal:
                pred = model(batch['observation'], batch['language'], batch['sub_goal'])
            else:
                pred = model(batch['observation'], batch['language'])
            
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            if use_sub_goal:
                pred = model(batch['observation'], batch['language'], batch['sub_goal'])
            else:
                pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def main():
    print("=" * 60)
    print("H1.453: Explicit Sub-Goal Conditioning Experiment")
    print("=" * 60)
    
    # Generate data
    demos, lang_dim = generate_multi_step_data_with_subgoals(
        n_demos=500, 
        n_steps_per_goal=3, 
        n_sub_goals=3
    )
    
    # Split data
    n_train = int(0.8 * len(demos))
    train_demos = demos[:n_train]
    val_demos = demos[n_train:]
    
    train_dataset = SubGoalDataset(train_demos)
    val_dataset = SubGoalDataset(val_demos)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"\n[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    results = {}
    
    # 1. Baseline (no language, no sub-goals)
    print("\n[1/4] Training Baseline...")
    baseline = BaselineMLP(obs_dim=8, action_dim=7)
    baseline_loss = train_and_eval(baseline, train_loader, val_loader, epochs=50, use_sub_goal=False)
    results['baseline_loss'] = baseline_loss
    print(f"  Baseline loss: {baseline_loss:.6f}")
    
    # 2. Simple Language Model (no sub-goals)
    print("\n[2/4] Training Simple Language Model...")
    simple_lang = SimpleLanguageModel(obs_dim=8, lang_dim=lang_dim, action_dim=7)
    simple_lang_loss = train_and_eval(simple_lang, train_loader, val_loader, epochs=50, use_sub_goal=False)
    results['simple_language_loss'] = simple_lang_loss
    print(f"  Simple Language loss: {simple_lang_loss:.6f}")
    
    # 3. CG Implicit (no explicit sub-goals)
    print("\n[3/4] Training CG Implicit...")
    cg_implicit = CognitiveGraphImplicit(obs_dim=8, lang_dim=lang_dim, action_dim=7, projection_dim=32)
    cg_implicit_loss = train_and_eval(cg_implicit, train_loader, val_loader, epochs=50, use_sub_goal=False)
    results['cg_implicit_loss'] = cg_implicit_loss
    print(f"  CG Implicit loss: {cg_implicit_loss:.6f}")
    
    # 4. CG Explicit (with sub-goal conditioning)
    print("\n[4/4] Training CG Explicit...")
    cg_explicit = CognitiveGraphExplicit(obs_dim=8, lang_dim=lang_dim, action_dim=7, projection_dim=32, n_sub_goals=3)
    cg_explicit_loss = train_and_eval(cg_explicit, train_loader, val_loader, epochs=50, use_sub_goal=True)
    results['cg_explicit_loss'] = cg_explicit_loss
    print(f"  CG Explicit loss: {cg_explicit_loss:.6f}")
    
    # Calculate improvements
    results['simple_vs_baseline_pct'] = (baseline_loss - simple_lang_loss) / baseline_loss * 100
    results['cg_implicit_vs_baseline_pct'] = (baseline_loss - cg_implicit_loss) / baseline_loss * 100
    results['cg_explicit_vs_baseline_pct'] = (baseline_loss - cg_explicit_loss) / baseline_loss * 100
    results['cg_explicit_vs_implicit_pct'] = (cg_implicit_loss - cg_explicit_loss) / cg_implicit_loss * 100
    results['cg_explicit_vs_simple_pct'] = (simple_lang_loss - cg_explicit_loss) / simple_lang_loss * 100
    
    # Determine winner
    results['cg_explicit_wins'] = cg_explicit_loss < min(baseline_loss, simple_lang_loss, cg_implicit_loss)
    
    # Config
    results['config'] = {
        'n_demos': 500,
        'n_steps_per_goal': 3,
        'n_sub_goals': 3,
        'lang_dim': lang_dim,
        'projection_dim': 32,
        'epochs': 50,
        'batch_size': 32
    }
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline:          {baseline_loss:.6f}")
    print(f"Simple Language:   {simple_lang_loss:.6f} ({results['simple_vs_baseline_pct']:+.2f}% vs baseline)")
    print(f"CG Implicit:       {cg_implicit_loss:.6f} ({results['cg_implicit_vs_baseline_pct']:+.2f}% vs baseline)")
    print(f"CG Explicit:       {cg_explicit_loss:.6f} ({results['cg_explicit_vs_baseline_pct']:+.2f}% vs baseline)")
    print(f"\nCG Explicit vs Implicit: {results['cg_explicit_vs_implicit_pct']:+.2f}%")
    print(f"CG Explicit vs Simple:   {results['cg_explicit_vs_simple_pct']:+.2f}%")
    print(f"\nCG Explicit wins: {results['cg_explicit_wins']}")
    
    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results saved to {output_path}]")
    
    return results


if __name__ == '__main__':
    main()