#!/usr/bin/env python3
"""
H1.209: Hierarchical Goal Reasoning on Long-Horizon Sequences (100-200 steps)

Previous findings:
- H3.100: Multi-scale goal decomposition: subgoal best (+20.1%), multi-scale +5.1% vs endpoint
- H1.208: Ultra-long (300-500 steps) +46.9% with combined goals, +33.4% over endpoint alone
- H3.95: Endpoint goal on 100+ steps (+95.3%), advantage grows with sequence length

Hypothesis: Hierarchical goal reasoning (multi-scale: endpoint + subgoals + milestones)
will outperform single-scale goal conditioning on 100-200 step sequences.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader


def generate_goal_conditioned_data(n_demos=400, max_steps=200, with_hierarchy=False):
    """Generate goal-conditioned manipulation data with temporal structure.
    
    Key insight from prior research: Goal state is CRITICAL for enabling attention.
    Now test hierarchical goal reasoning with multi-scale goals.
    """
    np.random.seed(42)
    data = []
    
    task_templates = [
        "pick up the {obj} and place it in the {container}",
        "grasp the {obj}, move to {loc}, and release",
        "reach for {obj1}, transport to {obj2}, and place",
        "navigate to {loc}, pick {obj}, carry to {container}",
    ]
    
    objects = ["red cube", "blue block", "green sphere", "yellow plate", "white bowl"]
    containers = ["basket", "bin", "box", "tray", "shelf"]
    locations = ["left corner", "right side", "center", "front area", "back region"]
    
    for i in range(n_demos):
        template = np.random.choice(task_templates)
        lang = template.format(
            obj=np.random.choice(objects),
            obj1=np.random.choice(objects),
            obj2=np.random.choice(objects),
            container=np.random.choice(containers),
            loc=np.random.choice(locations),
        )
        
        # Generate variable-length trajectory (100-200 steps)
        seq_len = np.random.randint(100, 201)
        
        # Generate with autocorrelation (robot-like dynamics, ρ=0.85)
        rho = 0.85
        obs = np.zeros((seq_len, 8), dtype=np.float32)
        obs[0] = np.random.randn(8) * 0.3
        for t in range(1, seq_len):
            obs[t] = rho * obs[t-1] + np.random.randn(8) * np.sqrt(1 - rho**2) * 0.3
        obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
        obs[:, 7] = np.clip(obs[:, 7], 0, 1)
        
        # Actions with temporal coherence
        actions = np.zeros((seq_len, 7), dtype=np.float32)
        actions[0] = np.random.randn(7) * 0.05
        for t in range(1, seq_len):
            actions[t] = rho * actions[t-1] + np.random.randn(7) * np.sqrt(1 - rho**2) * 0.05
        actions[:, 6] = np.clip(actions[:, 6], -1, 1)
        
        # Language embedding
        lang_emb = np.random.randn(32).astype(np.float32)
        
        # Goal state: endpoint (last observation)
        goal_endpoint = obs[-1].copy()
        
        # Hierarchical goals: milestone (25%), subgoal (50%), endpoint (100%)
        if with_hierarchy:
            milestones = [obs[seq_len//4].copy(), obs[seq_len//2].copy(), obs[3*seq_len//4].copy(), goal_endpoint]
        else:
            milestones = [goal_endpoint]
        
        data.append({
            "observations": obs,
            "actions": actions,
            "language": lang,
            "language_embedding": lang_emb,
            "goal_endpoint": goal_endpoint,
            "milestones": milestones,
            "task_id": i % 4,
            "seq_len": seq_len,
        })
    
    return data


class HierarchicalGoalDataset(Dataset):
    def __init__(self, data, seq_len=50):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        demo = self.data[idx]
        seq_len = len(demo["observations"])
        
        # Sample starting point and ensure we get exactly seq_len
        if seq_len > self.seq_len:
            start = np.random.randint(0, seq_len - self.seq_len)
            end = start + self.seq_len
        else:
            start = 0
            end = seq_len
        
        # Pad if needed
        obs = demo["observations"][start:end]
        if len(obs) < self.seq_len:
            pad_len = self.seq_len - len(obs)
            obs = np.pad(obs, ((0, pad_len), (0, 0)), mode='edge')
        
        actions = demo["actions"][start:end]
        if len(actions) < self.seq_len:
            pad_len = self.seq_len - len(actions)
            actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='edge')
        
        return {
            "observations": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "language": torch.tensor(demo["language_embedding"], dtype=torch.float32),
            "goal_endpoint": torch.tensor(demo["goal_endpoint"], dtype=torch.float32),
            "milestones": [torch.tensor(m, dtype=torch.float32) for m in demo["milestones"]],
            "task_id": demo["task_id"],
            "seq_len": end - start,
        }


class FlatGoalEncoder(nn.Module):
    """Baseline: Flat concatenation of goal state."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, goal_dim=8, hidden=256):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.goal_enc = nn.Sequential(nn.Linear(goal_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 64 + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang, goal):
        # obs_seq: [B, T, obs_dim] OR [B, obs_dim] if seq_len=1
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)  # [B, 1, obs_dim]
        obs_feat = self.obs_enc(obs_seq.mean(dim=1))  # [B, hidden]
        goal_feat = self.goal_enc(goal)  # [B, 64]
        lang_feat = self.lang_enc(lang)  # [B, 64]
        return self.fusion(torch.cat([obs_feat, goal_feat, lang_feat], dim=-1))


class HierarchicalGoalReasoning(nn.Module):
    """Multi-scale hierarchical goal reasoning.
    
    Processes endpoint goal, milestones, and observations at multiple scales.
    Based on H3.100 finding: subgoal (+20.1%) best, multi-scale (+5.1%) over endpoint.
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, goal_dim=8, hidden=256):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.goal_enc = nn.Sequential(nn.Linear(goal_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        
        # Hierarchical encoder for multi-scale goals
        self.milestone_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(goal_dim, 32), nn.ReLU()) for _ in range(4)
        ])
        
        # Attention over milestones - use simple weighted sum
        self.milestone_attn = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 64 + 64 + 32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang, goal, milestones):
        # obs_seq: [B, T, obs_dim] OR [B, obs_dim] if seq_len=1
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)  # [B, 1, obs_dim]
        obs_feat = self.obs_enc(obs_seq.mean(dim=1))  # [B, hidden]
        goal_feat = self.goal_enc(goal)  # [B, 64]
        lang_feat = self.lang_enc(lang)  # [B, 64]
        
        # Encode milestones and attention-pool
        milestone_feats = [enc(m) for enc, m in zip(self.milestone_enc, milestones)]
        milestone_stack = torch.stack(milestone_feats, dim=1)  # [B, 4, 32]
        
        # Attention weights over milestones
        weights = F.softmax(self.milestone_attn(milestone_stack), dim=1)  # [B, 4, 1]
        milestone_out = (milestone_stack * weights).sum(dim=1)  # [B, 32]
        
        return self.fusion(torch.cat([obs_feat, goal_feat, lang_feat, milestone_out], dim=-1))


def train_model(model, train_loader, val_loader, epochs=80, lr=3e-4):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            obs = batch["observations"]
            lang = batch["language"]
            goal = batch["goal_endpoint"]
            milestones = batch["milestones"]
            actions = batch["actions"][:, -1]  # Predict final action
            
            if isinstance(model, HierarchicalGoalReasoning):
                pred = model(obs, lang, goal, milestones)
            else:
                pred = model(obs, lang, goal)
            
            loss = criterion(pred, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Evaluate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            obs = batch["observations"]
            lang = batch["language"]
            goal = batch["goal_endpoint"]
            milestones = batch["milestones"]
            actions = batch["actions"][:, -1]
            
            if isinstance(model, HierarchicalGoalReasoning):
                pred = model(obs, lang, goal, milestones)
            else:
                pred = model(obs, lang, goal)
            
            val_losses.append(criterion(pred, actions).item())
    
    return np.mean(val_losses)


def run_experiment():
    print("=" * 70)
    print("H1.209: Hierarchical Goal Reasoning on 100-200 Step Sequences")
    print("=" * 70)
    
    # Generate data with hierarchical goals
    print("\nGenerating hierarchical goal-conditioned data...")
    data = generate_goal_conditioned_data(n_demos=400, max_steps=200, with_hierarchy=True)
    
    # Split
    train_data = HierarchicalGoalDataset(data[:320], seq_len=50)
    val_data = HierarchicalGoalDataset(data[320:400], seq_len=50)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    
    # Test different sequence lengths
    seq_lengths = [100, 125, 150, 175, 200]
    flat_results = {}
    hier_results = {}
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        train_loader_cur = DataLoader(
            HierarchicalGoalDataset(data[:320], seq_len=seq_len), 
            batch_size=16, shuffle=True
        )
        val_loader_cur = DataLoader(
            HierarchicalGoalDataset(data[320:400], seq_len=seq_len), 
            batch_size=16
        )
        
        # Flat goal model
        flat_model = FlatGoalEncoder()
        flat_loss = train_model(flat_model, train_loader_cur, val_loader_cur)
        flat_results[seq_len] = flat_loss
        
        # Hierarchical goal model
        hier_model = HierarchicalGoalReasoning()
        hier_loss = train_model(hier_model, train_loader_cur, val_loader_cur)
        hier_results[seq_len] = hier_loss
        
        delta = (flat_loss - hier_loss) / flat_loss * 100
        winner = "HIERARCHICAL" if hier_loss < flat_loss else "FLAT"
        print(f"  Flat Goal: {flat_loss:.6f}")
        print(f"  Hierarchical: {hier_loss:.6f}")
        print(f"  Δ: {delta:+.1f}% ({winner} wins)")
    
    # Summary
    avg_delta = np.mean([(flat_results[s] - hier_results[s]) / flat_results[s] * 100 
                         for s in seq_lengths])
    
    results = {
        "hypothesis_id": "H1.209",
        "hypothesis": "Hierarchical goal reasoning outperforms flat on 100-200 step sequences",
        "flat_results": {str(k): float(v) for k, v in flat_results.items()},
        "hierarchical_results": {str(k): float(v) for k, v in hier_results.items()},
        "avg_delta_percent": float(avg_delta),
        "hier_wins": bool(all(hier_results[s] < flat_results[s] for s in seq_lengths)),
        "wins_per_length": {str(s): bool(hier_results[s] < flat_results[s]) for s in seq_lengths},
        "status": "SUPPORTED" if avg_delta > 0 else "REFUTED",
        "note": f"Hierarchical {'wins' if avg_delta > 0 else 'loses'} {sum(hier_results[s] < flat_results[s] for s in seq_lengths)}/{len(seq_lengths)} lengths, avg {avg_delta:+.1f}%",
    }
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for s in seq_lengths:
        delta = (flat_results[s] - hier_results[s]) / flat_results[s] * 100
        print(f"  {s} steps: Flat={flat_results[s]:.6f}, Hier={hier_results[s]:.6f}, Δ={delta:+.1f}%")
    print(f"\nAverage Δ: {avg_delta:+.1f}%")
    print(f"Status: {results['status']}")
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == "__main__":
    results = run_experiment()