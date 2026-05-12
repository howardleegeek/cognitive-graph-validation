#!/usr/bin/env python3
"""
H3.101: SSM with Goal Conditioning on Manipulation Tasks

Previous findings:
- H1.193: SSM +97.6% on 50-step with autocorrelation (next-step prediction)
- H3.10: SSM vs Attention: +93.0% average (SSM wins over both!)
- H1.202: Task structure enables SSM (+37.2%) and Attention (+89.7%)

Hypothesis: SSM with goal conditioning will combine SSM's temporal modeling 
with goal state's power to exceed both vanilla SSM and goal-conditioned attention.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader


def generate_manipulation_data(n_demos=400, task_type="manipulation"):
    """Generate manipulation data with task structure (goal states, action outcomes).
    
    Based on H1.202: Task structure enables SSM (+37.2%) and Attention (+89.7%).
    Key insight: manipulation tasks have inherent temporal structure that SSM can exploit.
    """
    np.random.seed(43)
    data = []
    
    tasks = [
        "pick up the {obj}",
        "place the {obj} in the {cnt}",
        "push the {obj} to the {loc}",
        "stack {obj1} on {obj2}",
        "open the {cnt}",
    ]
    
    objects = ["red cube", "blue block", "green sphere", "yellow plate"]
    containers = ["basket", "bin", "box", "shelf"]
    locations = ["left", "right", "center", "front"]
    
    for i in range(n_demos):
        template = np.random.choice(tasks)
        lang = template.format(
            obj=np.random.choice(objects),
            obj1=np.random.choice(objects),
            obj2=np.random.choice(objects),
            cnt=np.random.choice(containers),
            loc=np.random.choice(locations),
        )
        
        # Generate trajectory with task structure
        seq_len = np.random.randint(20, 101)
        
        # Observations with autocorrelation (robot-like dynamics)
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
        
        # Goal state (endpoint)
        goal_endpoint = obs[-1].copy()
        
        # Subgoal at 50% progress
        goal_subgoal = obs[seq_len//2].copy()
        
        data.append({
            "observations": obs,
            "actions": actions,
            "language": lang,
            "language_embedding": lang_emb,
            "goal_endpoint": goal_endpoint,
            "goal_subgoal": goal_subgoal,
            "task_id": i % 5,
            "seq_len": seq_len,
        })
    
    return data


class SimpleSSM(nn.Module):
    """Baseline SSM without goal conditioning.
    
    Based on H3.9-10: SSM dramatically outperforms attention on manipulation tasks.
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=256, state_dim=64):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        
        # SSM state update: x_{t+1} = A x_t + B u_t
        self.A = nn.Parameter(torch.eye(state_dim) * 0.9 + torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Linear(hidden, state_dim)
        self.C = nn.Linear(state_dim, hidden)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        # obs_seq: [B, T, obs_dim]
        B, T, _ = obs_seq.shape
        
        # Encode observations
        obs_feat = self.obs_enc(obs_seq)  # [B, T, hidden]
        lang_feat = self.lang_enc(lang)  # [B, 64]
        
        # SSM forward pass
        state = torch.zeros(B, self.A.shape[0], device=obs_seq.device)
        for t in range(T):
            u_t = obs_feat[:, t]  # [B, hidden]
            state = torch.tanh(self.A @ state.T).T + self.B(u_t)
        
        ssm_out = self.C(state)  # [B, hidden]
        
        return self.fusion(torch.cat([ssm_out, lang_feat], dim=-1))


class SSMWithGoalConditioning(nn.Module):
    """SSM with goal-conditioned state initialization.
    
    Key innovation: Initialize SSM state based on goal, letting SSM model
    the trajectory FROM goal to current state (reverse planning).
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=256, state_dim=64):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.goal_enc = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, state_dim))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        
        # SSM matrices
        self.A = nn.Parameter(torch.eye(state_dim) * 0.9 + torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Linear(hidden, state_dim)
        self.C = nn.Linear(state_dim, hidden)
        
        # Goal-conditioned SSM modulation
        self.goal_gate = nn.Sequential(nn.Linear(state_dim, state_dim), nn.Sigmoid())
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 64 + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang, goal):
        B, T, _ = obs_seq.shape
        
        obs_feat = self.obs_enc(obs_seq)
        lang_feat = self.lang_enc(lang)
        goal_init = self.goal_enc(goal)  # [B, state_dim]
        
        # Initialize SSM state from goal
        state = goal_init
        gate = self.goal_gate(goal_init)
        
        for t in range(T):
            u_t = obs_feat[:, t]
            state = torch.tanh((self.A @ state.T).T + self.B(u_t))
            state = gate * state  # Goal-conditioned gating
        
        ssm_out = self.C(state)
        
        return self.fusion(torch.cat([ssm_out, lang_feat, goal_init[:, :64]], dim=-1))


class GoalConditionedAttention(nn.Module):
    """Goal-conditioned attention baseline for comparison.
    
    Based on H1.202: Task structure enables Attention (+89.7%).
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, goal_dim=8, hidden=256):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.goal_enc = nn.Sequential(nn.Linear(goal_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        
        self.attn = nn.MultiheadAttention(hidden + 64 + 64, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 64 + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang, goal):
        obs_feat = self.obs_enc(obs_seq)
        goal_feat = self.goal_enc(goal)
        lang_feat = self.lang_enc(lang)
        
        # Concatenate features at each timestep
        full_feat = torch.cat([
            obs_feat,
            goal_feat.unsqueeze(1).expand(-1, obs_feat.size(1), -1),
            lang_feat.unsqueeze(1).expand(-1, obs_feat.size(1), -1)
        ], dim=-1)
        
        # Attention pooling
        attn_out, _ = self.attn(full_feat, full_feat, full_feat)
        pooled = attn_out.mean(dim=1)
        
        return self.decoder(pooled)


def train_model(model, train_loader, val_loader, epochs=60, lr=3e-4, model_type="ssm"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            obs = batch["observations"]
            lang = batch["language"]
            goal = batch["goal_endpoint"]
            actions = batch["actions"][:, -1]
            
            if model_type == "ssm_goal":
                pred = model(obs, lang, goal)
            elif model_type == "attn_goal":
                pred = model(obs, lang, goal)
            else:
                pred = model(obs, lang)
            
            loss = criterion(pred, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            obs = batch["observations"]
            lang = batch["language"]
            goal = batch["goal_endpoint"]
            actions = batch["actions"][:, -1]
            
            if model_type == "ssm_goal":
                pred = model(obs, lang, goal)
            elif model_type == "attn_goal":
                pred = model(obs, lang, goal)
            else:
                pred = model(obs, lang)
            
            val_losses.append(criterion(pred, actions).item())
    
    return np.mean(val_losses)


def run_experiment():
    print("=" * 70)
    print("H3.101: SSM with Goal Conditioning on Manipulation Tasks")
    print("=" * 70)
    
    # Generate data
    print("\nGenerating manipulation data with goal states...")
    data = generate_manipulation_data(n_demos=400, task_type="manipulation")
    
    train_data = data[:320]
    val_data = data[320:400]
    
    # Test different sequence lengths
    seq_lengths = [20, 40, 60, 80, 100]
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        # Prepare data loaders
        class _Dataset(Dataset):
            def __init__(self, data, seq_len):
                self.data = data
                self.seq_len = seq_len
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                d = self.data[idx]
                sl = min(len(d["observations"]), self.seq_len)
                # Pad if needed
                obs = d["observations"][:sl]
                if len(obs) < self.seq_len:
                    obs = np.pad(obs, ((0, self.seq_len - len(obs)), (0, 0)), mode='edge')
                actions = d["actions"][:sl]
                if len(actions) < self.seq_len:
                    actions = np.pad(actions, ((0, self.seq_len - len(actions)), (0, 0)), mode='edge')
                return {
                    "observations": torch.tensor(obs, dtype=torch.float32),
                    "actions": torch.tensor(actions, dtype=torch.float32),
                    "language": torch.tensor(d["language_embedding"], dtype=torch.float32),
                    "goal_endpoint": torch.tensor(d["goal_endpoint"], dtype=torch.float32),
                }
        
        train_loader = DataLoader(_Dataset(train_data, seq_len), batch_size=16, shuffle=True)
        val_loader = DataLoader(_Dataset(val_data, seq_len), batch_size=16)
        
        # Train models
        ssm_base = SimpleSSM()
        ssm_base_loss = train_model(ssm_base, train_loader, val_loader, model_type="ssm")
        
        ssm_goal = SSMWithGoalConditioning()
        ssm_goal_loss = train_model(ssm_goal, train_loader, val_loader, model_type="ssm_goal")
        
        attn_goal = GoalConditionedAttention()
        attn_goal_loss = train_model(attn_goal, train_loader, val_loader, model_type="attn_goal")
        
        ssm_vs_base = (ssm_base_loss - ssm_goal_loss) / ssm_base_loss * 100
        ssm_vs_attn = (attn_goal_loss - ssm_goal_loss) / attn_goal_loss * 100
        
        print(f"  SSM (baseline): {ssm_base_loss:.6f}")
        print(f"  SSM + Goal:     {ssm_goal_loss:.6f} ({ssm_vs_base:+.1f}% vs SSM)")
        print(f"  Attn + Goal:    {attn_goal_loss:.6f}")
        print(f"  SSM+Goal vs Attn+Goal: {ssm_vs_attn:+.1f}%")
        
        results[seq_len] = {
            "ssm_base": float(ssm_base_loss),
            "ssm_goal": float(ssm_goal_loss),
            "attn_goal": float(attn_goal_loss),
            "ssm_goal_vs_ssm": float(ssm_vs_base),
            "ssm_goal_vs_attn": float(ssm_vs_attn),
        }
    
    # Summary
    avg_ssm_vs_base = np.mean([results[s]["ssm_goal_vs_ssm"] for s in seq_lengths])
    avg_ssm_vs_attn = np.mean([results[s]["ssm_goal_vs_attn"] for s in seq_lengths])
    
    out = {
        "hypothesis_id": "H3.101",
        "hypothesis": "SSM with goal conditioning outperforms vanilla SSM and goal-attention",
        "results_by_length": results,
        "avg_ssm_goal_vs_ssm": float(avg_ssm_vs_base),
        "avg_ssm_goal_vs_attn": float(avg_ssm_vs_attn),
        "status": "SUPPORTED" if avg_ssm_vs_base > 0 and avg_ssm_vs_attn > 0 else "PARTIAL",
        "note": f"SSM+Goal {'wins' if avg_ssm_vs_base > 0 else 'loses'} vs SSM ({avg_ssm_vs_base:+.1f}%), "
                f"{'wins' if avg_ssm_vs_attn > 0 else 'loses'} vs Attn+Goal ({avg_ssm_vs_attn:+.1f}%)",
    }
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for s in seq_lengths:
        r = results[s]
        print(f"  {s} steps: SSM={r['ssm_base']:.4f}, SSM+Goal={r['ssm_goal']:.4f}, Attn+Goal={r['attn_goal']:.4f}")
    print(f"\nSSM+Goal vs SSM: {avg_ssm_vs_base:+.1f}% avg")
    print(f"SSM+Goal vs Attn+Goal: {avg_ssm_vs_attn:+.1f}% avg")
    print(f"Status: {out['status']}")
    print(json.dumps(out, indent=2))
    
    return out


if __name__ == "__main__":
    results = run_experiment()