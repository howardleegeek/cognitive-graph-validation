#!/usr/bin/env python3
"""
H1.210: Bidirectional Goal-Conditioned Prediction

Previous findings:
- H3.94: Endpoint goal enables attention (+94.1%), complex representations hurt
- H3.95: Endpoint goal on 100+ steps (+95.3%), advantage grows with length
- H1.208: Combined goals (endpoint + subgoals) +46.9% on 300-500 steps

Hypothesis: Bidirectional prediction (forward from start + backward from goal)
will outperform unidirectional approaches by modeling both the path from start
and the path to goal simultaneously.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader


def generate_bidirectional_data(n_demos=400, with_goal=True):
    """Generate data with bidirectional structure for goal-conditioned prediction."""
    np.random.seed(44)
    data = []
    
    tasks = [
        "reach for and grasp the {obj}",
        "transport the {obj} to the {loc}",
        "place the {obj} in the {cnt}",
        "stack {obj1} on {obj2}",
    ]
    
    objects = ["red cube", "blue block", "green sphere", "yellow plate", "white bowl"]
    containers = ["basket", "bin", "box", "tray"]
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
        
        # Generate variable-length trajectory
        seq_len = np.random.randint(50, 151)
        
        # Autocorrelated observations (robot-like)
        rho = 0.85
        obs = np.zeros((seq_len, 8), dtype=np.float32)
        obs[0] = np.random.randn(8) * 0.3
        for t in range(1, seq_len):
            obs[t] = rho * obs[t-1] + np.random.randn(8) * np.sqrt(1 - rho**2) * 0.3
        obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
        obs[:, 7] = np.clip(obs[:, 7], 0, 1)
        
        # Reverse observations (for backward modeling)
        obs_reversed = obs[::-1].copy()
        
        # Actions
        actions = np.zeros((seq_len, 7), dtype=np.float32)
        actions[0] = np.random.randn(7) * 0.05
        for t in range(1, seq_len):
            actions[t] = rho * actions[t-1] + np.random.randn(7) * np.sqrt(1 - rho**2) * 0.05
        actions[:, 6] = np.clip(actions[:, 6], -1, 1)
        
        lang_emb = np.random.randn(32).astype(np.float32)
        goal_endpoint = obs[-1].copy() if with_goal else np.zeros(8, dtype=np.float32)
        
        data.append({
            "observations": obs,
            "observations_reversed": obs_reversed,
            "actions": actions,
            "language": lang,
            "language_embedding": lang_emb,
            "goal_endpoint": goal_endpoint,
            "task_id": i % 4,
            "seq_len": seq_len,
        })
    
    return data


class UnidirectionalModel(nn.Module):
    """Baseline: Forward-only prediction."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=256):
        super().__init__()
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden)
        )
        self.lang_enc = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        obs_feat = self.obs_enc(obs_seq.mean(dim=1))
        lang_feat = self.lang_enc(lang)
        return self.decoder(torch.cat([obs_feat, lang_feat], dim=-1))


class GoalConditionedUnidirectional(nn.Module):
    """Unidirectional with goal conditioning."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, goal_dim=8, hidden=256):
        super().__init__()
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden)
        )
        self.goal_enc = nn.Sequential(
            nn.Linear(goal_dim, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.lang_enc = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 64 + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang, goal):
        obs_feat = self.obs_enc(obs_seq.mean(dim=1))
        goal_feat = self.goal_enc(goal)
        lang_feat = self.lang_enc(lang)
        return self.decoder(torch.cat([obs_feat, goal_feat, lang_feat], dim=-1))


class BidirectionalGoalConditioned(nn.Module):
    """Bidirectional: Forward from start + Backward from goal.
    
    Key innovation: Model both temporal directions, letting the network
    understand both "where we're coming from" and "where we're going to".
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, goal_dim=8, hidden=256):
        super().__init__()
        # Forward encoder
        self.obs_enc_fwd = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden)
        )
        # Backward encoder
        self.obs_enc_bwd = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden)
        )
        self.goal_enc = nn.Sequential(
            nn.Linear(goal_dim, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.lang_enc = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Bidirectional attention
        total_dim = hidden * 2 + 64 + 64  # 640
        self.bi_attn = nn.MultiheadAttention(total_dim, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2 + 64 + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang, goal, obs_seq_rev=None):
        # Forward pass
        fwd_feat = self.obs_enc_fwd(obs_seq.mean(dim=1))  # [B, hidden]
        
        # Backward pass (reverse time)
        if obs_seq_rev is None:
            obs_rev = obs_seq.flip(1)
        else:
            obs_rev = obs_seq_rev
        bwd_feat = self.obs_enc_bwd(obs_rev.mean(dim=1))  # [B, hidden]
        
        goal_feat = self.goal_enc(goal)  # [B, 64]
        lang_feat = self.lang_enc(lang)  # [B, 64]
        
        # Concatenate for attention
        combined = torch.cat([fwd_feat, bwd_feat, goal_feat, lang_feat], dim=-1)  # [B, 2*hidden+128]
        
        # Self-attention to weight forward vs backward
        combined_expanded = combined.unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.bi_attn(combined_expanded, combined_expanded, combined_expanded)
        
        return self.decoder(attn_out.squeeze(1))


def train_model(model, train_loader, val_loader, epochs=60, lr=3e-4, model_type="uni"):
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
            
            if model_type == "bi":
                obs_rev = batch.get("observations_reversed")
                if obs_rev is not None:
                    pred = model(obs, lang, goal, obs_rev)
                else:
                    pred = model(obs, lang, goal, obs.flip(1))
            elif model_type == "uni_goal":
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
            
            if model_type == "bi":
                obs_rev = batch.get("observations_reversed")
                if obs_rev is not None:
                    pred = model(obs, lang, goal, obs_rev)
                else:
                    pred = model(obs, lang, goal, obs.flip(1))
            elif model_type == "uni_goal":
                pred = model(obs, lang, goal)
            else:
                pred = model(obs, lang)
            
            val_losses.append(criterion(pred, actions).item())
    
    return np.mean(val_losses)


class BiDirectionalDataset(Dataset):
    def __init__(self, data, seq_len=50):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        sl = len(d["observations"])
        
        # Sample starting point
        if sl > self.seq_len:
            start = np.random.randint(0, sl - self.seq_len)
            end = start + self.seq_len
        else:
            start = 0
            end = sl
        
        # Pad if needed
        obs = d["observations"][start:end]
        obs_rev = d["observations_reversed"][max(0, sl-end):max(0, sl-start)]
        actions = d["actions"][start:end]
        
        if len(obs) < self.seq_len:
            pad_len = self.seq_len - len(obs)
            obs = np.pad(obs, ((0, pad_len), (0, 0)), mode='edge')
            obs_rev = np.pad(obs_rev, ((0, pad_len), (0, 0)), mode='edge')
            actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='edge')
        
        return {
            "observations": torch.tensor(obs, dtype=torch.float32),
            "observations_reversed": torch.tensor(obs_rev, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "language": torch.tensor(d["language_embedding"], dtype=torch.float32),
            "goal_endpoint": torch.tensor(d["goal_endpoint"], dtype=torch.float32),
        }


def run_experiment():
    print("=" * 70)
    print("H1.210: Bidirectional Goal-Conditioned Prediction")
    print("=" * 70)
    
    # Generate data
    print("\nGenerating bidirectional goal-conditioned data...")
    data = generate_bidirectional_data(n_demos=400, with_goal=True)
    
    train_data = data[:320]
    val_data = data[320:400]
    
    # Test different sequence lengths
    seq_lengths = [50, 75, 100, 125, 150]
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        train_loader = DataLoader(
            BiDirectionalDataset(train_data, seq_len), batch_size=16, shuffle=True
        )
        val_loader = DataLoader(
            BiDirectionalDataset(val_data, seq_len), batch_size=16
        )
        
        # Train models
        uni = UnidirectionalModel()
        uni_loss = train_model(uni, train_loader, val_loader, model_type="uni")
        
        uni_goal = GoalConditionedUnidirectional()
        uni_goal_loss = train_model(uni_goal, train_loader, val_loader, model_type="uni_goal")
        
        bi = BidirectionalGoalConditioned()
        bi_loss = train_model(bi, train_loader, val_loader, model_type="bi")
        
        vs_uni = (uni_loss - bi_loss) / uni_loss * 100
        vs_uni_goal = (uni_goal_loss - bi_loss) / uni_goal_loss * 100
        
        print(f"  Unidirectional:     {uni_loss:.6f}")
        print(f"  Unidirectional+Goal: {uni_goal_loss:.6f}")
        print(f"  Bidirectional:     {bi_loss:.6f}")
        print(f"  Bidir vs Unidir: {vs_uni:+.1f}%")
        print(f"  Bidir vs Unidir+Goal: {vs_uni_goal:+.1f}%")
        
        results[seq_len] = {
            "unidirectional": float(uni_loss),
            "unidirectional_goal": float(uni_goal_loss),
            "bidirectional": float(bi_loss),
            "vs_unidirectional": float(vs_uni),
            "vs_unidirectional_goal": float(vs_uni_goal),
        }
    
    # Summary
    avg_vs_uni = np.mean([results[s]["vs_unidirectional"] for s in seq_lengths])
    avg_vs_uni_goal = np.mean([results[s]["vs_unidirectional_goal"] for s in seq_lengths])
    
    out = {
        "hypothesis_id": "H1.210",
        "hypothesis": "Bidirectional prediction outperforms unidirectional with goal conditioning",
        "results_by_length": results,
        "avg_bidir_vs_uni": float(avg_vs_uni),
        "avg_bidir_vs_uni_goal": float(avg_vs_uni_goal),
        "status": "SUPPORTED" if avg_vs_uni_goal > 0 else "REFUTED",
        "note": f"Bidirectional {'wins' if avg_vs_uni_goal > 0 else 'loses'} {sum(results[s]['bidirectional'] < results[s]['unidirectional_goal'] for s in seq_lengths)}/{len(seq_lengths)} lengths vs goal-uni, avg {avg_vs_uni_goal:+.1f}%",
    }
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for s in seq_lengths:
        r = results[s]
        print(f"  {s} steps: Uni={r['unidirectional']:.4f}, Uni+Goal={r['unidirectional_goal']:.4f}, Bi={r['bidirectional']:.4f}")
    print(f"\nBidir vs Unidir: {avg_vs_uni:+.1f}% avg")
    print(f"Bidir vs Unidir+Goal: {avg_vs_uni_goal:+.1f}% avg")
    print(f"Status: {out['status']}")
    print(json.dumps(out, indent=2))
    
    return out


if __name__ == "__main__":
    results = run_experiment()