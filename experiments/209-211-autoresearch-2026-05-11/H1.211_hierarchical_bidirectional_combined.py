#!/usr/bin/env python3
"""
H1.211: Hierarchical + Goal-Conditioned Combined on Extreme Complexity

Previous findings:
- H1.208: Ultra-long (300-500 steps) +46.9% with combined goals, +33.4% over endpoint
- H1.209: Hierarchical goal reasoning on 100-200 steps
- H3.100: Multi-scale goal decomposition: subgoal (+20.1%) best

Hypothesis: Combining hierarchical (multi-scale) goals with bidirectional reasoning
will achieve the best performance on extreme complexity (200-400 step sequences).
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader


def generate_extreme_complexity_data(n_demos=400):
    """Generate data for extreme complexity testing (200-400 step sequences)."""
    np.random.seed(45)
    data = []
    
    tasks = [
        "pick up {obj1}, transport to {loc1}, place in {cnt}, then retrieve {obj2} from {loc2}",
        "navigate to {loc1}, grasp {obj1}, carry past {loc2}, place in {cnt}",
        "rearrange {obj1} from {loc1} to {cnt}, move {obj2} from {loc2} to {loc1}",
    ]
    
    objects = ["red cube", "blue block", "green sphere", "yellow plate", "white bowl", "black cup"]
    containers = ["basket", "bin", "box", "tray", "shelf", "drawer"]
    locations = ["left corner", "right side", "center", "front area", "back region", "top shelf"]
    
    for i in range(n_demos):
        template = np.random.choice(tasks)
        lang = template.format(
            obj1=np.random.choice(objects),
            obj2=np.random.choice(objects),
            cnt=np.random.choice(containers),
            loc1=np.random.choice(locations),
            loc2=np.random.choice(locations),
        )
        
        # Generate very long trajectory (200-400 steps)
        seq_len = np.random.randint(200, 401)
        
        # Highly autocorrelated observations (robot-like, ρ=0.9 for longer sequences)
        rho = 0.9
        obs = np.zeros((seq_len, 8), dtype=np.float32)
        obs[0] = np.random.randn(8) * 0.25
        for t in range(1, seq_len):
            obs[t] = rho * obs[t-1] + np.random.randn(8) * np.sqrt(1 - rho**2) * 0.25
        obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
        obs[:, 7] = np.clip(obs[:, 7], 0, 1)
        
        # Actions with temporal coherence
        actions = np.zeros((seq_len, 7), dtype=np.float32)
        actions[0] = np.random.randn(7) * 0.05
        for t in range(1, seq_len):
            actions[t] = rho * actions[t-1] + np.random.randn(7) * np.sqrt(1 - rho**2) * 0.05
        actions[:, 6] = np.clip(actions[:, 6], -1, 1)
        
        lang_emb = np.random.randn(32).astype(np.float32)
        
        # Multi-scale goals
        goal_endpoint = obs[-1].copy()
        goals = {
            "endpoint": goal_endpoint,
            "milestone_25": obs[seq_len//4].copy(),
            "milestone_50": obs[seq_len//2].copy(),
            "milestone_75": obs[3*seq_len//4].copy(),
        }
        
        data.append({
            "observations": obs,
            "observations_reversed": obs[::-1].copy(),
            "actions": actions,
            "language": lang,
            "language_embedding": lang_emb,
            "goals": goals,
            "task_id": i % 3,
            "seq_len": seq_len,
        })
    
    return data


class EndpointGoalModel(nn.Module):
    """Baseline: Endpoint goal only."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, goal_dim=8, hidden=256):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.goal_enc = nn.Sequential(nn.Linear(goal_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 64))
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


class HierarchicalGoalModel(nn.Module):
    """Multi-scale hierarchical goals."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, goal_dim=8, hidden=256):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.goal_enc = nn.Sequential(nn.Linear(goal_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        
        # Encode 3 milestones + endpoint = 4 goals
        self.milestone_encs = nn.ModuleList([
            nn.Sequential(nn.Linear(goal_dim, 32), nn.ReLU()) for _ in range(4)
        ])
        
        # Simple weighted attention over goals
        self.goal_attn = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 64 + 64 + 32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang, goals_dict):
        obs_feat = self.obs_enc(obs_seq.mean(dim=1))
        goal_list = [goals_dict["endpoint"], goals_dict["milestone_25"], 
                     goals_dict["milestone_50"], goals_dict["milestone_75"]]
        goal_feats = [enc(g) for enc, g in zip(self.milestone_encs, goal_list)]
        goal_stack = torch.stack(goal_feats, dim=1)  # [B, 4, 32]
        
        # Attention weights over goals
        weights = F.softmax(self.goal_attn(goal_stack), dim=1)  # [B, 4, 1]
        goal_out = (goal_stack * weights).sum(dim=1)  # [B, 32]
        
        goal_feat = self.goal_enc(goal_list[0])  # endpoint
        lang_feat = self.lang_enc(lang)
        
        return self.decoder(torch.cat([obs_feat, goal_feat, lang_feat, goal_out], dim=-1))


class HierarchicalBidirectionalModel(nn.Module):
    """Combined: Hierarchical goals + Bidirectional reasoning."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, goal_dim=8, hidden=256):
        super().__init__()
        self.obs_enc_fwd = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.obs_enc_bwd = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden))
        self.goal_enc = nn.Sequential(nn.Linear(goal_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        
        self.milestone_encs = nn.ModuleList([
            nn.Sequential(nn.Linear(goal_dim, 32), nn.ReLU()) for _ in range(4)
        ])
        
        # Simple weighted attention
        self.goal_attn = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2 + 64 + 64 + 32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang, goals_dict, obs_seq_rev=None):
        obs_fwd = self.obs_enc_fwd(obs_seq.mean(dim=1))
        
        if obs_seq_rev is None:
            obs_rev = obs_seq.flip(1)
        else:
            obs_rev = obs_seq_rev
        obs_bwd = self.obs_enc_bwd(obs_rev.mean(dim=1))
        
        goal_list = [goals_dict["endpoint"], goals_dict["milestone_25"],
                     goals_dict["milestone_50"], goals_dict["milestone_75"]]
        goal_feats = [enc(g) for enc, g in zip(self.milestone_encs, goal_list)]
        goal_stack = torch.stack(goal_feats, dim=1)
        
        # Attention weights over goals
        weights = F.softmax(self.goal_attn(goal_stack), dim=1)
        goal_out = (goal_stack * weights).sum(dim=1)
        
        goal_feat = self.goal_enc(goal_list[0])
        lang_feat = self.lang_enc(lang)
        
        return self.decoder(torch.cat([obs_fwd, obs_bwd, goal_feat, lang_feat, goal_out], dim=-1))


def train_model(model, train_loader, val_loader, epochs=80, lr=3e-4, model_type="endpoint"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            obs = batch["observations"]
            lang = batch["language"]
            actions = batch["actions"][:, -1]
            
            if model_type == "hier_bi":
                obs_rev = batch.get("observations_reversed")
                if obs_rev is None:
                    obs_rev = obs.flip(1)
                pred = model(obs, lang, batch["goals"], obs_rev)
            elif model_type == "hier":
                pred = model(obs, lang, batch["goals"])
            else:
                pred = model(obs, lang, batch["goal_endpoint"])
            
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
            actions = batch["actions"][:, -1]
            
            if model_type == "hier_bi":
                obs_rev = batch.get("observations_reversed")
                if obs_rev is None:
                    obs_rev = obs.flip(1)
                pred = model(obs, lang, batch["goals"], obs_rev)
            elif model_type == "hier":
                pred = model(obs, lang, batch["goals"])
            else:
                goal_ep = batch["goal_endpoint"]
                if goal_ep.dim() == 1:
                    goal_ep = goal_ep.unsqueeze(0)
                if goal_ep.size(0) != obs.size(0):
                    goal_ep = goal_ep.expand(obs.size(0), -1)
                pred = model(obs, lang, goal_ep)
            
            val_losses.append(criterion(pred, actions).item())
    
    return np.mean(val_losses)


class ExtremeDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        sl = len(d["observations"])
        
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
            "goal_endpoint": torch.tensor(d["goals"]["endpoint"], dtype=torch.float32),
            "goals": {k: torch.tensor(v, dtype=torch.float32) for k, v in d["goals"].items()},
        }


def run_experiment():
    print("=" * 70)
    print("H1.211: Hierarchical + Goal-Conditioned Combined (Extreme Complexity)")
    print("=" * 70)
    
    # Generate extreme complexity data
    print("\nGenerating extreme complexity data (200-400 step sequences)...")
    data = generate_extreme_complexity_data(n_demos=400)
    
    train_data = data[:320]
    val_data = data[320:400]
    
    # Test different sequence lengths
    seq_lengths = [200, 250, 300, 350, 400]
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        train_loader = DataLoader(
            ExtremeDataset(train_data, seq_len), batch_size=16, shuffle=True
        )
        val_loader = DataLoader(
            ExtremeDataset(val_data, seq_len), batch_size=16
        )
        
        # Train models
        endpoint_model = EndpointGoalModel()
        endpoint_loss = train_model(endpoint_model, train_loader, val_loader, model_type="endpoint")
        
        hier_model = HierarchicalGoalModel()
        hier_loss = train_model(hier_model, train_loader, val_loader, model_type="hier")
        
        hier_bi_model = HierarchicalBidirectionalModel()
        hier_bi_loss = train_model(hier_bi_model, train_loader, val_loader, model_type="hier_bi")
        
        vs_endpoint = (endpoint_loss - hier_bi_loss) / endpoint_loss * 100
        vs_hier = (hier_loss - hier_bi_loss) / hier_loss * 100
        
        print(f"  Endpoint Goal:    {endpoint_loss:.6f}")
        print(f"  Hierarchical:     {hier_loss:.6f} ({vs_endpoint - (endpoint_loss-hier_loss)/endpoint_loss*100 + vs_endpoint:+.1f}% vs EP)")
        print(f"  Hier+Bi:         {hier_bi_loss:.6f} ({vs_endpoint:+.1f}% vs EP)")
        print(f"  Hier+Bi vs Hier: {vs_hier:+.1f}%")
        
        results[seq_len] = {
            "endpoint": float(endpoint_loss),
            "hierarchical": float(hier_loss),
            "hier_bidirectional": float(hier_bi_loss),
            "hier_bi_vs_endpoint": float(vs_endpoint),
            "hier_bi_vs_hier": float(vs_hier),
        }
    
    # Summary
    avg_vs_endpoint = np.mean([results[s]["hier_bi_vs_endpoint"] for s in seq_lengths])
    avg_vs_hier = np.mean([results[s]["hier_bi_vs_hier"] for s in seq_lengths])
    
    out = {
        "hypothesis_id": "H1.211",
        "hypothesis": "Hierarchical + Bidirectional combined on extreme complexity (200-400 steps)",
        "results_by_length": results,
        "avg_hier_bi_vs_endpoint": float(avg_vs_endpoint),
        "avg_hier_bi_vs_hierarchical": float(avg_vs_hier),
        "status": "SUPPORTED" if avg_vs_endpoint > 0 else "REFUTED",
        "note": f"Hier+Bi {'wins' if avg_vs_endpoint > 0 else 'loses'} {sum(results[s]['hier_bidirectional'] < results[s]['endpoint'] for s in seq_lengths)}/{len(seq_lengths)} lengths vs endpoint, avg {avg_vs_endpoint:+.1f}%",
    }
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for s in seq_lengths:
        r = results[s]
        print(f"  {s} steps: EP={r['endpoint']:.4f}, Hier={r['hierarchical']:.4f}, Hier+Bi={r['hier_bidirectional']:.4f}")
    print(f"\nHier+Bi vs Endpoint: {avg_vs_endpoint:+.1f}% avg")
    print(f"Hier+Bi vs Hierarchical: {avg_vs_hier:+.1f}% avg")
    print(f"Status: {out['status']}")
    print(json.dumps(out, indent=2))
    
    return out


if __name__ == "__main__":
    results = run_experiment()