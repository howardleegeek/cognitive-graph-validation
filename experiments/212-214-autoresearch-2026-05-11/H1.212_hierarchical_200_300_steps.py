#!/usr/bin/env python3
"""
H1.212: Hierarchical Goal Reasoning on 200-300 Step Sequences
Bridging H1.209 (REFUTED at 100-200, flat wins) and H1.211 (SUPPORTED at 300-500).
"""

import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

class FlatGoalEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fusion_dim = obs_dim + action_dim
        self.obs_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.goal_proj = nn.Linear(obs_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, obs_seq, action_seq, goal):
        x = torch.cat([obs_seq, action_seq], dim=-1)
        x = self.obs_proj(x)
        _, h_n = self.gru(x)
        h = h_n.squeeze(0)
        if goal is not None:
            goal_emb = self.goal_proj(goal).squeeze(0)
            h = h + goal_emb
        out = self.predictor(h)
        return out.squeeze(0)

class HierarchicalEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, n_levels=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fusion_dim = obs_dim + action_dim
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.obs_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.level_grus = nn.ModuleList([nn.GRU(hidden_dim, hidden_dim, batch_first=True) for _ in range(n_levels)])
        self.goal_proj = nn.ModuleList([nn.Linear(obs_dim, hidden_dim) for _ in range(n_levels)])
        self.predictor = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, obs_seq, action_seq, goal):
        x = torch.cat([obs_seq, action_seq], dim=-1)
        x = self.obs_proj(x)
        
        level_outputs = []
        for level in range(self.n_levels):
            _, h_n = self.level_grus[level](x)
            h = h_n.squeeze(0)
            if goal is not None:
                goal_emb = self.goal_proj[level](goal).squeeze(0)
                h = h + goal_emb
            level_outputs.append(h)
        
        fused = torch.stack(level_outputs, dim=-1).mean(dim=-1)
        out = self.predictor(fused)
        return out.squeeze(0)

def generate_manipulation_trajectory(T, rho=0.85, complexity=0.5):
    n_objects = 3
    n_goals = max(2, int(complexity * T / 50))
    goal_indices = np.linspace(T // n_goals, T, n_goals, dtype=int)
    
    positions = []
    actions = []
    goals = []
    
    for t in range(T):
        pos = np.zeros(n_objects * 3)
        for i in range(n_objects):
            phase = 2 * np.pi * i / n_objects + t * 0.1
            pos[3*i] = 0.5 * np.sin(phase) + 0.1 * np.random.randn()
            pos[3*i+1] = 0.5 * np.cos(phase) + 0.1 * np.random.randn()
            pos[3*i+2] = 0.1 + 0.05 * np.sin(t * 0.2)
        
        if t > 0 and t in goal_indices:
            goal_idx = np.where(goal_indices == t)[0][0]
            current_goal = pos.copy()
        elif t > 0:
            current_goal = goals[-1].copy()
        else:
            current_goal = pos.copy()
        
        if t < T - 1:
            action = np.random.randn(n_objects * 3) * 0.1
        else:
            action = np.zeros(n_objects * 3)
        
        positions.append(pos)
        actions.append(action)
        goals.append(current_goal)
    
    positions = np.array(positions)
    actions = np.array(actions)
    goals = np.array(goals)
    
    obs = positions
    
    if rho > 0:
        noise = np.random.randn(*obs.shape) * 0.1
        noisy_obs = obs.copy()
        for i in range(1, len(obs)):
            noisy_obs[i] = rho * noisy_obs[i-1] + (1 - rho) * obs[i] + np.sqrt(1 - rho**2) * noise[i]
        obs = noisy_obs
    
    return obs, actions, goals

def train_and_evaluate(model, train_data, val_data, epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_obs, train_acts, train_goals, train_targets = train_data
    val_obs, val_acts, val_goals, val_targets = val_data
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_obs))[:30]
        for idx in indices:
            optimizer.zero_grad()
            goal_np = np.array([train_goals[idx][-1]])
            obs_t = torch.FloatTensor(train_obs[idx])
            act_t = torch.FloatTensor(train_acts[idx])
            goal_t = torch.FloatTensor(goal_np)
            pred = model(obs_t.unsqueeze(0), act_t.unsqueeze(0), goal_t)
            target = torch.FloatTensor(train_targets[idx])
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = []
        for i in range(len(val_obs)):
            goal_np = np.array([val_goals[i][-1]])
            obs_t = torch.FloatTensor(val_obs[i])
            act_t = torch.FloatTensor(val_acts[i])
            goal_t = torch.FloatTensor(goal_np)
            pred = model(obs_t.unsqueeze(0), act_t.unsqueeze(0), goal_t)
            val_preds.append(pred.numpy())
    
    val_preds = np.array(val_preds)
    val_targets_arr = np.array(val_targets)
    mse = np.mean((val_preds - val_targets_arr)**2)
    return mse

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = {"hypothesis": "H1.212", "description": "Hierarchical on 200-300 step sequences", "results": []}
    
    test_lengths = [200, 225, 250, 275, 300]
    
    for T in test_lengths:
        print(f"\n=== Testing {T} step sequences ===")
        
        n_train = 50
        n_val = 15
        n_objects = 3
        obs_dim = n_objects * 3  # 9
        action_dim = n_objects * 3  # 9
        hidden_dim = 128
        
        train_obs, train_acts, train_goals, train_targets = [], [], [], []
        val_obs, val_acts, val_goals, val_targets = [], [], [], []
        
        for i in range(n_train):
            obs, actions, goals = generate_manipulation_trajectory(T, rho=0.85, complexity=0.6)
            train_obs.append(obs)
            train_acts.append(actions)
            train_goals.append(goals)
            target_idx = np.random.randint(T // 2, T)
            train_targets.append(obs[target_idx])
        
        for i in range(n_val):
            obs, actions, goals = generate_manipulation_trajectory(T, rho=0.85, complexity=0.6)
            val_obs.append(obs)
            val_acts.append(actions)
            val_goals.append(goals)
            target_idx = np.random.randint(T // 2, T)
            val_targets.append(obs[target_idx])
        
        train_data = (train_obs, train_acts, train_goals, train_targets)
        val_data = (val_obs, val_acts, val_goals, val_targets)
        
        flat_model = FlatGoalEncoder(obs_dim, action_dim, hidden_dim)
        flat_mse = train_and_evaluate(flat_model, train_data, val_data)
        
        hier_model = HierarchicalEncoder(obs_dim, action_dim, hidden_dim, n_levels=3)
        hier_mse = train_and_evaluate(hier_model, train_data, val_data)
        
        delta = (hier_mse - flat_mse) / flat_mse * 100
        
        result = {
            "length": T,
            "flat_mse": float(flat_mse),
            "hier_mse": float(hier_mse),
            "delta_percent": float(delta),
            "winner": "hierarchical" if delta < 0 else "flat"
        }
        results["results"].append(result)
        
        print(f"  Flat MSE: {flat_mse:.6f}")
        print(f"  Hier MSE: {hier_mse:.6f}")
        print(f"  Delta: {delta:+.1f}% -> {result['winner'].upper()}")
    
    flat_mses = [r["flat_mse"] for r in results["results"]]
    hier_mses = [r["hier_mse"] for r in results["results"]]
    
    avg_flat = np.mean(flat_mses)
    avg_hier = np.mean(hier_mses)
    avg_delta = np.mean([r["delta_percent"] for r in results["results"]])
    hier_wins = sum(1 for r in results["results"] if r["winner"] == "hierarchical")
    
    results["summary"] = {
        "avg_flat_mse": float(avg_flat),
        "avg_hier_mse": float(avg_hier),
        "avg_delta": float(avg_delta),
        "hier_wins": hier_wins,
        "total": len(test_lengths)
    }
    
    if hier_wins >= 3 and avg_delta < 0:
        results["status"] = "SUPPORTED"
    elif hier_wins <= 1:
        results["status"] = "REFUTED"
    else:
        results["status"] = "PARTIAL"
    
    print(f"\n=== H1.212 Summary ===")
    print(f"  Avg Flat MSE: {avg_flat:.6f}")
    print(f"  Avg Hier MSE: {avg_hier:.6f}")
    print(f"  Avg Delta: {avg_delta:+.1f}%")
    print(f"  Hier Wins: {hier_wins}/{len(test_lengths)}")
    print(f"  Status: {results['status']}")
    
    output_dir = Path(__file__).parent
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    return results

if __name__ == "__main__":
    main()