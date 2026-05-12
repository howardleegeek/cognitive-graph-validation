#!/usr/bin/env python3
"""
H3.102: SSM + Goal Conditioning on Short Sequences (≤40 steps)
H3.101 showed SSM+Goal loses to vanilla SSM overall BUT +17.2% at short sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

class SSMEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fusion_dim = obs_dim + action_dim
        self.x_proj = nn.Linear(self.fusion_dim, hidden_dim)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.goal_proj = nn.Linear(obs_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, obs_seq, action_seq, goal):
        x = torch.cat([obs_seq, action_seq], dim=-1)
        x = self.x_proj(x)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        for t in range(x.size(1)):
            h = torch.matmul(x[:, t], self.A) + torch.matmul(h, self.B)
        if goal is not None:
            goal_emb = self.goal_proj(goal).squeeze(0)
            h = h + goal_emb
        out = self.predictor(h)
        return out.squeeze(0)

class SSMGoalEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fusion_dim = obs_dim + action_dim
        self.x_proj = nn.Linear(self.fusion_dim, hidden_dim)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.goal_proj = nn.Linear(obs_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, obs_seq, action_seq, goal):
        x = torch.cat([obs_seq, action_seq], dim=-1)
        x = self.x_proj(x)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        for t in range(x.size(1)):
            h_new = torch.matmul(x[:, t], self.A) + torch.matmul(h, self.B)
            if goal is not None:
                goal_emb = self.goal_proj(goal).squeeze(0)
                h_new = h_new + 0.3 * goal_emb
            h = h_new
        if goal is not None:
            goal_emb = self.goal_proj(goal).squeeze(0)
            h = h + goal_emb
        out = self.predictor(h)
        return out.squeeze(0)

class AttnGoalEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fusion_dim = obs_dim + action_dim
        self.obs_proj = nn.Linear(self.fusion_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 2, batch_first=True)
        self.goal_proj = nn.Linear(obs_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, obs_seq, action_seq, goal):
        x = torch.cat([obs_seq, action_seq], dim=-1)
        x = self.obs_proj(x)
        if goal is not None:
            goal_emb = self.goal_proj(goal).unsqueeze(1)
            x = x + 0.5 * goal_emb.expand(-1, x.size(1), -1)
        attn_out, _ = self.attn(x, x, x)
        out = self.predictor(attn_out.mean(dim=1))
        return out.squeeze(0)

def generate_manipulation_trajectory(T, rho=0.85, complexity=0.5):
    n_objects = 3
    obs_dim = n_objects * 3
    n_goals = max(2, int(complexity * T / 50))
    goal_indices = np.linspace(T // n_goals, T, n_goals, dtype=int)
    
    positions, actions, goals = [], [], []
    for t in range(T):
        pos = np.zeros(obs_dim)
        for i in range(n_objects):
            phase = 2 * np.pi * i / n_objects + t * 0.1
            pos[3*i] = 0.5 * np.sin(phase) + 0.1 * np.random.randn()
            pos[3*i+1] = 0.5 * np.cos(phase) + 0.1 * np.random.randn()
            pos[3*i+2] = 0.1 + 0.05 * np.sin(t * 0.2)
        
        if t > 0 and t in goal_indices:
            current_goal = pos.copy()
        elif t > 0:
            current_goal = goals[-1].copy()
        else:
            current_goal = pos.copy()
        
        action = np.random.randn(obs_dim) * 0.1 if t < T - 1 else np.zeros(obs_dim)
        
        positions.append(pos)
        actions.append(action)
        goals.append(current_goal)
    
    positions = np.array(positions)
    if rho > 0:
        noisy_obs = positions.copy()
        for i in range(1, len(positions)):
            noisy_obs[i] = rho * noisy_obs[i-1] + (1 - rho) * positions[i]
        positions = noisy_obs
    
    return positions, np.array(actions), np.array(goals)

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
            pred = model(
                torch.FloatTensor(train_obs[idx]).unsqueeze(0),
                torch.FloatTensor(train_acts[idx]).unsqueeze(0),
                torch.FloatTensor(goal_np)
            )
            target = torch.FloatTensor(train_targets[idx])
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = []
        for i in range(len(val_obs)):
            goal_np = np.array([val_goals[i][-1]])
            pred = model(
                torch.FloatTensor(val_obs[i]).unsqueeze(0),
                torch.FloatTensor(val_acts[i]).unsqueeze(0),
                torch.FloatTensor(goal_np)
            )
            val_preds.append(pred.numpy())
    
    return np.mean((np.array(val_preds) - np.array(val_targets))**2)

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = {"hypothesis": "H3.102", "description": "SSM + Goal on short sequences (≤40)", "results": []}
    test_lengths = [20, 25, 30, 35, 40]
    
    for T in test_lengths:
        print(f"\n=== Testing {T} step sequences ===")
        n_train, n_val = 50, 15
        obs_dim, action_dim, hidden_dim = 9, 9, 128
        
        train_data, val_data = [], []
        for name, n in [("train", n_train), ("val", n_val)]:
            obs_list, acts_list, goals_list, targets_list = [], [], [], []
            for _ in range(n):
                obs, acts, goals = generate_manipulation_trajectory(T, rho=0.85, complexity=0.6)
                obs_list.append(obs)
                acts_list.append(acts)
                goals_list.append(goals)
                targets_list.append(obs[np.random.randint(T // 2, T)])
            if name == "train":
                train_data = (obs_list, acts_list, goals_list, targets_list)
            else:
                val_data = (obs_list, acts_list, goals_list, targets_list)
        
        ssm_mse = train_and_evaluate(SSMEncoder(obs_dim, action_dim, hidden_dim), train_data, val_data)
        ssm_goal_mse = train_and_evaluate(SSMGoalEncoder(obs_dim, action_dim, hidden_dim), train_data, val_data)
        attn_mse = train_and_evaluate(AttnGoalEncoder(obs_dim, action_dim, hidden_dim), train_data, val_data)
        
        ssm_goal_vs_ssm = (ssm_goal_mse - ssm_mse) / ssm_mse * 100
        ssm_goal_vs_attn = (ssm_goal_mse - attn_mse) / attn_mse * 100
        
        results["results"].append({
            "length": T, "ssm_mse": float(ssm_mse), "ssm_goal_mse": float(ssm_goal_mse),
            "attn_mse": float(attn_mse), "ssm_goal_vs_ssm": float(ssm_goal_vs_ssm),
            "ssm_goal_vs_attn": float(ssm_goal_vs_attn)
        })
        print(f"  SSM: {ssm_mse:.4f}, SSM+Goal: {ssm_goal_mse:.4f}, Attn: {attn_mse:.4f}")
        print(f"  SSM+Goal vs SSM: {ssm_goal_vs_ssm:+.1f}%, SSM+Goal vs Attn: {ssm_goal_vs_attn:+.1f}%")
    
    ssm_goal_vs_ssm_avg = np.mean([r["ssm_goal_vs_ssm"] for r in results["results"]])
    ssm_goal_vs_attn_avg = np.mean([r["ssm_goal_vs_attn"] for r in results["results"]])
    
    results["summary"] = {"avg_ssm_goal_vs_ssm": float(ssm_goal_vs_ssm_avg), "avg_ssm_goal_vs_attn": float(ssm_goal_vs_attn_avg)}
    results["status"] = "SUPPORTED" if ssm_goal_vs_ssm_avg < 0 or ssm_goal_vs_attn_avg < 0 else "REFUTED"
    
    print(f"\n=== H3.102 Summary ===")
    print(f"  SSM+Goal vs SSM: {ssm_goal_vs_ssm_avg:+.1f}%")
    print(f"  SSM+Goal vs Attn: {ssm_goal_vs_attn_avg:+.1f}%")
    print(f"  Status: {results['status']}")
    
    with open(Path(__file__).parent / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    main()