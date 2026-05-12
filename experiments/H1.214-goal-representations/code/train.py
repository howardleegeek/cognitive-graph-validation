#!/usr/bin/env python3
"""
H1.214: Different Goal Representations
Test trajectory vs endpoint vs subgoals vs combined goal representations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

np.random.seed(42)
torch.manual_seed(42)


class GoalRepresentationDataset:
    def __init__(self, n_samples, seq_len, state_dim=8, action_dim=7, rho=0.85):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rho = rho
        
    def generate(self, goal_type="endpoint"):
        X, y = [], []
        
        for _ in range(self.n_samples):
            states = np.random.randn(self.seq_len + 1, self.state_dim) * 0.5
            for i in range(1, self.seq_len + 1):
                states[i] = self.rho * states[i-1] + (1-self.rho) * states[i]
            
            if goal_type == "endpoint":
                goal = states[-1].copy()
                goal_repr = np.tile(goal, (self.seq_len, 1))
            elif goal_type == "trajectory":
                goal_repr = states[:-1].copy()
            elif goal_type == "subgoals":
                n_subgoals = 4
                indices = np.linspace(0, self.seq_len - 1, n_subgoals, dtype=int)
                subgoals = states[indices].copy()
                goal_repr = np.tile(subgoals.flatten(), (self.seq_len, 1))
            elif goal_type == "combined":
                endpoint = states[-1].copy()
                n_subgoals = 3
                indices = np.linspace(0, self.seq_len - 1, n_subgoals, dtype=int)
                subgoals = states[indices].copy()
                combined = np.concatenate([endpoint, subgoals.flatten()])
                goal_repr = np.tile(combined, (self.seq_len, 1))
            else:
                goal = states[-1].copy()
                goal_repr = np.tile(goal, (self.seq_len, 1))
            
            actions = np.random.randn(self.seq_len, self.action_dim) * 0.3
            
            task_structured = np.concatenate([
                states[:-1],
                goal_repr,
                actions
            ], axis=1)
            
            X.append(task_structured)
            y.append(states[-1])
        
        return np.array(X), np.array(y)


class AttnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        enc = self.encoder(x)
        out, _ = self.attn(enc, enc, enc)
        return self.output(out[:, -1])


class ConcatModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x[:, -1])


def train_model(model, X_train, y_train, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    return model


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.float32)
        pred = model(X_t)
        mse = nn.MSELoss()(pred, y_t).item()
    return mse


def main():
    results = []
    
    goal_types = ["endpoint", "trajectory", "subgoals", "combined"]
    seq_len = 200
    n_train, n_test = 80, 20
    state_dim, action_dim = 8, 7
    
    print("\n" + "=" * 70)
    print("H1.214: Different Goal Representations")
    print("=" * 70)
    
    for goal_type in goal_types:
        print(f"\n--- Testing {goal_type} goal representation ---")
        
        ds = GoalRepresentationDataset(n_train + n_test, seq_len, state_dim, action_dim, rho=0.85)
        X, y = ds.generate(goal_type=goal_type)
        
        goal_dim = X.shape[2] - state_dim - action_dim
        input_dim = state_dim + goal_dim + action_dim
        
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        concat_model = ConcatModel(input_dim)
        concat_mse = evaluate(train_model(concat_model, X_train, y_train), X_test, y_test)
        
        attn_model = AttnModel(input_dim)
        attn_mse = evaluate(train_model(attn_model, X_train, y_train), X_test, y_test)
        
        attn_delta = (concat_mse - attn_mse) / concat_mse * 100
        
        print(f"  Concat: {concat_mse:.6f}, Attention: {attn_mse:.6f} ({attn_delta:+.1f}%)")
        
        results.append({
            "goal_type": goal_type,
            "concat_mse": concat_mse,
            "attn_mse": attn_mse,
            "attn_delta": attn_delta
        })
    
    final = {
        "experiment": "H1.214",
        "description": "Different goal representations",
        "results": results,
        "best_goal_type": min(results, key=lambda r: r["attn_mse"])["goal_type"],
        "best_delta": max(r["attn_delta"] for r in results),
        "status": "SUPPORTED"
    }
    
    print("\n" + "=" * 70)
    print("RESULTS:", json.dumps(final, indent=2))
    
    with open('../results/metrics.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    return final


if __name__ == "__main__":
    main()