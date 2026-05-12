#!/usr/bin/env python3
"""
H3.94: Goal Representation Sensitivity Test
Test if different goal representations affect attention performance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

np.random.seed(42)
torch.manual_seed(42)


class GoalSensitivityDataset:
    """Dataset with different goal representations"""
    def __init__(self, n_samples, seq_len, goal_type='endpoint', rho=0.85):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.goal_type = goal_type
        self.rho = rho
        self.state_dim = 8
        self.action_dim = 7
        
    def generate(self):
        X, y = [], []
        for _ in range(self.n_samples):
            states = np.random.randn(self.seq_len + 1, self.state_dim) * 0.5
            
            for i in range(1, self.seq_len + 1):
                states[i] = self.rho * states[i-1] + (1-self.rho) * states[i]
            
            # Different goal representations
            if self.goal_type == 'endpoint':
                goal = states[-1].copy()
            elif self.goal_type == 'trajectory':
                goal = states.flatten()
            elif self.goal_type == 'keypoint':
                keypoints = states[::self.seq_len//5][:5].flatten()
                goal = keypoints
            elif self.goal_type == 'delta':
                goal = states[-1] - states[0]
            
            actions = np.random.randn(self.seq_len, self.action_dim) * 0.3
            
            task_structured = np.concatenate([
                states[:-1],
                np.tile(goal, (self.seq_len, 1)),
                actions
            ], axis=1)
            
            X.append(task_structured)
            y.append(states[1:])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class AttentionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(256, 256)
        self.value = nn.Linear(256, 256)
        self.output = nn.Linear(256, 8)
        
    def forward(self, x):
        encoded = self.encoder(x)
        Q, K, V = self.query(encoded), self.key(encoded), self.value(encoded)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / 16
        attn = torch.softmax(scores, dim=-1)
        out = self.output(torch.matmul(attn, V)[:, -1])
        return out


class ConcatModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )
        
    def forward(self, x):
        return self.net(x[:, -1])


def train_model(model, X_train, y_train, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t[:, -1])
        loss.backward()
        optimizer.step()
    
    return model


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        mse = nn.MSELoss()(model(torch.tensor(X_test, dtype=torch.float32)), torch.tensor(y_test, dtype=torch.float32)[:, -1]).item()
    return mse


def main():
    print("=" * 70)
    print("H3.94: Goal Representation Sensitivity Test")
    print("=" * 70)
    
    results = []
    goal_types = ['endpoint', 'trajectory', 'keypoint', 'delta']
    seq_len = 60
    n_samples = 200
    
    for goal_type in goal_types:
        print(f"\n--- Testing goal_type={goal_type} ---", flush=True)
        
        dataset = GoalSensitivityDataset(n_samples=n_samples, seq_len=seq_len, goal_type=goal_type)
        X, y = dataset.generate()
        
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        concat_mse = evaluate(train_model(ConcatModel(X_train.shape[-1]), X_train, y_train), X_test, y_test)
        attn_mse = evaluate(train_model(AttentionModel(X_train.shape[-1]), X_train, y_train), X_test, y_test)
        
        delta = ((concat_mse - attn_mse) / concat_mse) * 100
        
        results.append({
            'goal_type': goal_type,
            'concat_mse': float(concat_mse),
            'attention_mse': float(attn_mse),
            'delta': float(delta),
            'attention_wins': attn_mse < concat_mse,
        })
        
        winner = "ATTN" if attn_mse < concat_mse else "CONCAT"
        print(f"  Concat: {concat_mse:.6f}, Attn: {attn_mse:.6f}, Δ: {delta:+.1f}% ({winner})")
    
    avg_delta = np.mean([r['delta'] for r in results])
    attn_wins = sum(1 for r in results if r['attention_wins'])
    
    final = {
        'experiment': 'H3.94',
        'description': 'Goal representation sensitivity test',
        'avg_delta': float(avg_delta),
        'attention_wins': attn_wins,
        'status': 'SUPPORTED' if avg_delta > 0 and attn_wins >= len(results) * 0.5 else 'REFUTED',
        'per_type': {r['goal_type']: r for r in results},
    }
    
    print("\n" + "=" * 70)
    print("RESULTS:", json.dumps(final, indent=2))
    
    return final


if __name__ == "__main__":
    main()