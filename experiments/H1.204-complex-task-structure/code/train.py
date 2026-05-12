#!/usr/bin/env python3
"""
H1.204: Complex Multi-Step (50-100 steps) WITH Task Structure
Based on successful H3.91 pattern (+86.6%) - attention with goal conditioning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime

np.random.seed(42)
torch.manual_seed(42)


class TaskStructuredDataset:
    """Dataset with task structure: goal states, action outcomes"""
    def __init__(self, n_samples, seq_len, n_objects=2, state_dim=8, action_dim=7, rho=0.85):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_objects = n_objects
        self.state_dim = state_dim
        self.goal_dim = state_dim
        self.action_dim = action_dim
        self.rho = rho
        
    def generate(self):
        X, y = [], []
        for _ in range(self.n_samples):
            states = np.random.randn(self.seq_len + 1, self.state_dim) * 0.5
            
            for i in range(1, self.seq_len + 1):
                states[i] = self.rho * states[i-1] + (1-self.rho) * states[i]
            
            goal = states[-1].copy()
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
    def __init__(self, input_dim, hidden_dim=256, output_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        Q = self.query(encoded)
        K = self.key(encoded)
        V = self.value(encoded)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, V)
        out = self.output(attended[:, -1])
        return out


class ConcatModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x[:, -1])


def train_model(model, X_train, y_train, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t[:, -1])
        loss.backward()
        optimizer.step()
    
    return model


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.float32)
        pred = model(X_t)
        mse = nn.MSELoss()(pred, y_t[:, -1]).item()
    return mse


def main():
    print("=" * 70)
    print("H1.204: Complex Multi-Step (50-100) WITH Task Structure")
    print("=" * 70)
    
    results = []
    seq_lengths = [50, 60, 70, 80, 100]
    n_samples = 200
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing {seq_len}-step sequences ---", flush=True)
        
        dataset = TaskStructuredDataset(n_samples=n_samples, seq_len=seq_len, rho=0.85)
        X, y = dataset.generate()
        
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        input_dim = X_train.shape[-1]
        
        concat_model = ConcatModel(input_dim)
        concat_model = train_model(concat_model, X_train, y_train)
        concat_mse = evaluate(concat_model, X_test, y_test)
        
        attn_model = AttentionModel(input_dim)
        attn_model = train_model(attn_model, X_train, y_train)
        attn_mse = evaluate(attn_model, X_test, y_test)
        
        delta = ((concat_mse - attn_mse) / concat_mse) * 100
        
        results.append({
            'seq_len': seq_len,
            'concat_mse': float(concat_mse),
            'attention_mse': float(attn_mse),
            'delta': float(delta),
            'attention_wins': attn_mse < concat_mse,
        })
        
        winner = "ATTN" if attn_mse < concat_mse else "CONCAT"
        print(f"  Concat: {concat_mse:.6f}, Attn: {attn_mse:.6f}, Δ: {delta:+.1f}% ({winner})")
    
    deltas = [r["delta"] for r in results]
    avg_delta = np.mean(deltas)
    attn_wins = sum(1 for r in results if r['attention_wins'])
    
    final = {
        'experiment': 'H1.204',
        'description': 'Complex multi-step (50-100) WITH task structure',
        'avg_delta': float(avg_delta),
        'attention_wins': attn_wins,
        'status': 'SUPPORTED' if avg_delta > 0 and attn_wins >= len(results) * 0.5 else 'REFUTED',
        'per_length': {str(r['seq_len']): r for r in results},
    }
    
    print("\n" + "=" * 70)
    print("RESULTS:", json.dumps(final, indent=2))
    
    with open('experiments/H1.204-complex-task-structure/results/metrics.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    return final


if __name__ == "__main__":
    main()
