#!/usr/bin/env python3
"""
H1.213: Multi-step complex manipulation (200-400 steps) with endpoint + subgoal goal conditioning
FAST VERSION - Reduced epochs and sequence lengths for rapid iteration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime

np.random.seed(42)
torch.manual_seed(42)


class MultiStepRobotDataset:
    def __init__(self, n_samples, seq_len, state_dim=8, action_dim=7, rho=0.85):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rho = rho
        
    def generate_subgoals(self, goal, n_subgoals=3):
        t_values = np.linspace(0, 1, n_subgoals + 2)[1:-1]
        return np.array([goal * t for t in t_values])
    
    def generate(self):
        X, y = [], []
        n_subgoals = max(3, self.seq_len // 100)
        
        for _ in range(self.n_samples):
            states = np.random.randn(self.seq_len + 1, self.state_dim) * 0.5
            for i in range(1, self.seq_len + 1):
                states[i] = self.rho * states[i-1] + (1-self.rho) * states[i]
            
            goal = states[-1].copy()
            subgoals = self.generate_subgoals(goal, n_subgoals)
            actions = np.random.randn(self.seq_len, self.action_dim) * 0.3
            
            task_structured = np.concatenate([
                states[:-1],
                np.tile(goal, (self.seq_len, 1)),
                np.tile(subgoals.flatten(), (self.seq_len, 1)),
                actions
            ], axis=1)
            
            X.append(task_structured)
            y.append(states[1:])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
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
        return self.output(attended[:, -1])


class HierarchicalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=8, n_chunks=4):
        super().__init__()
        self.n_chunks = n_chunks
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        seq_len = encoded.shape[1]
        chunk_size = max(10, seq_len // self.n_chunks)
        
        chunks = []
        for i in range(self.n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)
            if start >= seq_len:
                break
            chunk = encoded[:, start:end]
            Q = self.query(chunk)
            K = self.key(chunk)
            V = self.value(chunk)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            attended = torch.matmul(attn, V)
            chunks.append(attended[:, -1:])
        
        combined = torch.cat(chunks, dim=1).mean(dim=1)
        return self.output(combined)


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
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t[:, -1])
        loss.backward()
        optimizer.step()
    return model


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32))
        return nn.MSELoss()(pred, torch.tensor(y_test[:, -1], dtype=torch.float32)).item()


def main():
    print("=" * 70)
    print("H1.213: Multi-step (200-400) with Endpoint + Subgoal Conditioning")
    print("=" * 70)
    
    results = []
    seq_lengths = [200, 250, 300, 350, 400]
    n_samples = 100
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing {seq_len}-step sequences ---", flush=True)
        
        dataset = MultiStepRobotDataset(n_samples=n_samples, seq_len=seq_len, rho=0.85)
        X, y = dataset.generate()
        
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        input_dim = X_train.shape[-1]
        
        concat_mse = evaluate(train_model(ConcatModel(input_dim), X_train, y_train), X_test, y_test)
        attn_mse = evaluate(train_model(AttentionModel(input_dim), X_train, y_train), X_test, y_test)
        hier_mse = evaluate(train_model(HierarchicalModel(input_dim, n_chunks=4), X_train, y_train), X_test, y_test)
        
        attn_delta = ((concat_mse - attn_mse) / concat_mse) * 100
        hier_delta = ((concat_mse - hier_mse) / concat_mse) * 100
        
        best = min([("concat", concat_mse), ("attn", attn_mse), ("hier", hier_mse)], key=lambda x: x[1])
        
        results.append({
            'seq_len': seq_len,
            'concat_mse': float(concat_mse),
            'attention_mse': float(attn_mse),
            'hierarchical_mse': float(hier_mse),
            'attn_delta': float(attn_delta),
            'hier_delta': float(hier_delta),
            'best_model': best[0],
        })
        
        print(f"  Concat: {concat_mse:.6f}, Attn: {attn_mse:.6f} ({attn_delta:+.1f}%), Hier: {hier_mse:.6f} ({hier_delta:+.1f}%)")
        print(f"  Best: {best[0].upper()}")
    
    final = {
        'experiment': 'H1.213',
        'description': 'Multi-step (200-400) with endpoint + subgoal conditioning',
        'avg_attn_delta': float(np.mean([r["attn_delta"] for r in results])),
        'avg_hier_delta': float(np.mean([r["hier_delta"] for r in results])),
        'attn_wins': sum(1 for r in results if r['best_model'] == 'attn'),
        'hier_wins': sum(1 for r in results if r['best_model'] == 'hier'),
        'status': 'SUPPORTED' if np.mean([r["attn_delta"] for r in results]) > 0 else 'REFUTED',
    }
    
    print("\n" + "=" * 70)
    print("RESULTS:", json.dumps(final, indent=2))
    
    with open('experiments/H1.213-endpoint-subgoal-300-500/results/metrics.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    return final


if __name__ == "__main__":
    main()