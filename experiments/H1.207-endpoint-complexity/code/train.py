#!/usr/bin/env python3
"""
H1.207: Endpoint Goal with Different Task Complexities
Test how endpoint goal enables attention across different complexity levels
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

np.random.seed(42)
torch.manual_seed(42)


class EndpointGoalDataset:
    """Dataset with endpoint goal representation"""
    def __init__(self, n_samples, seq_len, state_dim=8, action_dim=7, rho=0.85):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.state_dim = state_dim
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
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(256, 256)
        self.value = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        Q, K, V = self.query(encoded), self.key(encoded), self.value(encoded)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / 16
        attn = torch.softmax(scores, dim=-1)
        out = self.output(torch.matmul(attn, V)[:, -1])
        return out


class ConcatModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
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
    print("H1.207: Endpoint Goal with Different Task Complexities")
    print("=" * 70)
    
    results = []
    complexities = [
        {'state_dim': 4, 'action_dim': 3, 'name': 'simple'},
        {'state_dim': 8, 'action_dim': 7, 'name': 'medium'},
        {'state_dim': 16, 'action_dim': 12, 'name': 'complex'},
        {'state_dim': 24, 'action_dim': 16, 'name': 'very_complex'},
    ]
    seq_len = 80
    n_samples = 200
    
    for complexity in complexities:
        print(f"\n--- Testing {complexity['name']} (state_dim={complexity['state_dim']}, action_dim={complexity['action_dim']}) ---", flush=True)
        
        dataset = EndpointGoalDataset(
            n_samples=n_samples, 
            seq_len=seq_len,
            state_dim=complexity['state_dim'],
            action_dim=complexity['action_dim']
        )
        X, y = dataset.generate()
        
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        concat_mse = evaluate(
            train_model(ConcatModel(X_train.shape[-1], complexity['state_dim']), X_train, y_train), 
            X_test, y_test
        )
        attn_mse = evaluate(
            train_model(AttentionModel(X_train.shape[-1], complexity['state_dim']), X_train, y_train), 
            X_test, y_test
        )
        
        delta = ((concat_mse - attn_mse) / concat_mse) * 100
        
        results.append({
            'complexity': complexity['name'],
            'state_dim': complexity['state_dim'],
            'action_dim': complexity['action_dim'],
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
        'experiment': 'H1.207',
        'description': 'Endpoint goal with different task complexities',
        'avg_delta': float(avg_delta),
        'attention_wins': attn_wins,
        'status': 'SUPPORTED' if avg_delta > 0 and attn_wins >= len(results) * 0.5 else 'REFUTED',
        'per_complexity': {r['complexity']: r for r in results},
    }
    
    print("\n" + "=" * 70)
    print("RESULTS:", json.dumps(final, indent=2))
    
    return final


if __name__ == "__main__":
    main()