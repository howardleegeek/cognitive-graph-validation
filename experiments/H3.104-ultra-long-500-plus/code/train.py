#!/usr/bin/env python3
"""
H3.104: Attention on 500+ Step Ultra-Long Sequences
Test if attention mechanisms can scale to extreme sequence lengths
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

np.random.seed(42)
torch.manual_seed(42)


class StructuredDataset:
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


class FlatAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        enc = self.encoder(x)
        out, _ = self.attn(enc, enc, enc)
        return self.output(out[:, -1])


class HierarchicalAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=8, n_levels=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.gate = nn.Linear(hidden_dim * n_levels, n_levels)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.n_levels = n_levels
        
    def forward(self, x):
        enc = self.encoder(x)
        seq_len = enc.shape[1]
        chunk_size = max(50, seq_len // self.n_levels)
        
        level_outputs = []
        for level in range(self.n_levels):
            start = level * chunk_size
            end = min((level + 1) * chunk_size, seq_len)
            if start >= seq_len:
                break
            chunk = enc[:, start:end]
            out, _ = self.attn(chunk, chunk, chunk)
            level_outputs.append(out.mean(dim=1))
        
        if len(level_outputs) > 1:
            combined = torch.stack(level_outputs, dim=1)
            gate_input = combined.view(combined.size(0), -1)
            weights = torch.softmax(self.gate(gate_input), dim=1)
            weighted = (weights.unsqueeze(-1) * combined).sum(dim=1)
        else:
            weighted = level_outputs[0] if level_outputs else enc[:, -1]
        
        return self.output(weighted)


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
    
    seq_lengths = [500, 600, 700, 800, 1000]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing {seq_len}-step sequences ---")
        
        n_train, n_test = 80, 20
        state_dim, action_dim = 8, 7
        goal_dim = state_dim
        input_dim = state_dim + goal_dim + action_dim
        
        ds = StructuredDataset(n_train + n_test, seq_len, state_dim, action_dim, rho=0.85)
        X, y = ds.generate()
        
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        concat_model = ConcatModel(input_dim)
        concat_mse = evaluate(train_model(concat_model, X_train, y_train), X_test, y_test)
        
        flat_model = FlatAttn(input_dim)
        flat_mse = evaluate(train_model(flat_model, X_train, y_train), X_test, y_test)
        
        hier_model = HierarchicalAttn(input_dim, n_levels=4)
        hier_mse = evaluate(train_model(hier_model, X_train, y_train), X_test, y_test)
        
        flat_delta = (concat_mse - flat_mse) / concat_mse * 100
        hier_delta = (concat_mse - hier_mse) / concat_mse * 100
        
        best = "FLAT" if flat_mse < hier_mse else "HIER"
        
        print(f"  Concat: {concat_mse:.6f}, Flat: {flat_mse:.6f} ({flat_delta:+.1f}%), Hier: {hier_mse:.6f} ({hier_delta:+.1f}%)")
        print(f"  Best: {best}")
        
        results.append({
            "seq_len": seq_len,
            "concat_mse": concat_mse,
            "flat_mse": flat_mse,
            "hier_mse": hier_mse,
            "flat_delta": flat_delta,
            "hier_delta": hier_delta,
            "best_model": best
        })
    
    final = {
        "experiment": "H3.104",
        "description": "Attention on 500+ step ultra-long sequences",
        "results": results,
        "avg_flat_delta": float(np.mean([r["flat_delta"] for r in results])),
        "avg_hier_delta": float(np.mean([r["hier_delta"] for r in results])),
        "hier_wins": sum(1 for r in results if r['best_model'] == 'hier'),
        "status": "SUPPORTED" if np.mean([r["hier_delta"] for r in results]) > np.mean([r["flat_delta"] for r in results]) else "REFUTED"
    }
    
    print("\n" + "=" * 70)
    print("RESULTS:", json.dumps(final, indent=2))
    
    with open('../results/metrics.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    return final


if __name__ == "__main__":
    main()