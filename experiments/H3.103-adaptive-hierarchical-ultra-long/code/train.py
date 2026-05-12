#!/usr/bin/env python3
"""
H3.103: Adaptive Hierarchical for Long Sequences (250-400 steps)
FAST VERSION - Reduced for rapid iteration
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
            n_levels = max(2, int(np.log2(self.seq_len / 80)))
            goal_repr = [goal.copy()]
            for level in range(n_levels):
                n_milestones = 2 ** level
                milestones = np.array([goal * (i + 1) / (n_milestones + 1) for i in range(n_milestones)])
                goal_repr.extend(milestones.tolist())
            goal_flat = np.array(goal_repr).flatten()
            
            actions = np.random.randn(self.seq_len, self.action_dim) * 0.3
            
            task_structured = np.concatenate([
                states[:-1],
                np.tile(goal_flat, (self.seq_len, 1)),
                actions
            ], axis=1)
            
            X.append(task_structured)
            y.append(states[1:])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class FlatAttention(nn.Module):
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
    def __init__(self, input_dim, hidden_dim=128, output_dim=8, n_levels=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.gate = nn.Linear(hidden_dim * n_levels, n_levels)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.n_levels = n_levels
        
    def forward(self, x):
        enc = self.encoder(x)
        seq_len = enc.shape[1]
        chunk_size = max(20, seq_len // self.n_levels)
        
        level_outputs = []
        for level in range(self.n_levels):
            start = level * chunk_size
            end = min((level + 1) * chunk_size, seq_len)
            if start >= seq_len:
                break
            chunk = enc[:, start:end]
            out, _ = self.attn(chunk, chunk, chunk)
            level_outputs.append(out.mean(dim=1))  # Use mean to ensure consistent dim
        
        if len(level_outputs) > 1:
            combined = torch.stack(level_outputs, dim=1)  # Stack instead of cat: (batch, n_levels, hidden)
            # Gate expects (batch, n_levels*hidden) input
            gate_input = combined.view(combined.size(0), -1)
            weights = torch.softmax(self.gate(gate_input), dim=1)  # (batch, n_levels)
            # Weighted sum: (batch, hidden) = sum over levels
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
    print("H3.103: Adaptive Hierarchical for Long Sequences (250-400)")
    print("=" * 70)
    
    results = []
    seq_lengths = [250, 300, 350, 400]
    n_samples = 100
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing {seq_len}-step sequences ---", flush=True)
        
        dataset = StructuredDataset(n_samples=n_samples, seq_len=seq_len, rho=0.85)
        X, y = dataset.generate()
        
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        input_dim = X_train.shape[-1]
        
        concat_mse = evaluate(train_model(ConcatModel(input_dim), X_train, y_train), X_test, y_test)
        flat_mse = evaluate(train_model(FlatAttention(input_dim), X_train, y_train), X_test, y_test)
        hier_mse = evaluate(train_model(HierarchicalAttn(input_dim, n_levels=3), X_train, y_train), X_test, y_test)
        
        flat_delta = ((concat_mse - flat_mse) / concat_mse) * 100
        hier_delta = ((concat_mse - hier_mse) / concat_mse) * 100
        
        best = min([("concat", concat_mse), ("flat", flat_mse), ("hier", hier_mse)], key=lambda x: x[1])
        
        results.append({
            'seq_len': seq_len,
            'concat_mse': float(concat_mse),
            'flat_mse': float(flat_mse),
            'hier_mse': float(hier_mse),
            'flat_delta': float(flat_delta),
            'hier_delta': float(hier_delta),
            'best_model': best[0],
        })
        
        print(f"  Concat: {concat_mse:.6f}, Flat: {flat_mse:.6f} ({flat_delta:+.1f}%), Hier: {hier_mse:.6f} ({hier_delta:+.1f}%)")
        print(f"  Best: {best[0].upper()}")
    
    final = {
        'experiment': 'H3.103',
        'description': 'Adaptive hierarchical for long sequences (250-400)',
        'avg_flat_delta': float(np.mean([r["flat_delta"] for r in results])),
        'avg_hier_delta': float(np.mean([r["hier_delta"] for r in results])),
        'hier_wins': sum(1 for r in results if r['best_model'] == 'hier'),
        'status': 'SUPPORTED' if np.mean([r["hier_delta"] for r in results]) > np.mean([r["flat_delta"] for r in results]) else 'REFUTED',
    }
    
    print("\n" + "=" * 70)
    print("RESULTS:", json.dumps(final, indent=2))
    
    with open('../results/metrics.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    return final


if __name__ == "__main__":
    main()