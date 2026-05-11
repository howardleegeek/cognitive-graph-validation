#!/usr/bin/env python3
"""
H3.89: Attention on Longer Sequences with Real Robot Autocorrelation
Building on:
- H3.69: +34.2% on 20-30 timesteps (attention wins)
- H3.70: -34.6% on 30-50 timesteps (attention loses)
- H1.193: SSM (+97.6%) with autocorrelation

Hypothesis: With real robot autocorrelation (ρ=0.85), attention will
outperform concatenation across ALL tested sequence lengths.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ExperimentConfig:
    n_samples: int = 200
    n_features: int = 12
    hidden_dim: int = 256
    learning_rate: float = 0.001
    epochs: int = 100
    autocorrelation: float = 0.85


class RealRobotSequenceDataset(Dataset):
    """Dataset with real robot-like temporal structure."""
    
    def __init__(self, n_samples=200, timesteps=50, n_features=12, autocorrelation=0.85):
        self.n_samples = n_samples
        self.timesteps = timesteps
        self.n_features = n_features
        self.autocorrelation = autocorrelation
        np.random.seed(42)
        self.data = []
        for _ in range(n_samples):
            seq = self._generate_robot_like_sequence(timesteps, n_features, autocorrelation)
            self.data.append(seq)
        self.data = np.array(self.data, dtype=np.float32)
    
    def _generate_robot_like_sequence(self, T, n_features, rho):
        """Generate sequence with real robot-like temporal structure."""
        seq = np.zeros((T, n_features))
        for i in range(n_features // 2):
            t = np.linspace(0, 1, T)
            freq = np.random.uniform(1.0, 3.0)
            phase = np.random.uniform(0, np.pi)
            base = np.sin(2 * np.pi * freq * t + phase)
            
            for t_idx in range(T):
                if t_idx == 0:
                    seq[t_idx, i*2] = base[t_idx]
                else:
                    seq[t_idx, i*2] = rho * seq[t_idx-1, i*2] + (1 - rho) * base[t_idx]
            
            seq[:, i*2+1] = np.diff(np.concatenate([[0], seq[:, i*2]]))
        
        return seq
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class ConcatModel(nn.Module):
    """Concatenation baseline model."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.decoder(pooled)


class AttentionModel(nn.Module):
    """Attention-based model."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        attn_out, _ = self.attention(encoded, encoded, encoded)
        return self.decoder(attn_out.mean(dim=1))


def train_model(model, train_loader, val_loader, epochs=100):
    """Train model and return best validation MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            target = batch[:, -1, :]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                target = batch[:, -1, :]
                val_losses.append(criterion(output, target).item())
        
        val_mse = np.mean(val_losses)
        if val_mse < best_val:
            best_val = val_mse
    
    return best_val


def run_experiment():
    print("=" * 70)
    print("H3.89: Attention on Longer Sequences with Real Robot Autocorrelation")
    print("=" * 70)
    
    config = ExperimentConfig()
    
    results = {
        'experiment': 'H3.89',
        'timestamp': datetime.now().isoformat(),
        'parent': 'H3',
        'timesteps': [10, 15, 20, 25, 30, 40, 50],
        'concat_mses': [],
        'attn_mses': [],
        'delta': []
    }
    
    for timesteps in results['timesteps']:
        print(f"\n--- Testing {timesteps}-step tasks (ρ={config.autocorrelation}) ---")
        
        full_dataset = RealRobotSequenceDataset(
            n_samples=config.n_samples,
            timesteps=timesteps,
            n_features=config.n_features,
            autocorrelation=config.autocorrelation
        )
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        input_dim = config.n_features
        output_dim = config.n_features
        hidden_dim = config.hidden_dim
        
        # Concatenation baseline
        concat_model = ConcatModel(input_dim, hidden_dim, output_dim)
        concat_mse = train_model(concat_model, train_loader, val_loader)
        
        # Attention model
        attn_model = AttentionModel(input_dim, hidden_dim, output_dim)
        attn_mse = train_model(attn_model, train_loader, val_loader)
        
        delta = (concat_mse - attn_mse) / concat_mse * 100
        
        print(f"  Concatenation MSE: {concat_mse:.6f}")
        print(f"  Attention MSE: {attn_mse:.6f}")
        print(f"  Delta: {delta:+.1f}% ({'Attention WIN' if delta > 0 else 'Concat WIN'})")
        
        results['concat_mses'].append(concat_mse)
        results['attn_mses'].append(attn_mse)
        results['delta'].append(delta)
    
    # Summary
    avg_delta = np.mean(results['delta'])
    attn_wins = sum(1 for d in results['delta'] if d > 0)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n| Timesteps | Concat MSE | Attn MSE | Delta | Winner |")
    print(f"|-----------|------------|---------|-------|--------|")
    for i, ts in enumerate(results['timesteps']):
        winner = "Attn" if results['delta'][i] > 0 else "Concat"
        print(f"| {ts:9d} | {results['concat_mses'][i]:10.6f} | {results['attn_mses'][i]:8.6f} | {results['delta'][i]:+6.1f}% | {winner:6s} |")
    
    print(f"\n| Average | {np.mean(results['concat_mses']):10.6f} | {np.mean(results['attn_mses']):8.6f} | {avg_delta:+6.1f}% | {attn_wins}/{len(results['timesteps'])} wins |")
    
    # Key insight
    if avg_delta > 10:
        insight = "Attention WINS with autocorrelation - temporal structure enables attention"
    elif avg_delta < -10:
        insight = "Concatenation WINS - attention overhead not justified even with autocorrelation"
    else:
        insight = "MIXED - Neither clearly wins across all sequence lengths"
    
    status = "SUPPORTED" if avg_delta > 10 else "REFUTED"
    
    print(f"\nInsight: {insight}")
    print(f"Status: {status}")
    
    results['avg_delta'] = avg_delta
    results['attn_wins'] = attn_wins
    results['insight'] = insight
    results['status'] = status
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = run_experiment()
