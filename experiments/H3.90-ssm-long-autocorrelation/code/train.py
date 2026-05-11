#!/usr/bin/env python3
"""
H3.90: SSM on Long Sequences with Real Robot Autocorrelation
Building on:
- H1.193: SSM (+97.6%) on 50-step with autocorrelation
- H1.195: SSM vs Attention crossover - SSM wins on next-step prediction
- H3.89: Attention (-30.5%) - concat wins over attention

Hypothesis: SSM will outperform concatenation across all tested
sequence lengths due to its sequential state modeling.
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
    ssm_state_dim: int = 16
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


class SSMBlock(nn.Module):
    """State Space Model block (simplified Mamba-style)."""
    
    def __init__(self, input_dim, state_dim=16, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        output_dim = output_dim or input_dim
        
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(input_dim, state_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(state_dim, output_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.state_dim, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            u = x[:, t]
            h = torch.tanh(h @ self.A.T + u @ self.B)
            y = h @ self.C + u @ self.D
            gate = self.gate(u)
            y = y * gate
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class SSMModel(nn.Module):
    """SSM-based sequence model."""
    
    def __init__(self, input_dim, hidden_dim, ssm_state_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ssm = SSMBlock(hidden_dim, ssm_state_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        ssm_out = self.ssm(encoded)
        return self.decoder(ssm_out.mean(dim=1))


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
    print("H3.90: SSM on Long Sequences with Real Robot Autocorrelation")
    print("=" * 70)
    
    config = ExperimentConfig()
    
    results = {
        'experiment': 'H3.90',
        'timestamp': datetime.now().isoformat(),
        'parent': 'H3',
        'timesteps': [20, 30, 40, 50, 60, 70, 80],
        'concat_mses': [],
        'ssm_mses': [],
        'attn_mses': [],
        'ssm_delta': [],
        'attn_delta': []
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
        ssm_state_dim = config.ssm_state_dim
        
        # Concatenation baseline
        concat_model = ConcatModel(input_dim, hidden_dim, output_dim)
        concat_mse = train_model(concat_model, train_loader, val_loader)
        
        # SSM model
        ssm_model = SSMModel(input_dim, hidden_dim, ssm_state_dim, output_dim)
        ssm_mse = train_model(ssm_model, train_loader, val_loader)
        
        # Attention model
        attn_model = AttentionModel(input_dim, hidden_dim, output_dim)
        attn_mse = train_model(attn_model, train_loader, val_loader)
        
        ssm_delta = (concat_mse - ssm_mse) / concat_mse * 100
        attn_delta = (concat_mse - attn_mse) / concat_mse * 100
        
        print(f"  Concatenation MSE: {concat_mse:.6f}")
        print(f"  SSM MSE: {ssm_mse:.6f} ({ssm_delta:+.1f}%)")
        print(f"  Attention MSE: {attn_mse:.6f} ({attn_delta:+.1f}%)")
        
        best = "SSM" if ssm_delta > attn_delta else "Attn"
        print(f"  Best: {best}")
        
        results['concat_mses'].append(concat_mse)
        results['ssm_mses'].append(ssm_mse)
        results['attn_mses'].append(attn_mse)
        results['ssm_delta'].append(ssm_delta)
        results['attn_delta'].append(attn_delta)
    
    # Summary
    avg_ssm_delta = np.mean(results['ssm_delta'])
    avg_attn_delta = np.mean(results['attn_delta'])
    ssm_wins = sum(1 for d in results['ssm_delta'] if d > 0)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n| Timesteps | Concat MSE | SSM MSE | SSM Δ | Attn MSE | Attn Δ |")
    print(f"|-----------|------------|---------|-------|----------|-------|")
    for i, ts in enumerate(results['timesteps']):
        print(f"| {ts:9d} | {results['concat_mses'][i]:10.6f} | {results['ssm_mses'][i]:7.6f} | {results['ssm_delta'][i]:+5.1f}% | {results['attn_mses'][i]:8.6f} | {results['attn_delta'][i]:+5.1f}% |")
    
    print(f"\n| Average | {np.mean(results['concat_mses']):10.6f} | {np.mean(results['ssm_mses']):7.6f} | {avg_ssm_delta:+5.1f}% | {np.mean(results['attn_mses']):8.6f} | {avg_attn_delta:+5.1f}% |")
    
    # Key insight
    if avg_ssm_delta > 10:
        insight = "SSM WINS with autocorrelation - sequential state modeling enables better prediction"
    elif avg_ssm_delta > 0:
        insight = "SSM WINS marginally - slight advantage over concatenation"
    else:
        insight = "Concatenation WINS - SSM not beneficial in this synthetic setting"
    
    status = "SUPPORTED" if avg_ssm_delta > 10 else "REFUTED"
    
    print(f"\nInsight: {insight}")
    print(f"SSM wins: {ssm_wins}/{len(results['timesteps'])}")
    print(f"Status: {status}")
    
    results['avg_ssm_delta'] = avg_ssm_delta
    results['avg_attn_delta'] = avg_attn_delta
    results['ssm_wins'] = ssm_wins
    results['insight'] = insight
    results['status'] = status
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = run_experiment()