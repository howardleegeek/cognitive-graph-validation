#!/usr/bin/env python3
"""
H1.202: SSM/Attention on Manipulation Tasks with Goal States
Building on:
- H1.193: SSM (+97.6%) wins on manipulation tasks
- H3.90: SSM (-20.5%) loses on pure sequence prediction

Hypothesis: With manipulation-style task structure (goal states, 
action outcomes), SSM/Attention will outperform concatenation.
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


class ManipulationDataset(Dataset):
    """Dataset with manipulation-style task structure (goal states, action outcomes)."""
    
    def __init__(self, n_samples=200, timesteps=50, n_features=12, 
                 autocorrelation=0.85, has_goal=True, has_action=True):
        self.n_samples = n_samples
        self.timesteps = timesteps
        self.n_features = n_features
        self.autocorrelation = autocorrelation
        self.has_goal = has_goal
        self.has_action = has_action
        np.random.seed(42)
        
        self.data = []
        for _ in range(n_samples):
            seq, target = self._generate_manipulation_sequence(
                timesteps, n_features, autocorrelation, has_goal, has_action
            )
            self.data.append((seq, target))
    
    def _generate_manipulation_sequence(self, T, n_features, rho, has_goal, has_action):
        """Generate sequence with manipulation-style structure."""
        # Generate base state trajectory
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
        
        # Add goal state (manipulation target)
        if has_goal:
            goal = np.random.randn(n_features) * 0.5
            seq = np.vstack([seq, goal[np.newaxis, :]])  # Append goal as final state
            T += 1
        
        # Add action signal
        if has_action:
            actions = np.random.randn(T - 1, 4) * 0.1
            # Action influences next state
            for t in range(1, T):
                seq[t, :4] += actions[t-1] * 0.5
        
        # Target: predict goal state (for manipulation task)
        target = seq[-1, :].copy()
        
        return seq.astype(np.float32), target.astype(np.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.from_numpy(seq), torch.from_numpy(target)


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
    """State Space Model block (Mamba-style)."""
    
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
        for batch, target in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch, target in val_loader:
                output = model(batch)
                val_losses.append(criterion(output, target).item())
        
        val_mse = np.mean(val_losses)
        if val_mse < best_val:
            best_val = val_mse
    
    return best_val


def run_experiment():
    print("=" * 70)
    print("H1.202: SSM/Attention on Manipulation Tasks with Goal States")
    print("=" * 70)
    
    config = ExperimentConfig()
    
    results = {
        'experiment': 'H1.202',
        'timestamp': datetime.now().isoformat(),
        'parent': 'H1',
        'timesteps': [20, 30, 40, 50],
        'has_goal': True,
        'has_action': True,
        'concat_mses': [],
        'ssm_mses': [],
        'attn_mses': [],
        'ssm_delta': [],
        'attn_delta': []
    }
    
    for timesteps in results['timesteps']:
        print(f"\n--- Testing {timesteps}-step manipulation task (ρ={config.autocorrelation}) ---")
        print("     Task structure: goal states + action outcomes")
        
        full_dataset = ManipulationDataset(
            n_samples=config.n_samples,
            timesteps=timesteps,
            n_features=config.n_features,
            autocorrelation=config.autocorrelation,
            has_goal=True,
            has_action=True
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
        winner = "Concat" if ssm_delta < 0 and attn_delta < 0 else best
        print(f"  Best: {winner}")
        
        results['concat_mses'].append(concat_mse)
        results['ssm_mses'].append(ssm_mse)
        results['attn_mses'].append(attn_mse)
        results['ssm_delta'].append(ssm_delta)
        results['attn_delta'].append(attn_delta)
    
    # Summary
    avg_ssm_delta = np.mean(results['ssm_delta'])
    avg_attn_delta = np.mean(results['attn_delta'])
    ssm_wins = sum(1 for d in results['ssm_delta'] if d > 0)
    attn_wins = sum(1 for d in results['attn_delta'] if d > 0)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - Manipulation Task Structure")
    print("=" * 70)
    print(f"\n| Timesteps | Concat MSE | SSM MSE | SSM Δ | Attn MSE | Attn Δ | Winner |")
    print(f"|-----------|------------|---------|-------|----------|-------|--------|")
    for i, ts in enumerate(results['timesteps']):
        winner = "SSM" if results['ssm_delta'][i] > results['attn_delta'][i] and results['ssm_delta'][i] > 0 else \
                 "Attn" if results['attn_delta'][i] > 0 else "Concat"
        print(f"| {ts:9d} | {results['concat_mses'][i]:10.6f} | {results['ssm_mses'][i]:7.6f} | {results['ssm_delta'][i]:+5.1f}% | {results['attn_mses'][i]:8.6f} | {results['attn_delta'][i]:+5.1f}% | {winner:6s} |")
    
    print(f"\n| Average | {np.mean(results['concat_mses']):10.6f} | {np.mean(results['ssm_mses']):7.6f} | {avg_ssm_delta:+5.1f}% | {np.mean(results['attn_mses']):8.6f} | {avg_attn_delta:+5.1f}% | {ssm_wins}/4 SSM, {attn_wins}/4 Attn |")
    
    # Compare with pure sequence (H3.90)
    print("\n" + "=" * 70)
    print("COMPARISON: Manipulation vs Pure Sequence (H3.90)")
    print("=" * 70)
    print(f"SSM Delta: H1.202={avg_ssm_delta:+.1f}% vs H3.90=-20.5%")
    print(f"Attn Delta: H1.202={avg_attn_delta:+.1f}% vs H3.89=-30.5%")
    
    insight = "TASK STRUCTURE ENABLES SSM/ATTENTION" if avg_ssm_delta > 0 or avg_attn_delta > 0 else "TASK STRUCTURE NOT SUFFICIENT"
    
    status = "SUPPORTED" if avg_ssm_delta > 10 or avg_attn_delta > 10 else "REFUTED"
    
    print(f"\nInsight: {insight}")
    print(f"SSM wins: {ssm_wins}/4")
    print(f"Attn wins: {attn_wins}/4")
    print(f"Status: {status}")
    
    results['avg_ssm_delta'] = avg_ssm_delta
    results['avg_attn_delta'] = avg_attn_delta
    results['ssm_wins'] = ssm_wins
    results['attn_wins'] = attn_wins
    results['insight'] = insight
    results['status'] = status
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = run_experiment()