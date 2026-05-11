#!/usr/bin/env python3
"""
H1.201: Ultra-Complex Multi-Step with Real Robot Temporal Dynamics
Building on:
- H1: +25.6% improvement with real robot data
- H1.193: SSM (+97.6%) on 50-step with autocorrelation
- H1.182: +82.2% on temporal reasoning with autocorrelation

Hypothesis: Cognitive Graph advantage GROWS with complexity when data has
real robot-like temporal structure (autocorrelation).
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
    """Dataset with real robot-like temporal structure (high autocorrelation)."""
    
    def __init__(self, n_samples=200, timesteps=100, n_features=12, autocorrelation=0.85):
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


class BaselineModel(nn.Module):
    """Baseline MLP model with mean pooling."""
    
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


class CognitiveGraphModel(nn.Module):
    """Cognitive Graph model with physical/semantic split."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, physical_dim=112):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = hidden_dim - physical_dim
        
        self.physical_encoder = nn.Sequential(
            nn.Linear(input_dim, physical_dim),
            nn.ReLU()
        )
        
        self.semantic_encoder = nn.Sequential(
            nn.Linear(input_dim, self.semantic_dim),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        fused = torch.cat([physical, semantic], dim=-1)
        fused = self.fusion(fused)
        pooled = fused.mean(dim=1)
        return self.decoder(pooled)


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
    print("H1.201: Ultra-Complex Multi-Step with Real Robot Temporal Dynamics")
    print("=" * 70)
    
    config = ExperimentConfig()
    
    results = {
        'experiment': 'H1.201',
        'timestamp': datetime.now().isoformat(),
        'parent': 'H1',
        'timesteps': [20, 50, 75, 100, 125],
        'baseline_mses': [],
        'cg_mses': [],
        'improvements': []
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
        
        # Baseline
        baseline = BaselineModel(input_dim, hidden_dim, output_dim)
        baseline_mse = train_model(baseline, train_loader, val_loader)
        
        # Cognitive Graph
        cg_model = CognitiveGraphModel(input_dim, hidden_dim, output_dim)
        cg_mse = train_model(cg_model, train_loader, val_loader)
        
        improvement = (baseline_mse - cg_mse) / baseline_mse * 100
        
        print(f"  Baseline MSE: {baseline_mse:.6f}")
        print(f"  Cognitive Graph MSE: {cg_mse:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results['baseline_mses'].append(baseline_mse)
        results['cg_mses'].append(cg_mse)
        results['improvements'].append(improvement)
    
    # Summary
    avg_improvement = np.mean(results['improvements'])
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n| Timesteps | Baseline MSE | CG MSE | Improvement |")
    print(f"|-----------|--------------|--------|-------------|")
    for i, ts in enumerate(results['timesteps']):
        print(f"| {ts:9d} | {results['baseline_mses'][i]:12.6f} | {results['cg_mses'][i]:6.6f} | {results['improvements'][i]:+10.1f}% |")
    
    print(f"\n| Average | {np.mean(results['baseline_mses']):12.6f} | {np.mean(results['cg_mses']):6.6f} | {avg_improvement:+10.1f}% |")
    
    # Analyze trend
    if results['improvements'][-1] > results['improvements'][0]:
        trend = "GROWING - CG advantage increases with complexity"
    else:
        trend = "DECLINING - CG advantage decreases with complexity"
    
    status = "SUPPORTED" if avg_improvement > 15 else "REFUTED"
    
    print(f"\nTrend: {trend}")
    print(f"Status: {status}")
    
    results['avg_improvement'] = avg_improvement
    results['trend'] = trend
    results['status'] = status
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = run_experiment()
