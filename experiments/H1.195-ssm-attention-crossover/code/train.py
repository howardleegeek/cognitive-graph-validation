"""
H1.195: SSM vs Attention Crossover Point Discovery
Tests multiple sequence lengths to find where SSM starts outperforming.

Key insights from H1.193/H1.194:
- H1.193: SSM (+97.6%) wins on 50-step with ρ=0.85
- H1.194: Baseline wins on 80-step (SSM -14.1%, Attn -24.2%)

Hypothesis: Find the crossover point where SSM starts to outperform.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    n_samples: int = 200
    timesteps_list: list = None
    n_features: int = 12
    hidden_dim: int = 256
    ssm_state_dim: int = 16
    learning_rate: float = 0.001
    epochs: int = 100
    autocorrelation: float = 0.85


class RobotLikeSequenceDataset(Dataset):
    """Dataset with robot-like temporal structure (high autocorrelation)."""
    
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
        """Generate sequence with specified autocorrelation (robot-like)."""
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


class BaselineConcatModel(nn.Module):
    """Baseline concatenation model."""
    
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
    """Attention-based sequence model."""
    
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
    """Train model and return validation MSE."""
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


def run_experiment(timesteps, config):
    """Run experiment for a specific timestep."""
    print(f"\n--- Testing {timesteps} timesteps ---")
    
    full_dataset = RobotLikeSequenceDataset(
        n_samples=config.n_samples,
        timesteps=timesteps,
        n_features=config.n_features,
        autocorrelation=config.autocorrelation
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    input_dim = config.n_features
    output_dim = config.n_features
    hidden_dim = config.hidden_dim
    ssm_state_dim = config.ssm_state_dim
    
    # Baseline
    baseline = BaselineConcatModel(input_dim, hidden_dim, output_dim)
    baseline_mse = train_model(baseline, train_loader, val_loader)
    
    # SSM
    ssm_model = SSMModel(input_dim, hidden_dim, ssm_state_dim, output_dim)
    ssm_mse = train_model(ssm_model, train_loader, val_loader)
    
    # Attention
    attn_model = AttentionModel(input_dim, hidden_dim, output_dim)
    attn_mse = train_model(attn_model, train_loader, val_loader)
    
    ssm_imp = (baseline_mse - ssm_mse) / baseline_mse * 100
    attn_imp = (baseline_mse - attn_mse) / baseline_mse * 100
    
    print(f"  Baseline: {baseline_mse:.6f}")
    print(f"  SSM: {ssm_mse:.6f} ({ssm_imp:+.1f}%)")
    print(f"  Attention: {attn_mse:.6f} ({attn_imp:+.1f}%)")
    
    return {
        'timesteps': timesteps,
        'baseline': baseline_mse,
        'ssm': ssm_mse,
        'ssm_imp': ssm_imp,
        'attention': attn_mse,
        'attn_imp': attn_imp
    }


def main():
    print("=" * 60)
    print("H1.195: SSM vs Attention Crossover Point Discovery")
    print("=" * 60)
    
    config = ExperimentConfig()
    timesteps_list = [20, 30, 40, 50, 60, 70, 80]
    
    print(f"\nConfiguration:")
    print(f"  Samples: {config.n_samples}")
    print(f"  Timesteps: {timesteps_list}")
    print(f"  Autocorrelation: {config.autocorrelation}")
    
    results = []
    for ts in timesteps_list:
        result = run_experiment(ts, config)
        results.append(result)
    
    print(f"\n{'=' * 60}")
    print("CROSSOVER POINT ANALYSIS")
    print(f"{'=' * 60}")
    
    print(f"\n| Timesteps | Baseline | SSM | SSM Δ | Attention | Attn Δ | Winner |")
    print(f"|-----------|---------|-----|-------|-----------|--------|--------|")
    for r in results:
        winner = "SSM" if r['ssm_imp'] > 0 and r['ssm_imp'] > r['attn_imp'] else "Baseline"
        print(f"| {r['timesteps']:3d} | {r['baseline']:.6f} | {r['ssm']:.6f} | {r['ssm_imp']:+5.1f}% | {r['attention']:.6f} | {r['attn_imp']:+5.1f}% | {winner} |")
    
    # Find crossover
    for i in range(len(results) - 1):
        if results[i]['ssm_imp'] > 0 and results[i+1]['ssm_imp'] < 0:
            print(f"\nCrossover point: SSM wins at {results[i]['timesteps']} steps, loses at {results[i+1]['timesteps']} steps")
            break
    else:
        print(f"\nNo clear crossover found in tested range")


if __name__ == "__main__":
    main()