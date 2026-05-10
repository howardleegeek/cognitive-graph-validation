"""
H1.194: SSM on Ultra-Long Sequences (60-100 steps)
Tests whether SSM maintains advantage on even longer sequences.

Key insights from H1.193:
- SSM (+97.6%) dramatically outperforms attention (-23.6%) on 50-step sequences
- SSM's sequential state modeling is well-suited for robot-like temporal structure

Hypothesis: SSM will maintain advantage on 60-100 step sequences.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    n_samples: int = 200
    timesteps: int = 80
    n_features: int = 12
    hidden_dim: int = 256
    ssm_state_dim: int = 16
    learning_rate: float = 0.001
    epochs: int = 100
    autocorrelation: float = 0.85


class RobotLikeSequenceDataset(Dataset):
    """Dataset with robot-like temporal structure (high autocorrelation)."""
    
    def __init__(self, n_samples=200, timesteps=80, n_features=12, autocorrelation=0.85):
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
        batch, seq_len, feat = x.shape
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


def main():
    print("=" * 60)
    print("H1.194: SSM on Ultra-Long Sequences (60-100 steps)")
    print("=" * 60)
    
    config = ExperimentConfig()
    
    print(f"\nConfiguration:")
    print(f"  Samples: {config.n_samples}")
    print(f"  Timesteps: {config.timesteps}")
    print(f"  Autocorrelation: {config.autocorrelation}")
    
    # Create datasets
    full_dataset = RobotLikeSequenceDataset(
        n_samples=config.n_samples,
        timesteps=config.timesteps,
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
    
    # Test 1: Baseline (Concatenation)
    print(f"\n--- Test 1: Baseline (Concatenation) ---")
    baseline = BaselineConcatModel(input_dim, hidden_dim, output_dim)
    baseline_mse = train_model(baseline, train_loader, val_loader)
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    
    # Test 2: SSM
    print(f"\n--- Test 2: SSM ---")
    ssm_model = SSMModel(input_dim, hidden_dim, ssm_state_dim, output_dim)
    ssm_mse = train_model(ssm_model, train_loader, val_loader)
    print(f"  SSM MSE: {ssm_mse:.6f}")
    
    # Test 3: Attention
    print(f"\n--- Test 3: Attention ---")
    attn_model = AttentionModel(input_dim, hidden_dim, output_dim)
    attn_mse = train_model(attn_model, train_loader, val_loader)
    print(f"  Attention MSE: {attn_mse:.6f}")
    
    # Results
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    
    baseline_imp = 0
    ssm_imp = (baseline_mse - ssm_mse) / baseline_mse * 100
    attn_imp = (baseline_mse - attn_mse) / baseline_mse * 100
    
    print(f"Baseline (Concat) MSE: {baseline_mse:.6f}")
    print(f"SSM MSE: {ssm_mse:.6f} ({ssm_imp:+.1f}%)")
    print(f"Attention MSE: {attn_mse:.6f} ({attn_imp:+.1f}%)")
    
    if ssm_mse < baseline_mse and ssm_mse < attn_mse:
        print(f"\nBest: SSM with {ssm_imp:+.1f}% improvement")
        print("Status: SUPPORTED")
    elif attn_mse < baseline_mse:
        print(f"\nBest: Attention with {attn_imp:+.1f}% improvement")
        print("Status: PARTIAL - SSM doesn't maintain advantage")
    else:
        print(f"\nBest: Baseline")
        print("Status: REFUTED")


if __name__ == "__main__":
    main()