"""
H1.192: Attention + SSM Combined with Autocorrelation Injection
Tests whether combining autocorrelation injection with SSM can unlock 
attention on complex multi-step synthetic tasks.

Key insights:
- H1.181: Autocorrelation injection enables attention (+26.9% at ρ=0.95)
- H1.182b: SSM excels on next-step prediction with robot-like temporal structure
- H3.8: SSM outperforms attention on 20+ step sequences
- H3.9: Mamba-style gated mechanism outperforms standard attention

Hypothesis: Combined SSM + Attention with autocorrelation injection 
will outperform both individual methods on complex multi-step tasks.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Literal

@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    n_samples: int = 200
    timesteps: int = 30
    n_features: int = 12
    hidden_dim: int = 256
    ssm_state_dim: int = 16
    learning_rate: float = 0.001
    epochs: int = 100
    autocorrelation: float = 0.85  # High autocorrelation like real robot data

class RobotLikeSequenceDataset(Dataset):
    """Dataset with robot-like temporal structure (high autocorrelation)."""
    
    def __init__(self, n_samples=200, timesteps=30, n_features=12, autocorrelation=0.85):
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
            # Base smooth signal
            t = np.linspace(0, 1, T)
            freq = np.random.uniform(1.0, 3.0)
            phase = np.random.uniform(0, np.pi)
            base = np.sin(2 * np.pi * freq * t + phase)
            
            # Add autocorrelation by smoothing
            for t_idx in range(T):
                if t_idx == 0:
                    seq[t_idx, i*2] = base[t_idx]
                else:
                    # Autocorrelation: blend current with previous
                    seq[t_idx, i*2] = rho * seq[t_idx-1, i*2] + (1 - rho) * base[t_idx]
            
            # Second feature is derivative (velocity)
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
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(input_dim, state_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(state_dim, output_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, seq, input_dim)
        batch, seq_len, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch, self.state_dim, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            # SSM transition
            u = x[:, t]
            h = torch.tanh(h @ self.A.T + u @ self.B)
            y = h @ self.C + u @ self.D
            
            # Apply gate
            gate = self.gate(u)
            y = y * gate
            
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class AttentionSSMModel(nn.Module):
    """Combined Attention + SSM model with autocorrelation injection."""
    
    def __init__(self, input_dim=12, hidden_dim=256, ssm_state_dim=16):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # SSM branch
        self.ssm = SSMBlock(hidden_dim, ssm_state_dim, hidden_dim)
        
        # Attention branch
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output
        self.output = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x: (batch, seq, input_dim)
        batch, seq_len, _ = x.shape
        
        # Project input
        h = self.input_proj(x)
        
        # SSM branch
        ssm_out = self.ssm(h)
        
        # Attention branch with causal mask
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attention(h, h, h, attn_mask=attn_mask)
        attn_out = self.attn_norm(attn_out)
        
        # Fusion
        combined = torch.cat([ssm_out, attn_out], dim=-1)
        fused = self.fusion(combined)
        
        # Output (predict next step)
        output = self.output(fused)
        
        # Return prediction for last timestep
        return output[:, -1]


class BaselineModel(nn.Module):
    """Baseline concatenation model."""
    
    def __init__(self, input_dim=12, hidden_dim=256, timesteps=30):
        super().__init__()
        self.timesteps = timesteps
        self.net = nn.Sequential(
            nn.Linear(input_dim * timesteps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        batch = x.shape[0]
        # Flatten sequence
        x_flat = x.reshape(batch, -1)
        return self.net(x_flat)


def train_model(model, train_loader, epochs=100, lr=0.001):
    """Train a single model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Input is all but last, target is last
            x = batch[:, :-1]
            y = batch[:, -1]
            
            pred = model(x)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / epochs


def evaluate_model(model, test_loader):
    """Evaluate model on test data."""
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch[:, :-1]
            y = batch[:, -1]
            
            pred = model(x)
            loss = criterion(pred, y)
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def run_experiment():
    """Run the full experiment."""
    print("=" * 60)
    print("H1.192: Attention + SSM Combined with Autocorrelation")
    print("=" * 60)
    
    # Configuration
    config = ExperimentConfig(
        n_samples=200,
        timesteps=30,
        n_features=12,
        hidden_dim=256,
        ssm_state_dim=16,
        learning_rate=0.001,
        epochs=100,
        autocorrelation=0.85
    )
    
    print(f"\nConfiguration:")
    print(f"  Samples: {config.n_samples}")
    print(f"  Timesteps: {config.timesteps}")
    print(f"  Autocorrelation: {config.autocorrelation}")
    
    # Create dataset
    dataset = RobotLikeSequenceDataset(
        n_samples=config.n_samples,
        timesteps=config.timesteps,
        n_features=config.n_features,
        autocorrelation=config.autocorrelation
    )
    
    # Split train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = {}
    
    # Test 1: Baseline (concatenation)
    print("\n--- Test 1: Baseline (Concatenation) ---")
    baseline = BaselineModel(input_dim=config.n_features, hidden_dim=config.hidden_dim, timesteps=config.timesteps-1)
    train_model(baseline, train_loader, epochs=config.epochs, lr=config.learning_rate)
    baseline_loss = evaluate_model(baseline, test_loader)
    results['baseline'] = baseline_loss
    print(f"  Baseline MSE: {baseline_loss:.6f}")
    
    # Test 2: SSM only
    print("\n--- Test 2: SSM Only ---")
    ssm_model = AttentionSSMModel(input_dim=config.n_features, hidden_dim=config.hidden_dim, ssm_state_dim=config.ssm_state_dim)
    # Disable attention branch by zeroing
    for param in ssm_model.attention.parameters():
        param.data.zero_()
    train_model(ssm_model, train_loader, epochs=config.epochs, lr=config.learning_rate)
    ssm_loss = evaluate_model(ssm_model, test_loader)
    results['ssm'] = ssm_loss
    print(f"  SSM MSE: {ssm_loss:.6f}")
    
    # Test 3: Attention only
    print("\n--- Test 3: Attention Only ---")
    attn_model = AttentionSSMModel(input_dim=config.n_features, hidden_dim=config.hidden_dim, ssm_state_dim=config.ssm_state_dim)
    # Disable SSM branch
    for param in ssm_model.ssm.parameters():
        param.data.zero_()
    train_model(attn_model, train_loader, epochs=config.epochs, lr=config.learning_rate)
    attn_loss = evaluate_model(attn_model, test_loader)
    results['attention'] = attn_loss
    print(f"  Attention MSE: {attn_loss:.6f}")
    
    # Test 4: Combined (Attention + SSM)
    print("\n--- Test 4: Combined (Attention + SSM) ---")
    combined_model = AttentionSSMModel(input_dim=config.n_features, hidden_dim=config.hidden_dim, ssm_state_dim=config.ssm_state_dim)
    train_model(combined_model, train_loader, epochs=config.epochs, lr=config.learning_rate)
    combined_loss = evaluate_model(combined_model, test_loader)
    results['combined'] = combined_loss
    print(f"  Combined MSE: {combined_loss:.6f}")
    
    # Calculate improvements
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    baseline_mse = results['baseline']
    print(f"\nBaseline (Concat) MSE: {baseline_mse:.6f}")
    
    for method, mse in results.items():
        if method != 'baseline':
            delta = (baseline_mse - mse) / baseline_mse * 100
            print(f"{method.capitalize()} MSE: {mse:.6f} ({delta:+.1f}%)")
    
    # Determine best method
    best_method = min(results, key=results.get)
    best_mse = results[best_method]
    improvement = (baseline_mse - best_mse) / baseline_mse * 100
    
    print(f"\nBest: {best_method} with {improvement:+.1f}% improvement")
    
    # Determine status
    if improvement > 10:
        status = "SUPPORTED"
    elif improvement > 0:
        status = "MARGINAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    return {
        'results': results,
        'improvement': improvement,
        'status': status
    }


if __name__ == "__main__":
    results = run_experiment()