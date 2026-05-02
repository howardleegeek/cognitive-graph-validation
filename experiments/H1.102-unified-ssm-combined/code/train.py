"""
H1.102: Unified + SSM Combined Architecture
Tests combining unified cognitive graph with SSM for temporal reasoning
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class UnifiedEncoder(nn.Module):
    """Unified encoder - early fusion of physical and semantic"""
    def __init__(self, input_dim=64, physical_dim=112, semantic_dim=400, total_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.physical_encoder = nn.Linear(input_dim, physical_dim)
        self.semantic_encoder = nn.Linear(input_dim, semantic_dim)
        self.fusion = nn.Linear(total_dim, total_dim)
        
    def forward(self, x):
        # Handle both [batch, seq, dim] and [batch*seq, dim] shapes
        if len(x.shape) == 3:
            batch_size, seq_len, dim = x.shape
            # Reshape to process each timestep
            x_flat = x.view(-1, dim)
            physical_enc = self.physical_encoder(x_flat)
            semantic_enc = self.semantic_encoder(x_flat)
            combined = torch.cat([physical_enc, semantic_enc], dim=-1)
            encoded = self.fusion(combined)
            return encoded.view(batch_size, seq_len, -1)
        else:
            physical_enc = self.physical_encoder(x)
            semantic_enc = self.semantic_encoder(x)
            combined = torch.cat([physical_enc, semantic_enc], dim=-1)
            return self.fusion(combined)


class SSMProcessor(nn.Module):
    """Mamba-style SSM for temporal processing"""
    def __init__(self, dim=512, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.input_proj = nn.Linear(dim, state_dim)
        self.ssm_A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.ssm_B = nn.Linear(state_dim, state_dim)
        self.ssm_C = nn.Linear(state_dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        h = torch.zeros(batch_size, self.state_dim, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            inp = self.input_proj(x[:, t])
            h = torch.matmul(h, self.ssm_A.t()) + self.ssm_B(inp)
            gate = self.gate(h)
            out = self.ssm_C(h * gate)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)


class UnifiedSSMModel(nn.Module):
    """Combined Unified + SSM architecture"""
    def __init__(self, input_dim=64, hidden_dim=512, state_dim=16):
        super().__init__()
        self.unified = UnifiedEncoder(input_dim, 112, 400, 512)
        self.ssm = SSMProcessor(512, state_dim)
        self.output = nn.Linear(512, input_dim)
        
    def forward(self, x):
        # Unified encoding
        encoded = self.unified(x)
        # SSM temporal processing
        ssm_out = self.ssm(encoded)
        # Output from last timestep
        return self.output(ssm_out[:, -1, :])


class BaselineModel(nn.Module):
    """Baseline: separated JEPA-style architecture"""
    def __init__(self, input_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        self.output = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        h, _ = self.lstm(self.encoder(x))
        return self.output(h[:, -1, :])


def generate_multistep_data(n_samples, seq_len, input_dim, n_steps=5):
    """Generate multi-step task data"""
    np.random.seed(42)
    X, Y = [], []
    
    for _ in range(n_samples):
        # Initial state
        state = torch.randn(seq_len, input_dim) * 0.1
        
        # Multi-step transitions
        for step in range(n_steps):
            # Random action
            action = torch.randn(seq_len, 4) * 0.1
            
            # State transition with dynamics
            transition = state * 0.5 + action[:, :1] * 0.3 + torch.randn(seq_len, input_dim) * 0.05
            state = state + transition
        
        # Target is final state
        y = state[-1] + torch.randn(input_dim) * 0.1
        X.append(state)
        Y.append(y)
    
    return torch.stack(X), torch.stack(Y)


def main():
    print("=" * 60)
    print("H1.102: Unified + SSM Combined Architecture")
    print("=" * 60)
    
    results = {}
    
    # Test configurations
    configs = [
        (5, "Baseline"),
        (5, "Unified+SSM"),
        (10, "Baseline"),
        (10, "Unified+SSM"),
        (15, "Baseline"),
        (15, "Unified+SSM"),
    ]
    
    input_dim = 64
    n_samples = 200
    
    for n_steps, arch in configs:
        config_name = f"{n_steps}step_{arch}"
        print(f"\n--- {config_name} ---")
        
        # Generate data
        X, Y = generate_multistep_data(n_samples, 10, input_dim, n_steps)
        X, Y = X.to(device), Y.to(device)
        
        # Split train/test
        split = int(0.8 * n_samples)
        X_train, Y_train = X[:split], Y[:split]
        X_test, Y_test = X[split:], Y[split:]
        
        torch.manual_seed(42)
        
        if arch == "Baseline":
            model = BaselineModel(input_dim, 256).to(device)
        else:
            model = UnifiedSSMModel(input_dim, 512, 16).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Train
        for epoch in range(150):
            model.train()
            pred = model(X_train)
            loss = criterion(pred, Y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            mse = criterion(test_pred, Y_test).item()
        
        results[config_name] = {"mse": mse, "n_steps": n_steps, "arch": arch}
        print(f"  MSE: {mse:.6f}")
    
    # Compare architectures at each step count
    print("\n=== Comparison ===")
    for n_steps in [5, 10, 15]:
        baseline_mse = results[f"{n_steps}step_Baseline"]["mse"]
        unified_mse = results[f"{n_steps}step_Unified+SSM"]["mse"]
        improvement = (baseline_mse - unified_mse) / baseline_mse * 100
        print(f"{n_steps} steps: Baseline={baseline_mse:.6f}, Unified+SSM={unified_mse:.6f}, Δ={improvement:+.1f}%")
    
    # Save results
    output_dir = "experiments/H1.102-unified-ssm-combined"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/results.json")
    
    return results


if __name__ == "__main__":
    results = main()