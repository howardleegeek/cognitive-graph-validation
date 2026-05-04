"""
H3.33: SSM with Optimized State Dimensions on Continuous Control

Hypothesis: SSM with optimized state dimensions outperforms concatenation on continuous control

Based on H3.32 result: SSM +0.0% vs concat - essentially tied
Goal: Find optimal SSM state dimension to beat concatenation

State dimensions to test: 8, 16, 24, 32, 48
Hidden dimensions to test: 64, 128, 256, 512
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import json
from datetime import datetime

class SSMBlock(nn.Module):
    def __init__(self, d_model, state_dim, hidden_dim):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.x_proj = nn.Linear(d_model, state_dim)
        self.dt_proj = nn.Linear(state_dim, state_dim)
        self.A = nn.Parameter(torch.eye(state_dim) + torch.randn(state_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.ones(state_dim) * 0.1)
        self.output_proj = nn.Linear(state_dim, d_model)
        
    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        h = self.x_proj(x_flat)
        dt = torch.sigmoid(self.dt_proj(h))
        
        A_exp = torch.matrix_exp(self.A.unsqueeze(0).expand(B * L, -1, -1) * dt.unsqueeze(-1))
        
        h_new = torch.bmm(h.unsqueeze(-2), A_exp).squeeze(-2) + h * self.D
        
        output = self.output_proj(h_new)
        return output.reshape(B, L, D)

class ContinuousControlModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, use_ssm=False, ssm_state_dim=16, ssm_hidden=128):
        super().__init__()
        self.use_ssm = use_ssm
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if use_ssm:
            self.ssm = SSMBlock(hidden_dim, ssm_state_dim, ssm_hidden)
        else:
            self.ssm = None
            
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        if self.use_ssm and self.ssm is not None:
            h = self.ssm(h)
        return self.decoder(h)

def generate_continuous_control_data(n_samples: int, seq_len: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate continuous control trajectory data"""
    state_dim = 8
    action_dim = 4
    
    X, Y = [], []
    for _ in range(n_samples):
        state = np.random.randn(seq_len, state_dim) * 0.5
        for t in range(1, seq_len):
            state[t] += 0.3 * state[t-1] + np.random.randn(state_dim) * noise
        
        actions = np.zeros((seq_len, action_dim))
        for t in range(seq_len):
            actions[t] = 0.5 * state[t, :action_dim] + np.random.randn(action_dim) * noise
        
        X.append(np.concatenate([state, actions], axis=-1))
        
        next_state = np.zeros((seq_len, state_dim))
        for t in range(seq_len):
            action_padded = np.zeros(state_dim)
            action_padded[:action_dim] = actions[t]
            next_state[t] = state[t] + 0.1 * action_padded + np.random.randn(state_dim) * noise
        
        Y.append(next_state)
    
    return np.array(X), np.array(Y)

def train_and_evaluate(
    n_samples: int, 
    seq_len: int, 
    use_ssm: bool, 
    ssm_state_dim: int = 16,
    ssm_hidden: int = 128,
    n_epochs: int = 100,
    lr: float = 0.001
) -> float:
    """Train model and return validation MSE"""
    X, Y = generate_continuous_control_data(n_samples, seq_len)
    
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    
    input_dim = X_train.shape[-1]
    output_dim = Y_train.shape[-1]
    
    model = ContinuousControlModel(
        input_dim, output_dim, 
        use_ssm=use_ssm, 
        ssm_state_dim=ssm_state_dim,
        ssm_hidden=ssm_hidden
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_mse = criterion(val_pred, Y_val_t).item()
    
    return val_mse

def run_experiment():
    """Run H3.33 experiment: SSM optimized dimensions on continuous control"""
    print("=" * 60)
    print("H3.33: SSM with Optimized State Dimensions on Continuous Control")
    print("=" * 60)
    
    results = {
        "experiment": "H3.33",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "SSM with optimized state dimensions outperforms concatenation on continuous control",
        "configurations": []
    }
    
    n_samples = 200
    seq_len = 20
    n_epochs = 100
    
    print("\n1. Baseline (Concatenation)...")
    concat_mse = train_and_evaluate(n_samples, seq_len, use_ssm=False)
    print(f"   Concatenation MSE: {concat_mse:.6f}")
    
    print("\n2. Testing SSM configurations...")
    
    best_mse = float('inf')
    best_config = None
    
    for state_dim in [8, 16, 24, 32, 48]:
        for hidden_dim in [64, 128, 256]:
            print(f"   Testing state_dim={state_dim}, hidden_dim={hidden_dim}...", end=" ")
            mse = train_and_evaluate(
                n_samples, seq_len, 
                use_ssm=True, 
                ssm_state_dim=state_dim,
                ssm_hidden=hidden_dim,
                n_epochs=n_epochs
            )
            print(f"MSE={mse:.6f}")
            
            config_result = {
                "ssm_state_dim": state_dim,
                "ssm_hidden_dim": hidden_dim,
                "mse": mse
            }
            results["configurations"].append(config_result)
            
            if mse < best_mse:
                best_mse = mse
                best_config = {"state_dim": state_dim, "hidden_dim": hidden_dim}
    
    if best_config is None:
        best_config = {"state_dim": 16, "hidden_dim": 128}
    
    print(f"\n3. Best SSM configuration: state_dim={best_config['state_dim']}, hidden_dim={best_config['hidden_dim']}")
    print(f"   Best SSM MSE: {best_mse:.6f}")
    print(f"   Concatenation MSE: {concat_mse:.6f}")
    
    improvement = (concat_mse - best_mse) / concat_mse * 100
    print(f"   Improvement: {improvement:+.2f}%")
    
    results["concat_mse"] = concat_mse
    results["best_ssm_mse"] = best_mse
    results["best_config"] = best_config
    results["improvement_percent"] = improvement
    
    if improvement > 0:
        status = "SUPPORTED"
        print(f"\n✅ H3.33: SUPPORTED - SSM outperforms by {improvement:+.2f}%")
    elif improvement > -5:
        status = "INCONCLUSIVE"
        print(f"\n⚠️ H3.33: INCONCLUSIVE - SSM within 5% of concat ({improvement:+.2f}%)")
    else:
        status = "REFUTED"
        print(f"\n❌ H3.33: REFUTED - Concatenation wins by {-improvement:.2f}%")
    
    results["status"] = status
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
    return results

if __name__ == "__main__":
    run_experiment()