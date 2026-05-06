"""
H1.128: Attention with Regularization vs Standard
Test attention with different regularization strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegularizedAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(16, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.predictor = nn.Linear(hidden_dim, 16)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        z = self.dropout(self.input_proj(x))
        x_seq = z.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.predictor(attn_out.squeeze(1))


class StandardAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(16, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.predictor = nn.Linear(hidden_dim, 16)
        
    def forward(self, x):
        z = self.input_proj(x)
        x_seq = z.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.predictor(attn_out.squeeze(1))


def generate_data(num_samples, noise=0.1):
    state = torch.randn(num_samples, 16) * noise
    action = torch.randn(num_samples, 8) * noise
    
    next_state = state.clone()
    next_state[:, :3] += action[:, :3] * 0.1
    
    return state, next_state


def train(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        state, next_state = data
        pred = model(state)
        loss = criterion(pred, next_state)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def evaluate(model):
    data = generate_data(100)
    with torch.no_grad():
        state, next_state = data
        pred = model(state)
        mse = F.mse_loss(pred, next_state).item()
    return mse


def main():
    print("="*60)
    print("H1.128: Regularized vs Standard Attention")
    print("="*60 + "\n")
    
    results = {"regularized": [], "standard": []}
    
    for trial in range(5):
        print(f"--- Trial {trial+1} ---")
        
        model_r = RegularizedAttention()
        data = generate_data(300)
        train(model_r, data)
        results["regularized"].append(evaluate(model_r))
        
        model_s = StandardAttention()
        train(model_s, data)
        results["standard"].append(evaluate(model_s))
        
        print(f"  Regularized: {results['regularized'][-1]:.4f}, Standard: {results['standard'][-1]:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    reg_avg = np.mean(results["regularized"])
    stand_avg = np.mean(results["standard"])
    improvement = (stand_avg - reg_avg) / stand_avg * 100
    
    print(f"Regularized avg: {reg_avg:.4f}")
    print(f"Standard avg: {stand_avg:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    status = "✅ SUPPORTED" if improvement > 0 else "❌ REFUTED"
    print(f"\nStatus: {status}")
    
    return results


if __name__ == "__main__":
    main()