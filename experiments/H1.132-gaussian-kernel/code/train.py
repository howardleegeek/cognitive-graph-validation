"""
H1.132: Gaussian Kernel Attention
Test attention with Gaussian kernel (RBF-like)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianKernelAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_centers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_centers = num_centers
        
        self.input_proj = nn.Linear(16, hidden_dim)
        self.centers = nn.Parameter(torch.randn(num_centers, hidden_dim) * 0.01)
        self.scale = nn.Parameter(torch.tensor([1.0]))
        self.predictor = nn.Linear(hidden_dim, 16)
        
    def forward(self, x):
        z = self.input_proj(x)
        
        z_exp = z.unsqueeze(1)
        centers_exp = self.centers.unsqueeze(0).expand(z.shape[0], -1, -1)
        
        dist_sq = ((z_exp - centers_exp) ** 2).sum(dim=-1)
        
        kernel = torch.exp(-dist_sq / (2 * self.scale ** 2 + 1e-8))
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
        
        attended = (centers_exp * kernel.unsqueeze(-1)).sum(dim=1)
        
        return self.predictor(attended)


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
    final_loss = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        state, next_state = data
        pred = model(state)
        loss = criterion(pred, next_state)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    
    return final_loss


def evaluate(model):
    data = generate_data(100)
    with torch.no_grad():
        state, next_state = data
        pred = model(state)
        mse = F.mse_loss(pred, next_state).item()
    return mse


def main():
    print("="*60)
    print("H1.132: Gaussian Kernel Attention")
    print("="*60 + "\n")
    
    results = {"gaussian": [], "standard": []}
    seq_lengths = [5, 10, 20, 40]
    
    for seq_len in seq_lengths:
        print(f"--- Seq len: {seq_len} ---")
        
        model_g = GaussianKernelAttention()
        data = generate_data(300)
        train(model_g, data)
        results["gaussian"].append(evaluate(model_g))
        
        model_s = StandardAttention()
        train(model_s, data)
        results["standard"].append(evaluate(model_s))
        
        delta = (results["standard"][-1] - results["gaussian"][-1]) / results["standard"][-1] * 100
        print(f"  Gaussian: {results['gaussian'][-1]:.4f}, Standard: {results['standard'][-1]:.4f}, Delta: {delta:+.1f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    gauss_avg = np.mean(results["gaussian"])
    stand_avg = np.mean(results["standard"])
    improvement = (stand_avg - gauss_avg) / stand_avg * 100
    
    print(f"Gaussian avg: {gauss_avg:.4f}")
    print(f"Standard avg: {stand_avg:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    status = "✅ SUPPORTED" if improvement > 0 else "❌ REFUTED"
    print(f"\nStatus: {status}")
    
    return results


if __name__ == "__main__":
    main()