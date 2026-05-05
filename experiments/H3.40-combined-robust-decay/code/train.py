"""
H3.40: Simplified combined test
"""

import numpy as np
import torch
import torch.nn as nn


class DecayAttention(nn.Module):
    """Simple decay attention"""
    def __init__(self, input_dim=64, hidden_dim=512, num_heads=8, decay=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay = decay
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, mask=None):
        B, T, D = x.shape
        x = self.input_proj(x)
        
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        t_idx = torch.arange(T, device=x.device).float()
        decay_weights = (self.decay ** t_idx).view(1, 1, T, 1)
        attn_scores = attn_scores * decay_weights
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(B, T, self.hidden_dim)
        return self.out_proj(out)


class StandardAttention(nn.Module):
    """Standard attention"""
    def __init__(self, input_dim=64, hidden_dim=512, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, mask=None):
        B, T, D = x.shape
        x = self.input_proj(x)
        
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(B, T, self.hidden_dim)
        return self.out_proj(out)


def simulate():
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = {}
    
    n_samples = 500
    n_timesteps = 30
    input_dim = 64
    hidden_dim = 512
    
    X = torch.randn(n_samples, n_timesteps, input_dim) * 0.5
    
    sigma = 0.3
    epsilon = torch.randn(n_samples, n_timesteps, input_dim) * sigma
    
    Y = X.sum(dim=1, keepdim=True).expand(-1, n_timesteps, -1) + epsilon
    
    train_split = int(0.8 * n_samples)
    
    configs = [
        ('standard', StandardAttention(input_dim, hidden_dim)),
        ('decay-0.9', DecayAttention(input_dim, hidden_dim, decay=0.9)),
        ('decay-0.7', DecayAttention(input_dim, hidden_dim, decay=0.7)),
        ('decay-0.5', DecayAttention(input_dim, hidden_dim, decay=0.5)),
    ]
    
    for name, model in configs:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            pred = model(X[:train_split])
            loss = criterion(pred, Y[:train_split])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = model(X[train_split:])
            mse = criterion(pred, Y[train_split:]).item()
        
        results[name] = mse
    
    print("\n=== H3.40: Decay Attention Scaling ===")
    baseline = results['standard']
    for name, mse in results.items():
        pct = (baseline - mse) / baseline * 100
        print(f"{name}: MSE={mse:.4f} ({pct:+.1f}%)")
    
    best = min(results.items(), key=lambda x: x[1])
    print(f"\nBest: {best[0]} ({best[1]:.4f})")


if __name__ == "__main__":
    simulate()