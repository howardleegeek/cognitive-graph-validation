"""
H3.39: Simpler test for learned uncertainty attention
"""

import numpy as np
import torch
import torch.nn as nn


class SimpleQueryKeyDecay(nn.Module):
    """Query-key decay attention for stochastic dynamics"""
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
        
        decayed_k = K * (self.decay ** torch.arange(T, device=K.device).view(1, 1, T, 1))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(B, T, self.hidden_dim)
        return self.out_proj(out)


class NoDecayAttention(nn.Module):
    """Standard attention without decay"""
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
    
    configs = {
        'no-decay': {'decay': 1.0},
        'decay-0.9': {'decay': 0.9},
        'decay-0.8': {'decay': 0.8},
        'decay-0.7': {'decay': 0.7},
    }
    
    for name, cfg in configs.items():
        if 'decay' in cfg:
            model = SimpleQueryKeyDecay(input_dim, hidden_dim, decay=cfg['decay'])
        else:
            model = NoDecayAttention(input_dim, hidden_dim)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            model.train()
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
    
    print("\n=== H3.39: Stochastic Attention Results ===")
    for name, mse in results.items():
        print(f"{name}: MSE = {mse:.6f}")
    
    baseline = results.get('no-decay', results[list(results.keys())[0]])
    
    print("\nImprovement vs no-decay:")
    for name, mse in results.items():
        if name != 'no-decay':
            pct = (baseline - mse) / baseline * 100
            print(f"{name}: {pct:+.1f}%")
    
    best = min(results.items(), key=lambda x: x[1])
    print(f"\nBest: {best[0]} ({best[1]:.6f})")
    
    return results


if __name__ == "__main__":
    results = simulate()