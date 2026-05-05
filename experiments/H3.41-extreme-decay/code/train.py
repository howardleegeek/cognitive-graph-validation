"""H3.41: Test even lower decay values"""

import numpy as np
import torch
import torch.nn as nn


class DecayAttention(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=512, num_heads=8, decay=0.9):
        super().__init__()
        self.decay = decay
        self.head_dim = hidden_dim // num_heads
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        B, T, D = x.shape
        x = self.input_proj(x)
        
        Q = self.q_proj(x).view(B, T, 8, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, 8, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, 8, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        t_idx = torch.arange(T, device=x.device).float()
        attn = attn * (self.decay ** t_idx).view(1, 1, T, 1)
        
        out = torch.matmul(torch.softmax(attn, dim=-1), V)
        out = out.transpose(1, 2).reshape(B, T, 512)
        return self.out_proj(out)


np.random.seed(42)
torch.manual_seed(42)

X = torch.randn(500, 30, 64) * 0.5
Y = X.sum(dim=1, keepdim=True).expand(-1, 30, -1) + torch.randn(500, 30, 64) * 0.3
train_split = 400

results = {}
for decay in [0.5, 0.3, 0.1]:
    model = DecayAttention(64, 512, decay=decay)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(50):
        loss = nn.MSELoss()(model(X[:train_split]), Y[:train_split])
        opt.zero_grad(); loss.backward(); opt.step()
    mse = nn.MSELoss()(model(X[train_split:]), Y[train_split:]).item()
    results[decay] = mse
    print(f"decay={decay}: MSE={mse:.4f}")

baseline = results[0.5]
for d in [0.3, 0.1]:
    print(f"{d}: {(baseline - results[d])/baseline*100:+.1f}%")