#!/usr/bin/env python3
"""H1.106: Attention on Extreme Multi-Step Tasks (40-60 steps)"""
import numpy as np
import torch
import torch.nn as nn
import json

np.random.seed(42)
torch.manual_seed(42)

class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )
    
    def forward(self, x):
        return self.net(x)

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(24, 256)
        self.self_attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )
    
    def forward(self, x):
        h = self.embed(x).unsqueeze(1)
        h_attn, _ = self.self_attn(h, h, h)
        return self.fc(h_attn.squeeze(1))

results = {}
criterion = nn.MSELoss()

for n_steps in [40, 45, 50, 55, 60]:
    print(f"\n=== {n_steps} steps ===")
    X, y = [], []
    for _ in range(200):
        for t in range(n_steps):
            s = np.random.randn(16) * 0.1
            a = np.random.randn(8) * 0.1
            action_16 = np.concatenate([a, np.zeros(8)])
            s_next = np.exp(-0.05 * t) * s + action_16 + np.random.randn(16) * 0.01
            X.append(np.concatenate([s, a]))
            y.append(s_next)
    
    X = np.float32(X)
    y = np.float32(y)
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    
    # Training concat
    cm = ConcatModel()
    opt = torch.optim.Adam(cm.parameters(), lr=1e-3)
    for epoch in range(100):
        opt.zero_grad()
        loss = criterion(cm(X_t), y_t).mean()
        loss.backward()
        opt.step()
    concat_loss = criterion(cm(X_t), y_t).mean().item()
    
    # Training attention
    am = AttentionModel()
    opt = torch.optim.Adam(am.parameters(), lr=1e-3)
    for epoch in range(100):
        opt.zero_grad()
        loss = criterion(am(X_t), y_t).mean()
        loss.backward()
        opt.step()
    attn_loss = criterion(am(X_t), y_t).mean().item()
    
    delta = (concat_loss - attn_loss) / concat_loss * 100
    results[n_steps] = {'concat': concat_loss, 'attn': attn_loss, 'delta': delta}
    print(f"Concat: {concat_loss:.6f}, Attn: {attn_loss:.6f}, Delta: {delta:+.1f}%")

avg = np.mean([r['delta'] for r in results.values()])
print(f"\n=== SUMMARY: avg {avg:+.1f}% ===")
status = "SUPPORTED" if avg > 50 else "MARGINAL" if avg > 0 else "REFUTED"
print(f"Status: {status}")

with open('experiments/H1.106-extreme-multi-step/results.json', 'w') as f:
    json.dump({**results, 'summary': {'avg_delta': avg, 'status': status}}, f, indent=2)