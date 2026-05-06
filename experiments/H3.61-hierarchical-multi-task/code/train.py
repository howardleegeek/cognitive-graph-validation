#!/usr/bin/env python3
"""H3.61: Temporal Abstraction Attention for Robotic Tasks

Tests attention with temporal abstraction for long-horizon robotic manipulation.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StandardMLP(nn.Module):
    """Standard MLP baseline"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        if state.dim() == 3:
            state = state[:, -1, :]  # Use last timestep
        return self.layers(state)


class TemporalAttention(nn.Module):
    """Multi-scale temporal attention"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        # Different scale projections
        self.scale1 = nn.Linear(state_dim, hidden_dim)
        self.scale2 = nn.Linear(state_dim, hidden_dim)
        self.scale3 = nn.Linear(state_dim, hidden_dim)
        
        # Attention
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = (hidden_dim // num_heads) ** 0.5
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        batch, seq, _ = state.shape
        
        # Multi-scale features
        s1 = self.scale1(state)
        s2 = self.scale2(state)
        s3 = self.scale3(state)
        
        # Combine scales
        combined = s1 + s2 + s3
        
        # Attention
        Q = self.q(combined).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        K = self.k(combined).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        V = self.v(combined).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = out.mean(dim=1)
        
        return self.out(out)


def create_task(batch_size, seq_len, state_dim=32, action_dim=14):
    state = torch.randn(batch_size, seq_len, state_dim) * 0.5
    target = torch.randn(batch_size, action_dim) * 0.1
    return state, target


def train_eval(agent, train_state, train_target, eval_state, eval_target, epochs=100):
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for ep in range(epochs):
        agent.train()
        optimizer.zero_grad()
        output = agent(train_state)
        loss = criterion(output, train_target)
        loss.backward()
        optimizer.step()
    
    agent.eval()
    with torch.no_grad():
        output = agent(eval_state)
        mse = criterion(output, eval_target).item()
    
    return mse


def run():
    print("=" * 60)
    print("H3.61: Multi-Scale Temporal Attention")
    print("=" * 60)
    
    state_dim = 32
    action_dim = 14
    seq_lengths = [50, 100]
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n--- Length: {seq_len} ---")
        
        train_state, train_target = create_task(64, seq_len, state_dim, action_dim)
        eval_state, eval_target = create_task(32, seq_len, state_dim, action_dim)
        
        std = StandardMLP(state_dim, action_dim).to(device)
        std_mse = train_eval(std, train_state, train_target, eval_state, eval_target)
        
        attn = TemporalAttention(state_dim, action_dim).to(device)
        attn_mse = train_eval(attn, train_state, train_target, eval_state, eval_target)
        
        improvement = (std_mse - attn_mse) / (std_mse + 1e-6) * 100
        print(f"  Standard: {std_mse:.6f}, TemporalAttn: {attn_mse:.6f}, Δ: {improvement:+.1f}%")
        
        results[seq_len] = {'std': std_mse, 'attn': attn_mse, 'imp': improvement}
    
    avg_imp = np.mean([r['imp'] for r in results.values()])
    print(f"\nAvg: {avg_imp:+.1f}%")
    status = "SUPPORTED" if avg_imp > 5 else "REFUTED"
    print(f"Status: {status}")
    return results, status


if __name__ == "__main__":
    results, status = run()
    print(f"\nH3.61: {status}")