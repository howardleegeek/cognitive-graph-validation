#!/usr/bin/env python3
"""H3.62: Causal Attention for Continuous Control

Tests causal (unidirectional) attention for continuous control tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Baseline(nn.Module):
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
            state = state[:, -1, :]
        return self.layers(state)


class CausalAttention(nn.Module):
    """Causal (unidirectional) attention"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        self.scale = (hidden_dim // num_heads) ** 0.5
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        batch, seq, _ = state.shape
        
        Q = self.q(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        K = self.k(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        V = self.v(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        
        # Causal mask: only attend to previous timesteps
        mask = torch.triu(torch.ones(seq, seq, device=state.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch, self.num_heads, 1, 1)
        
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = out.mean(dim=1)
        
        return self.out(out)


def create_task(batch_size, seq_len, state_dim=32, action_dim=14, difficulty=0.5):
    """Create continuous control task"""
    state = torch.randn(batch_size, seq_len, state_dim)
    # Add temporal structure
    for t in range(1, seq_len):
        state[:, t, :] = state[:, t-1, :] + torch.randn_like(state[:, t, :]) * difficulty * 0.1
    
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
    print("=" * 50)
    print("H3.62: Causal Attention")
    print("=" * 50)
    
    results = {}
    for seq_len in [50, 100]:
        train_state, train_target = create_task(64, seq_len)
        eval_state, eval_target = create_task(32, seq_len)
        
        base = Baseline(32, 14).to(device)
        base_mse = train_eval(base, train_state, train_target, eval_state, eval_target)
        
        causal = CausalAttention(32, 14).to(device)
        causal_mse = train_eval(causal, train_state, train_target, eval_state, eval_target)
        
        imp = (base_mse - causal_mse) / (base_mse + 1e-6) * 100
        print(f"Len {seq_len}: Baseline={base_mse:.4f}, Causal={causal_mse:.4f}, Δ={imp:+.1f}%")
        results[seq_len] = imp
    
    avg = np.mean(list(results.values()))
    print(f"\nAvg: {avg:+.1f}%")
    print(f"Status: {'SUPPORTED' if avg > 5 else 'REFUTED'}")
    return results


if __name__ == "__main__":
    run()