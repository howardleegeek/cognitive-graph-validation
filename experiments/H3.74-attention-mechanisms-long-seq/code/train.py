#!/usr/bin/env python3
"""H3.74: Attention Mechanisms Comparison on Long Sequences (40-60 steps)

Building on:
- H3.69: +34.2% attention on 20-30 timesteps
- H3.72: +6.0% SSM on 30-50 timesteps
- H3.71: -45.2% decay attention (refuted)

Tests different attention mechanisms on 40-60 step sequences.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LongSeqEnv:
    """Long sequence task with temporal dependencies"""
    def __init__(self):
        self.state_dim = 16
        self.action_dim = 8
    
    def reset(self):
        return np.random.randn(self.state_dim) * 0.1
    
    def step(self, state, action):
        state = state.copy()
        action_arr = np.array(action).flatten()
        
        for i in range(min(8, len(action_arr))):
            state[i] += action_arr[i] * 0.08
        
        # Temporal dependencies - state depends on previous states
        state[8:] = np.mean(state[:8]) * 0.5 + np.random.randn(8) * 0.02
        
        reward = -np.sum(state ** 2)
        return state, reward


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
            return self.layers(state[:, -1, :])
        return self.layers(state)


class StandardAttention(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        q = self.q(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        return self.out(out[:, -1, :])


class LinearAttention(nn.Module):
    """Linear attention (kernel-based)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.key = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Linear(state_dim, hidden_dim)
        self.query = nn.Linear(state_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        k = torch.tanh(self.key(state))
        v = self.value(state)
        q = self.query(state)
        
        # Linear attention: attention = (Q @ K^T) @ V
        # Using feature map approximation
        q = q.unsqueeze(2)  # (B, T, 1, D)
        k = k.unsqueeze(1)  # (B, 1, T, D)
        
        attn = torch.tanh(q * k).sum(dim=-1)  # (B, T, T)
        attn = attn / (T ** 0.5)
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('btk,bkd->btd', attn, v)
        
        return self.out(out[:, -1, :])


class CausalAttention(nn.Module):
    """Causal attention (autoregressive)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        q = self.q(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        return self.out(out[:, -1, :])


class GatedAttention(nn.Module):
    """Gated attention with learned gating"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        self.gate = nn.Linear(state_dim, num_heads)
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        q = self.q(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # Apply gating
        gates = torch.sigmoid(self.gate(state)).unsqueeze(-1).transpose(1, 2)
        out = (attn * gates) @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        return self.out(out[:, -1, :])


def train_eval(env, policy, seq_len, epochs=100):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        traj = [env.reset()]
        for _ in range(seq_len):
            with torch.no_grad():
                state_in = torch.FloatTensor(traj).unsqueeze(0).to(device)
                action = policy(state_in).cpu().numpy()[0]
            next_state, _ = env.step(traj[-1], action)
            traj.append(next_state)
        
        states = torch.FloatTensor(np.array(traj[:-1])).unsqueeze(0).to(device)
        targets = torch.randn(action.shape).to(device) * 0.1
        
        pred = policy(states)
        loss = criterion(pred, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    rewards = []
    for _ in range(20):
        traj = [env.reset()]
        for _ in range(seq_len):
            with torch.no_grad():
                state_in = torch.FloatTensor(traj).unsqueeze(0).to(device)
                action = policy(state_in).cpu().numpy()[0]
            next_state, reward = env.step(traj[-1], action)
            traj.append(next_state)
            rewards.append(reward)
    
    return np.mean(rewards), np.std(rewards)


def run_experiment():
    print("=" * 60)
    print("H3.74: Attention Mechanisms on Long Sequences (40-60 steps)")
    print("=" * 60)
    
    results = {}
    env = LongSeqEnv()
    
    seq_lengths = [40, 45, 50, 55, 60]
    
    policies = {
        "Baseline": lambda: Baseline(16, 8, hidden_dim=256).to(device),
        "Standard": lambda: StandardAttention(16, 8, hidden_dim=256, num_heads=4).to(device),
        "Linear": lambda: LinearAttention(16, 8, hidden_dim=256).to(device),
        "Causal": lambda: CausalAttention(16, 8, hidden_dim=256, num_heads=4).to(device),
        "Gated": lambda: GatedAttention(16, 8, hidden_dim=256, num_heads=4).to(device),
    }
    
    for seq_len in seq_lengths:
        print(f"\n--- Seq Length: {seq_len} ---")
        
        for name, factory in policies.items():
            policy = factory()
            mean, std = train_eval(env, policy, seq_len)
            results[f"{name}_{seq_len}"] = (mean, std)
            print(f"  {name}: {mean:.4f} ± {std:.4f}")
    
    print("\n" + "=" * 60)
    print("Summary (Higher reward = Better):")
    print("=" * 60)
    
    best_per_seq = {}
    for seq_len in seq_lengths:
        best_name = None
        best_val = float('-inf')
        for name in policies.keys():
            val = results[f"{name}_{seq_len}"][0]
            if val > best_val:
                best_val = val
                best_name = name
        best_per_seq[seq_len] = (best_name, best_val)
        print(f"  {seq_len}: {best_name} wins ({best_val:.2f})")
    
    # Count wins
    win_counts = {name: 0 for name in policies.keys()}
    for seq_len, (name, _) in best_per_seq.items():
        win_counts[name] += 1
    
    print("\nWin counts:")
    for name, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}/{len(seq_lengths)}")
    
    # Compare to baseline
    print("\nImprovement over Baseline:")
    for name in ["Standard", "Linear", "Causal", "Gated"]:
        improvements = []
        for seq_len in seq_lengths:
            base = results[f"Baseline_{seq_len}"][0]
            curr = results[f"{name}_{seq_len}"][0]
            imp = (curr - base) / abs(base) * 100 if base != 0 else 0
            improvements.append(imp)
        avg_imp = np.mean(improvements)
        print(f"  {name}: {avg_imp:.1f}% avg")
    
    return results


if __name__ == "__main__":
    run_experiment()