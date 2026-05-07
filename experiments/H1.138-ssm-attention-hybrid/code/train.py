#!/usr/bin/env python3
"""H1.138: SSM+Attention Hybrid for Ultra-Long Sequences

Building on H3.72 (+6% SSM at 30-50) and H3.69 (+34.2% attention at 20-30),
tests if SSM+Attention hybrid can capture both temporal patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UltraLongTaskEnv:
    """Ultra-long sequence task with mixed dynamics"""
    def __init__(self):
        self.state_dim = 16
        self.action_dim = 8  # Increased for SSM
    
    def reset(self):
        return np.random.randn(self.state_dim) * 0.1
    
    def step(self, state, action):
        """Mixed dynamics"""
        state = state.copy()
        
        # Simple vector addition
        action_arr = np.array(action).flatten()
        
        # Map action to state
        for i in range(min(8, len(action_arr))):
            state[i] += action_arr[i] * 0.1
        
        # Add some dynamics mixing
        state[8:] = state[:8] * 0.9 + np.random.randn(8) * 0.01
        
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


class AttentionPolicy(nn.Module):
    """Standard attention"""
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


class SSMAttentionHybrid(nn.Module):
    """Combines SSM-like selection + Attention"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Selection network (like SSM select)
        self.select = nn.Linear(state_dim, hidden_dim)
        
        # Attention
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        # Output
        self.out = nn.Linear(hidden_dim, action_dim)
        
        # Fusion weights
        self.alpha = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        # SSM-like selective output
        sel = torch.tanh(self.select(state))  # (B, T, hidden)
        
        # Attention
        q = self.q(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn_out = attn @ v
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        
        # Selective read
        sel_out = attn_out * sel
        
        # Weight both paths
        w = torch.sigmoid(self.alpha)
        
        out = w * self.out(sel_out[:, -1, :]) + (1-w) * self.out(attn_out[:, -1, :])
        
        return out


def train_eval(env, policy, seq_len, epochs=100):
    """Train and evaluate"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        traj = [env.reset()]
        for _ in range(seq_len):
            with torch.no_grad():
                if isinstance(policy, (AttentionPolicy, SSMAttentionHybrid)):
                    state_in = torch.FloatTensor(traj).unsqueeze(0).to(device)
                else:
                    state_in = torch.FloatTensor(traj[-1]).unsqueeze(0).to(device)
                action = policy(state_in).cpu().numpy()[0]
            next_state, _ = env.step(traj[-1], action)
            traj.append(next_state)
        
        # Train on the trajectory
        states = torch.FloatTensor(np.array(traj[:-1])).unsqueeze(0).to(device)
        targets = torch.randn(action.shape).to(device) * 0.1
        
        pred = policy(states)
        loss = criterion(pred, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    rewards = []
    for _ in range(20):
        traj = [env.reset()]
        for _ in range(seq_len):
            with torch.no_grad():
                if isinstance(policy, (AttentionPolicy, SSMAttentionHybrid)):
                    state_in = torch.FloatTensor(traj).unsqueeze(0).to(device)
                else:
                    state_in = torch.FloatTensor(traj[-1]).unsqueeze(0).to(device)
                action = policy(state_in).cpu().numpy()[0]
            next_state, reward = env.step(traj[-1], action)
            traj.append(next_state)
            rewards.append(reward)
    
    return np.mean(rewards), np.std(rewards)


def run_experiment():
    print("=" * 60)
    print("H1.138: SSM+Attention Hybrid for Ultra-Long Sequences")
    print("=" * 60)
    
    results = {}
    env = UltraLongTaskEnv()
    
    seq_lengths = [30, 35, 40, 45, 50]
    
    for seq_len in seq_lengths:
        print(f"\n--- Seq Length: {seq_len} ---")
        
        # Baseline
        baseline = Baseline(16, 8, hidden_dim=256).to(device)
        mean, std = train_eval(env, baseline, seq_len)
        results[f"Baseline_{seq_len}"] = (mean, std)
        print(f"  Baseline: {mean:.4f} ± {std:.4f}")
        
        # Attention
        attn = AttentionPolicy(16, 8, hidden_dim=256, num_heads=4).to(device)
        mean, std = train_eval(env, attn, seq_len)
        results[f"Attention_{seq_len}"] = (mean, std)
        print(f"  Attention: {mean:.4f} ± {std:.4f}")
        
        # Hybrid
        hybrid = SSMAttentionHybrid(16, 8, hidden_dim=256).to(device)
        mean, std = train_eval(env, hybrid, seq_len)
        results[f"Hybrid_{seq_len}"] = (mean, std)
        print(f"  Hybrid: {mean:.4f} ± {std:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary (Higher reward = Better):")
    print("=" * 60)
    
    hybrid_wins = 0
    for seq_len in seq_lengths:
        base = results[f"Baseline_{seq_len}"][0]
        att = results[f"Attention_{seq_len}"][0]
        hyb = results[f"Hybrid_{seq_len}"][0]
        
        print(f"  {seq_len}: Baseline={base:.2f}, Attn={att:.2f}, Hybrid={hyb:.2f}")
        
        if hyb > max(base, att):
            hybrid_wins += 1
    
    print(f"\nHybrid wins: {hybrid_wins}/{len(seq_lengths)}")
    
    return results


if __name__ == "__main__":
    run_experiment()