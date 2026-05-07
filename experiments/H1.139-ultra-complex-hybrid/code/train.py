#!/usr/bin/env python3
"""H1.139: Ultra-Complex Hybrid Tasks (60-100 steps)

Building on H1.138 (Hybrid wins 3/5 on 30-50 step sequences),
tests if hybrid architecture maintains advantage on even more complex tasks (60-100 steps).
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UltraComplexEnv:
    """Ultra-complex task with multiple phases and objectives"""
    def __init__(self):
        self.state_dim = 20
        self.action_dim = 10
    
    def reset(self):
        state = np.random.randn(self.state_dim) * 0.1
        state[:5] = 0  # Initial position
        return state
    
    def step(self, state, action):
        state = state.copy()
        action_arr = np.array(action).flatten()
        
        # Phase 1: Move to target (steps 0-30)
        # Phase 2: Manipulate (steps 30-60)
        # Phase 3: Complete (steps 60+)
        
        for i in range(min(10, len(action_arr))):
            state[i] += action_arr[i] * 0.05
        
        # Add complexity with non-linear dynamics
        state[5:10] = np.tanh(state[:5]) * 0.5 + np.random.randn(5) * 0.02
        state[10:15] = state[5:10] * np.sin(state[0]) + np.random.randn(5) * 0.01
        state[15:] = state[10:15] * 0.8
        
        reward = -np.sum(state[:10] ** 2) - np.sum(state[10:15] ** 2) * 0.5
        return state, reward


class Baseline(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_heads=8):
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
    """SSM+Attention hybrid from H1.138"""
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.select = nn.Linear(state_dim, hidden_dim)
        
        self.num_heads = 8
        self.head_dim = hidden_dim // self.num_heads
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        self.out = nn.Linear(hidden_dim, action_dim)
        self.alpha = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        sel = torch.tanh(self.select(state))
        
        q = self.q(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn_out = attn @ v
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        
        sel_out = attn_out * sel
        
        w = torch.sigmoid(self.alpha)
        out = w * self.out(sel_out[:, -1, :]) + (1-w) * self.out(attn_out[:, -1, :])
        
        return out


def train_eval(env, policy, seq_len, epochs=150):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0005)
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
    print("H1.139: Ultra-Complex Hybrid Tasks (60-100 steps)")
    print("=" * 60)
    
    results = {}
    env = UltraComplexEnv()
    
    seq_lengths = [60, 70, 80, 90, 100]
    
    for seq_len in seq_lengths:
        print(f"\n--- Seq Length: {seq_len} ---")
        
        baseline = Baseline(20, 10, hidden_dim=512).to(device)
        mean, std = train_eval(env, baseline, seq_len)
        results[f"Baseline_{seq_len}"] = (mean, std)
        print(f"  Baseline: {mean:.4f} ± {std:.4f}")
        
        attn = AttentionPolicy(20, 10, hidden_dim=512, num_heads=8).to(device)
        mean, std = train_eval(env, attn, seq_len)
        results[f"Attention_{seq_len}"] = (mean, std)
        print(f"  Attention: {mean:.4f} ± {std:.4f}")
        
        hybrid = SSMAttentionHybrid(20, 10, hidden_dim=512).to(device)
        mean, std = train_eval(env, hybrid, seq_len)
        results[f"Hybrid_{seq_len}"] = (mean, std)
        print(f"  Hybrid: {mean:.4f} ± {std:.4f}")
    
    print("\n" + "=" * 60)
    print("Summary (Higher reward = Better):")
    print("=" * 60)
    
    hybrid_wins = 0
    attn_wins = 0
    improvements = []
    
    for seq_len in seq_lengths:
        base = results[f"Baseline_{seq_len}"][0]
        att = results[f"Attention_{seq_len}"][0]
        hyb = results[f"Hybrid_{seq_len}"][0]
        
        best = max(base, att, hyb)
        if hyb == best:
            hybrid_wins += 1
        if att == best:
            attn_wins += 1
        
        imp = (hyb - base) / abs(base) * 100 if base != 0 else 0
        improvements.append(imp)
        
        print(f"  {seq_len}: Baseline={base:.2f}, Attn={att:.2f}, Hybrid={hyb:.2f}, Hybrid Δ={imp:.1f}%")
    
    print(f"\nHybrid wins: {hybrid_wins}/{len(seq_lengths)}")
    print(f"Attention wins: {attn_wins}/{len(seq_lengths)}")
    print(f"Average Hybrid improvement: {np.mean(improvements):.1f}%")
    
    return results


if __name__ == "__main__":
    run_experiment()