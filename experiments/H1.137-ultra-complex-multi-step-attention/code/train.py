#!/usr/bin/env python3
"""H1.137: Adaptive Attention on Ultra-Complex Multi-Step Tasks (40-60 steps)

Building on H3.69 (+34.2% at 20-30) and H3.70 (-34.6% at 30-50),
tests adaptive attention mechanisms to extend the crossover point to longer sequences.

Key insight: crossover at ~25 timesteps, adaptive decay may extend this.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UltraComplexTaskEnv:
    """Ultra-complex multi-step task with compositional subgoals (40-60 steps)"""
    def __init__(self, num_objects=4, num_subgoals=8):
        self.num_objects = num_objects
        self.num_subgoals = num_subgoals
        self.state_dim = num_objects * 4  # x, y, vx, vy for each object
    
    def reset(self):
        return np.random.randn(self.state_dim) * 0.1
    
    def step(self, state, action):
        """Complex multi-scale dynamics"""
        state = state.copy()
        action = action[:self.num_objects]
        
        # Multi-scale dynamics (oscillatory structure for attention to exploit)
        for i in range(self.num_objects):
            idx = i * 4
            state[idx] += action[i] * 0.1
            state[idx + 1] += action[i] * 0.05
            # Add oscillatory structure to exploit
            state[idx + 2] = action[i] * 0.8 * np.sin(idx * 0.1)
            state[idx + 3] = action[i] * 0.4 * np.cos(idx * 0.1)
        
        reward = -np.sum(state ** 2)
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
            state = state[:, -1, :]
        return self.layers(state)


class AdaptiveAttentionPolicy(nn.Module):
    """Attention with adaptive decay based on sequence complexity"""
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
        self.out = nn.Linear(hidden_dim, action_dim)
        
        # Adaptive decay network
        self.decay_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_heads)
        )
        
        self.action_gate = nn.Linear(state_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        # Compute adaptive decay based on complexity
        last_state = state[:, -1, :]
        decay_weights = torch.sigmoid(self.decay_net(last_state))  # (B, num_heads)
        
        # Multi-head attention with adaptive decay
        q = self.q(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply learned decay weighting per head
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Adaptive decay modulation
        decay_mod = decay_weights.unsqueeze(-1).unsqueeze(-1)  # (B, num_heads, 1, 1)
        attn = attn * decay_mod
        
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        attn_out = self.out(out[:, -1, :])
        gate = torch.sigmoid(self.action_gate(last_state))
        
        return attn_out * gate


class FixedDecayAttentionPolicy(nn.Module):
    """Standard attention with fixed decay"""
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_heads=8, decay=0.8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.decay = decay
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
        self.out = nn.Linear(hidden_dim, action_dim)
        
        self.action_gate = nn.Linear(state_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        q = self.q(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply decay mask
        mask = torch.tril(torch.ones(T, T, device=device)) * (1 - self.decay) + self.decay
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        attn_out = self.out(out[:, -1, :])
        gate = torch.sigmoid(self.action_gate(state[:, -1, :]))
        
        return attn_out * gate


def generate_trajectory(env, policy, seq_len):
    """Generate a trajectory of given length"""
    state = env.reset()
    trajectory = [state]
    total_reward = 0
    
    for _ in range(seq_len):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if isinstance(policy, (AdaptiveAttentionPolicy, FixedDecayAttentionPolicy)):
                action = policy(torch.FloatTensor(trajectory).unsqueeze(0).to(device))
            else:
                action = policy(state_tensor)
            action = action.cpu().numpy()[0]
        
        next_state, reward = env.step(state, action)
        trajectory.append(next_state)
        total_reward += reward
        state = next_state
    
    return np.array(trajectory[:-1]), total_reward


def evaluate_policy(env, policy, seq_len, num_trajectories=20):
    """Evaluate policy performance"""
    rewards = []
    for _ in range(num_trajectories):
        _, reward = generate_trajectory(env, policy, seq_len)
        rewards.append(reward)
    return np.mean(rewards), np.std(rewards)


def train_policy(env, policy, seq_len, epochs=200, lr=0.001):
    """Train policy"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Generate training trajectories
    trajectories = []
    for _ in range(50):
        traj, _ = generate_trajectory(env, policy, seq_len)
        trajectories.append((traj, 0))
    
    for epoch in range(epochs):
        total_loss = 0
        for traj, _ in trajectories:
            if len(traj) < 2:
                continue
            states = torch.FloatTensor(traj).unsqueeze(0).to(device)
            
            if isinstance(policy, (AdaptiveAttentionPolicy, FixedDecayAttentionPolicy)):
                pred = policy(states)
            else:
                pred = policy(states)
            
            target = torch.randn_like(pred) * 0.1
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / epochs if epochs > 0 else total_loss


def run_experiment():
    print("=" * 60)
    print("H1.137: Adaptive Attention on Ultra-Complex Multi-Step Tasks")
    print("=" * 60)
    
    results = {}
    
    # Test different sequence lengths
    seq_lengths = [40, 45, 50, 55, 60]
    
    env = UltraComplexTaskEnv(num_objects=4, num_subgoals=8)
    state_dim = env.state_dim
    action_dim = env.num_objects
    
    print(f"\nState dim: {state_dim}, Action dim: {action_dim}")
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        # Test Baseline (concatenation)
        baseline = Baseline(state_dim, action_dim, hidden_dim=512).to(device)
        train_policy(env, baseline, seq_len, epochs=100)
        
        # Test Fixed Decay Attention (various decay values)
        for decay in [0.6, 0.7, 0.8, 0.9]:
            fixed_attn = FixedDecayAttentionPolicy(state_dim, action_dim, hidden_dim=512, decay=decay).to(device)
            train_policy(env, fixed_attn, seq_len, epochs=100)
            
            mean_reward, std_reward = evaluate_policy(env, fixed_attn, seq_len)
            results[f"FixedDecay_{decay}_{seq_len}"] = (mean_reward, std_reward)
            print(f"  FixedDecay({decay}): {mean_reward:.4f} ± {std_reward:.4f}")
        
        # Test Adaptive Attention
        adaptive_attn = AdaptiveAttentionPolicy(state_dim, action_dim, hidden_dim=512, num_heads=8).to(device)
        train_policy(env, adaptive_attn, seq_len, epochs=100)
        
        mean_reward, std_reward = evaluate_policy(env, adaptive_attn, seq_len)
        results[f"Adaptive_{seq_len}"] = (mean_reward, std_reward)
        print(f"  Adaptive: {mean_reward:.4f} ± {std_reward:.4f}")
    
    # Compute improvements
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for seq_len in seq_lengths:
        baseline_rewards = [results[f"FixedDecay_{d}_{seq_len}"][0] for d in [0.6, 0.7, 0.8, 0.9]]
        adaptive_reward = results[f"Adaptive_{seq_len}"][0]
        
        best_fixed = max(baseline_rewards)
        
        if best_fixed > 0:
            improvement = ((adaptive_reward - best_fixed) / abs(best_fixed)) * 100
        else:
            improvement = 0
        
        print(f"Seq {seq_len}: Adaptive vs Best Fixed = {improvement:+.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_experiment()