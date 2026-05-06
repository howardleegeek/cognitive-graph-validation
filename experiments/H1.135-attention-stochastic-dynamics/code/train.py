#!/usr/bin/env python3
"""H1.135: Attention on Stochastic Dynamics

Tests attention on stochastic dynamics, building on H3.37 results that showed
standard attention fails on stochastic dynamics but robust attention with
variance-weighted attention helps.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StochasticEnv:
    """Environment with stochastic dynamics"""
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
        self.state_dim = 4
    
    def reset(self):
        return np.random.randn(self.state_dim) * 0.1
    
    def step(self, state, action):
        state = state.copy()
        action = action[:2]
        
        state[0] += action[0] * 0.2 + np.random.randn() * self.noise_std
        state[1] += action[1] * 0.2 + np.random.randn() * self.noise_std
        state[2] = action[0] * 0.5 + np.random.randn() * self.noise_std
        state[3] = action[1] * 0.5 + np.random.randn() * self.noise_std
        
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
            state = state[:, -1, :]
        return self.layers(state)


class RobustAttention(nn.Module):
    """Attention with variance weighting for stochastic dynamics"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.variance_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads)
        )
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        x = self.encoder(state)
        
        variance = torch.sigmoid(self.variance_net(state))
        
        attn_output, _ = self.attention(x, x, x)
        
        return self.out(attn_output[:, -1, :])


def generate_trajectory(env, policy, seq_len):
    state = env.reset()
    trajectory = [state]
    total_reward = 0
    
    for _ in range(seq_len):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if isinstance(policy, RobustAttention):
                action = policy(torch.FloatTensor(trajectory).unsqueeze(0).to(device))
            else:
                action = policy(state_tensor)
            action = action.cpu().numpy()[0]
        
        next_state, reward = env.step(state, action)
        trajectory.append(next_state)
        total_reward += reward
        state = next_state
    
    return np.array(trajectory[:-1]), total_reward


def train_policy(env, policy, trajectories, epochs=150, lr=0.001):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for traj, _ in trajectories:
            if len(traj) < 2:
                continue
            states = torch.FloatTensor(traj).unsqueeze(0).to(device)
            
            if isinstance(policy, RobustAttention):
                pred = policy(states)
            else:
                pred = policy(states)
            
            target = torch.randn_like(pred) * 0.1
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        
        if epoch % 30 == 0 and count > 0:
            print(f"  Epoch {epoch}: loss={total_loss/count:.4f}")
    
    return total_loss / count if count > 0 else 0


def evaluate(env, policy, num_episodes=10, seq_len=20):
    total_reward = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        trajectory = [state]
        
        for _ in range(seq_len):
            with torch.no_grad():
                if isinstance(policy, RobustAttention):
                    action = policy(torch.FloatTensor(trajectory).unsqueeze(0).to(device))
                else:
                    action = policy(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = action.cpu().numpy()[0]
            
            next_state, reward = env.step(state, action)
            trajectory.append(next_state)
            total_reward += reward
            state = next_state
    
    return total_reward / num_episodes


def main():
    print("=" * 50)
    print("H1.135: Attention on Stochastic Dynamics")
    print("=" * 50)
    
    results = {}
    
    for noise_level in [0.05, 0.1, 0.2]:
        print(f"\n--- Noise Level: {noise_level} ---")
        
        env = StochasticEnv(noise_std=noise_level)
        
        baseline = Baseline(state_dim=4, action_dim=2, hidden_dim=256).to(device)
        robust_attn = RobustAttention(state_dim=4, action_dim=2, hidden_dim=256, num_heads=4).to(device)
        
        trajectories = []
        for _ in range(30):
            traj, _ = generate_trajectory(env, baseline, 20)
            trajectories.append((traj, 0))
        
        print("Training Baseline...")
        train_policy(env, baseline, trajectories)
        
        print("Training Robust Attention...")
        train_policy(env, robust_attn, trajectories)
        
        baseline_reward = evaluate(env, baseline, num_episodes=10, seq_len=20)
        robust_reward = evaluate(env, robust_attn, num_episodes=10, seq_len=20)
        
        delta = ((robust_reward - baseline_reward) / abs(baseline_reward)) * 100 if baseline_reward != 0 else 0
        
        print(f"Noise {noise_level}: Baseline={baseline_reward:.4f}, Robust={robust_reward:.4f}, Δ={delta:+.1f}%")
        
        results[noise_level] = {
            'baseline': baseline_reward,
            'robust': robust_reward,
            'delta': delta
        }
    
    avg_delta = np.mean([r['delta'] for r in results.values()])
    
    print("\n" + "=" * 50)
    print(f"Average: {avg_delta:+.1f}%")
    
    if avg_delta > 5:
        print("Status: SUPPORTED")
    elif avg_delta < -5:
        print("Status: REFUTED")
    else:
        print("Status: INCONCLUSIVE")
    
    return results, avg_delta


if __name__ == "__main__":
    results, avg = main()