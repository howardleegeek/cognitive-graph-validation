#!/usr/bin/env python3
"""H3.63: Attention on Physics-Based Long Sequences (50-100 steps)

Tests attention on very long sequences with physics-based dynamics,
building on H3.35-36 results that showed attention wins with physics.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhysicsEnv:
    """Simple physics-based environment with mass-spring dynamics"""
    def __init__(self, mass=1.0, k=1.0, dt=0.1):
        self.mass = mass
        self.k = k
        self.dt = dt
    
    def step(self, state, action):
        x, v = state
        a = action[0]
        force = -self.k * x + a
        new_v = v + (force / self.mass) * self.dt
        new_x = x + new_v * self.dt
        return np.array([new_x, new_v])


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


class AttentionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q = nn.Linear(state_dim, hidden_dim)
        self.k = nn.Linear(state_dim, hidden_dim)
        self.v = nn.Linear(state_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        B, T, D = state.shape
        
        q = self.q(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(state).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        return self.out(out[:, -1, :])


def generate_trajectory(env, policy, seq_len, noise=0.01):
    """Generate a trajectory of given length"""
    state = np.array([0.0, 0.0])
    trajectory = []
    
    for _ in range(seq_len):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy(state_tensor).cpu().numpy()[0]
        
        action = action + np.random.randn(1) * noise
        next_state = env.step(state, action)
        trajectory.append(state)
        state = next_state
    
    return np.array(trajectory)


def train_policy(env, policy, trajectories, epochs=100, lr=0.001):
    """Train policy on trajectories"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for traj in trajectories:
            if len(traj) < 2:
                continue
            states = torch.FloatTensor(traj).unsqueeze(0).to(device)
            actions = torch.randn(1, len(traj), 1).to(device) * 0.1
            
            if isinstance(policy, AttentionPolicy):
                pred = policy(states)
            else:
                pred = policy(states)
            
            loss = criterion(pred, actions[:, -1, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/len(trajectories):.4f}")
    
    return total_loss / len(trajectories)


def evaluate(env, policy, num_episodes=10, seq_len=50):
    """Evaluate policy performance"""
    total_mse = 0
    
    for _ in range(num_episodes):
        state = np.array([0.0, 0.0])
        states = [state]
        
        for _ in range(seq_len):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                if isinstance(policy, AttentionPolicy):
                    action = policy(torch.FloatTensor(states).unsqueeze(0).to(device))
                else:
                    action = policy(state_tensor)
                action = action.cpu().numpy()[0]
            
            next_state = env.step(state, action)
            states.append(next_state)
            state = next_state
        
        states = np.array(states[:-1])
        target = np.zeros_like(states)
        mse = np.mean((states - target) ** 2)
        total_mse += mse
    
    return total_mse / num_episodes


def main():
    print("=" * 50)
    print("H3.63: Attention on Physics-Based Long Sequences")
    print("=" * 50)
    
    results = {}
    
    for seq_len in [50, 75, 100]:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        env = PhysicsEnv(mass=1.0, k=1.0, dt=0.1)
        
        baseline = Baseline(state_dim=2, action_dim=1, hidden_dim=256).to(device)
        attention = AttentionPolicy(state_dim=2, action_dim=1, hidden_dim=256, num_heads=4).to(device)
        
        trajectories = [generate_trajectory(env, baseline, seq_len) for _ in range(20)]
        
        print("Training Baseline...")
        train_policy(env, baseline, trajectories)
        
        print("Training Attention...")
        train_policy(env, attention, trajectories)
        
        baseline_mse = evaluate(env, baseline, num_episodes=10, seq_len=seq_len)
        attention_mse = evaluate(env, attention, num_episodes=10, seq_len=seq_len)
        
        delta = ((baseline_mse - attention_mse) / baseline_mse) * 100
        
        print(f"Len {seq_len}: Baseline={baseline_mse:.4f}, Attention={attention_mse:.4f}, Δ={delta:+.1f}%")
        
        results[seq_len] = {
            'baseline': baseline_mse,
            'attention': attention_mse,
            'delta': delta
        }
    
    avg_delta = np.mean([r['delta'] for r in results.values()])
    
    print("\n" + "=" * 50)
    print(f"Average: {avg_delta:+.1f}%")
    
    if avg_delta > 0:
        print("Status: SUPPORTED")
    else:
        print("Status: REFUTED")
    
    return results, avg_delta


if __name__ == "__main__":
    results, avg = main()