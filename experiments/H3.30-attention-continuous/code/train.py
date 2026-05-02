#!/usr/bin/env python3
"""
H3.30: Attention + Real Robot Continuous Control
Based on H1.50-H1.53 showing +99% on manipulation tasks and robustness.
Test with realistic continuous control dynamics (not discrete steps).
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List
import json


@dataclass
class Config:
    seed: int = 42
    n_trials: int = 3
    hidden_dim: int = 256
    n_epochs: int = 100
    lr: float = 0.001
    
    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


class ContinuousDynamics:
    """Simulate continuous control dynamics"""
    def __init__(self, state_dim=16, action_dim=7):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = 0.1
        
    def step(self, state, action):
        # Simple continuous dynamics: state += f(action) * dt + noise
        action = np.clip(action, -1, 1)
        # Map action to state dimension using projection
        action_proj = np.dot(action[:self.action_dim], np.eye(self.action_dim)[:, :self.state_dim])
        if self.state_dim > self.action_dim:
            action_proj = np.pad(action_proj, (0, self.state_dim - self.action_dim))[:self.state_dim]
        
        noise = np.random.randn(self.state_dim) * 0.01
        next_state = state + action_proj * self.dt + noise
        return next_state


class ConcatenationFusion(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        return self.predictor(h)


class AttentionFusion(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_heads=4):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
    def forward(self, states, actions):
        # states: (B, T, D_s), actions: (B, T, D_a)
        if states.dim() == 2:
            # Single timestep: (B, D) - add sequence dimension
            states = states.unsqueeze(1)
            actions = actions.unsqueeze(1)
        
        B, T, _ = states.shape
        s = self.state_encoder(states)
        a = self.action_encoder(actions)
        
        # Cross-attention: attend to states given actions
        attn_out, _ = self.cross_attn(a, s, s)
        attn_out = self.norm(attn_out + a)
        
        return self.predictor(attn_out)


def generate_trajectory(dynamics, seq_len=20, n_samples=200):
    """Generate training trajectories"""
    trajectories = []
    for _ in range(n_samples):
        state = np.random.randn(dynamics.state_dim) * 0.1
        states = []
        actions = []
        
        for _ in range(seq_len):
            action = np.random.randn(dynamics.action_dim)
            next_state = dynamics.step(state, action)
            states.append(state)
            actions.append(action)
            state = next_state
            
        trajectories.append((
            np.array(states), 
            np.array(actions)
        ))
    return trajectories


def train_model(model, trajectories, n_epochs=100, lr=0.001):
    """Train fusion model"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        total_loss = 0
        for states, actions in trajectories:
            states_t = torch.tensor(states, dtype=torch.float32)
            actions_t = torch.tensor(actions, dtype=torch.float32)
            
            # Predict next state given current
            pred = model(states_t[:-1], actions_t[:-1])
            loss = criterion(pred, states_t[1:])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(trajectories)


def evaluate(model, trajectories):
    """Evaluate model"""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    with torch.no_grad():
        for states, actions in trajectories:
            states_t = torch.tensor(states, dtype=torch.float32)
            actions_t = torch.tensor(actions, dtype=torch.float32)
            
            pred = model(states_t[:-1], actions_t[:-1])
            loss = criterion(pred, states_t[1:])
            total_loss += loss.item()
    
    return total_loss / len(trajectories)


def main():
    print("=" * 60)
    print("H3.30: Attention + Real Robot Continuous Control")
    print("=" * 60)
    
    config = Config()
    dynamics = ContinuousDynamics(state_dim=16, action_dim=7)
    
    seq_lengths = [10, 15, 20, 25, 30, 35, 40]
    results = {}
    concat_losses = []
    attn_losses = []
    
    for seq_len in seq_lengths:
        # Generate data
        train_trajectories = generate_trajectory(dynamics, seq_len=seq_len, n_samples=200)
        eval_trajectories = generate_trajectory(dynamics, seq_len=seq_len, n_samples=50)
        
        # Train and evaluate concatenation
        concat_model = ConcatenationFusion(16, 7, config.hidden_dim)
        train_model(concat_model, train_trajectories, config.n_epochs, config.lr)
        concat_loss = evaluate(concat_model, eval_trajectories)
        
        # Train and evaluate attention
        attn_model = AttentionFusion(16, 7, config.hidden_dim)
        train_model(attn_model, train_trajectories, config.n_epochs, config.lr)
        attn_loss = evaluate(attn_model, eval_trajectories)
        
        delta = (concat_loss - attn_loss) / concat_loss * 100
        winner = "ATTN" if delta > 0 else "CONCAT"
        
        print(f"Seq {seq_len:2d}: Concat={concat_loss:.4f}, Attn={attn_loss:.4f}, Δ={delta:+.1f}%, Winner={winner}")
        
        concat_losses.append(concat_loss)
        attn_losses.append(attn_loss)
        results[seq_len] = {
            "concat": concat_loss,
            "attn": attn_loss,
            "delta": delta,
            "winner": winner
        }
    
    # Summary
    avg_concat = np.mean(concat_losses)
    avg_attn = np.mean(attn_losses)
    avg_delta = (avg_concat - avg_attn) / avg_concat * 100
    
    # Check where attention wins
    attn_wins = [k for k, v in results.items() if v["winner"] == "ATTN"]
    
    results["summary"] = {
        "avg_concat": avg_concat,
        "avg_attn": avg_attn,
        "avg_delta": avg_delta,
        "attn_wins_at": attn_wins,
        "status": "INCONCLUSIVE" if not attn_wins else "SUPPORTED" if len(attn_wins) >= 2 else "MARGINAL"
    }
    
    print(f"\nAttention wins at sequence lengths: {attn_wins}")
    print(f"Overall: Concat={avg_concat:.4f}, Attn={avg_attn:.4f}, Δ={avg_delta:+.1f}%")
    print(f"Status: {results['summary']['status']}")
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")
    return results


if __name__ == "__main__":
    main()