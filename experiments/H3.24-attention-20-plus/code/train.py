#!/usr/bin/env python3
"""
H3.24: Attention on 20+ Step Sequences

Test whether attention mechanisms provide benefit over concatenation on longer sequences.
Based on H3.4 finding that attention wins at 24 and 30 steps but not on shorter.

This tests: At what sequence length does attention start to outperform concatenation?
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import json


@dataclass
class Config:
    seed: int = 42
    seq_lengths: List[int] = None
    n_trials: int = 3
    hidden_dim: int = 256
    n_epochs: int = 100
    lr: float = 0.001
    
    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [5, 10, 15, 20, 25, 30, 35, 40]


class ConcatenationFusion(nn.Module):
    """Concatenation-based fusion baseline"""
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
        # states: (B, T, D_s), actions: (B, T, D_a)
        B, T, _ = states.shape
        # Concatenate all timesteps
        x = torch.cat([states, actions], dim=-1)  # (B, T, D_s + D_a)
        # Process each timestep
        h = self.encoder(x)  # (B, T, hidden)
        # Predict next state
        pred = self.predictor(h)  # (B, T, D_s)
        return pred


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_heads=4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
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
        B, T, _ = states.shape
        
        # Encode each modality
        s_enc = self.state_encoder(states)  # (B, T, hidden)
        a_enc = self.action_encoder(actions)  # (B, T, hidden)
        
        # Cross-attention: attend to actions given states
        h, _ = self.cross_attn(s_enc, a_enc, a_enc)
        h = self.norm(h + s_enc)  # residual
        
        # Predict next state
        pred = self.predictor(h)  # (B, T, D_s)
        return pred


def generate_trajectory_data(seq_length, batch_size=64, state_dim=14, action_dim=7):
    """Generate synthetic trajectory data"""
    states = []
    actions = []
    next_states = []
    
    for _ in range(batch_size):
        # Random initial state
        s = np.random.randn(state_dim).astype(np.float32)
        s = np.tanh(s)  # bound to [-1, 1]
        
        traj_states = [s]
        traj_actions = []
        traj_next = []
        
        for t in range(seq_length):
            # Random action
            a = np.random.randn(action_dim).astype(np.float32) * 0.1
            a = np.tanh(a)
            
            # Next state (with some dynamics)
            s_next = s + 0.1 * np.tanh(np.random.randn(state_dim).astype(np.float32))
            s_next = np.clip(s_next, -1, 1)
            
            traj_states.append(s)
            traj_actions.append(a)
            traj_next.append(s_next)
            
            s = s_next
        
        states.append(traj_states[:-1])
        actions.append(traj_actions)
        next_states.append(traj_next)
    
    # (B, T, D)
    states = torch.tensor(np.array(states))
    actions = torch.tensor(np.array(actions))
    next_states = torch.tensor(np.array(next_states))
    
    return states, actions, next_states


def train_and_evaluate(model, states, actions, next_states, config):
    """Train model and evaluate MSE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(config.n_epochs):
        optimizer.zero_grad()
        pred = model(states, actions)
        loss = criterion(pred, next_states)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred = model(states, actions)
        mse = criterion(pred, next_states).item()
    
    return mse


def run_trial(seq_length, config, seed):
    """Run a single trial"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = 14
    action_dim = 7
    
    # Generate data
    states, actions, next_states = generate_trajectory_data(seq_length, batch_size=64)
    
    # Train both models
    concat_model = ConcatenationFusion(state_dim, action_dim, config.hidden_dim)
    attn_model = AttentionFusion(state_dim, action_dim, config.hidden_dim)
    
    concat_mse = train_and_evaluate(concat_model, states, actions, next_states, config)
    attn_mse = train_and_evaluate(attn_model, states, actions, next_states, config)
    
    return concat_mse, attn_mse


def main():
    config = Config()
    
    results = {
        "experiment": "H3.24",
        "description": "Attention on 20+ step sequences",
        "findings": [],
    }
    
    print("=" * 60)
    print("H3.24: Attention on 20+ Step Sequences")
    print("=" * 60)
    
    for seq_len in config.seq_lengths:
        concat_mses = []
        attn_mses = []
        
        for trial in range(config.n_trials):
            seed = config.seed + trial * 1000
            concat_mse, attn_mse = run_trial(seq_len, config, seed)
            concat_mses.append(concat_mse)
            attn_mses.append(attn_mse)
        
        concat_avg = np.mean(concat_mses)
        attn_avg = np.mean(attn_mses)
        
        if concat_avg > 0:
            delta = (attn_avg - concat_avg) / concat_avg * 100
        else:
            delta = 0
        
        winner = "ATTN" if attn_avg < concat_avg else "CONCAT"
        
        print(f"Seq {seq_len:2d}: Concat={concat_avg:.6f}, Attn={attn_avg:.6f}, Δ={delta:+.1f}%, Winner={winner}")
        
        results["findings"].append({
            "seq_length": seq_len,
            "concat_mse": concat_avg,
            "attn_mse": attn_avg,
            "delta": delta,
            "winner": winner,
        })
    
    # Find threshold
    wins_at = [f["seq_length"] for f in results["findings"] if f["winner"] == "ATTN"]
    if wins_at:
        print(f"\nAttention wins at sequence lengths: {wins_at}")
        results["threshold"] = min(wins_at) if wins_at else None
    else:
        print(f"\nConcatenation wins at all lengths")
        results["threshold"] = None
    
    # Summary
    concat_overall = np.mean([f["concat_mse"] for f in results["findings"]])
    attn_overall = np.mean([f["attn_mse"] for f in results["findings"]])
    
    if concat_overall > 0:
        avg_delta = (attn_overall - concat_overall) / concat_overall * 100
    else:
        avg_delta = 0
    
    results["summary"] = {
        "concat_overall": concat_overall,
        "attn_overall": attn_overall,
        "avg_delta": avg_delta,
    }
    
    print(f"\nOverall: Concat={concat_overall:.6f}, Attn={attn_overall:.6f}, Δ={avg_delta:+.1f}%")
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
    
    return results


if __name__ == "__main__":
    main()