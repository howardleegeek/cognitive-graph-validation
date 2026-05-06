#!/usr/bin/env python3
"""H3.59: ALOHA-Style Long-Horizon Manipulation with Attention

Tests attention on ALOHA long-horizon manipulation tasks (50-200 steps)
with real robot manipulation patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConcatBaseline(nn.Module):
    """Simple concatenation baseline"""
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)


class AttentionPolicy(nn.Module):
    """Attention-based policy for long-horizon tasks"""
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_heads=8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(state_dim, hidden_dim)
        self.k_proj = nn.Linear(state_dim, hidden_dim)
        self.v_proj = nn.Linear(state_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)
        
        self.scale = (hidden_dim // num_heads) ** 0.5
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        batch, seq, _ = state.shape
        
        Q = self.q_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        K = self.k_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = out.mean(dim=1)
        
        return self.out_proj(out)


def create_aloha_task(batch_size, seq_len, state_dim=32, action_dim=14):
    """Create ALOHA-style manipulation task with realistic patterns"""
    # State: [ee_pos, object_pos, gripper_state, joint_pos, joint_vel]
    state = torch.zeros(batch_size, seq_len, state_dim)
    
    for b in range(batch_size):
        # Initial positions
        ee_pos = torch.randn(3) * 0.3
        obj_pos = torch.randn(3) * 0.2 + 0.1
        gripper = torch.rand(1)
        
        for t in range(seq_len):
            # Realistic manipulation motion
            progress = t / seq_len
            
            # Reach toward object
            if progress < 0.2:
                target = obj_pos
                ee_pos = ee_pos + (target - ee_pos) * 0.1
            # Grasp
            elif progress < 0.3:
                dist = (ee_pos - obj_pos).norm()
                if dist < 0.05:
                    gripper = torch.tensor([0.0])
            # Lift and move
            elif progress < 0.7:
                target = obj_pos + torch.tensor([0.2, 0.0, 0.3])
                ee_pos = ee_pos + (target - ee_pos) * 0.08
            # Place
            elif progress < 0.9:
                target = obj_pos + torch.tensor([0.2, 0.2, 0.0])
                ee_pos = ee_pos + (target - ee_pos) * 0.1
            # Release
            else:
                gripper = torch.tensor([1.0])
            
            # Compose state
            state[b, t, :3] = ee_pos
            state[b, t, 3:6] = obj_pos
            state[b, t, 6] = gripper
            state[b, t, 7:32] = torch.randn(25) * 0.01
    
    # Action targets: predict next action
    target = torch.randn(batch_size, action_dim) * 0.1
    
    return state, target


def train_eval(agent, train_states, train_targets, eval_states, eval_targets, epochs=200):
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for ep in range(epochs):
        agent.train()
        optimizer.zero_grad()
        
        if train_states.dim() == 3:
            output = agent(train_states)
            # Average over sequence dimension for attention model
            if output.dim() == 3:
                output = output.mean(dim=1)
        else:
            output = agent(train_states.unsqueeze(1))
        
        loss = criterion(output, train_targets)
        loss.backward()
        optimizer.step()
    
    agent.eval()
    with torch.no_grad():
        if eval_states.dim() == 3:
            output = agent(eval_states)
            if output.dim() == 3:
                output = output.mean(dim=1)
        else:
            output = agent(eval_states.unsqueeze(1))
        
        mse = criterion(output, eval_targets).item()
    
    return mse


def run():
    print("=" * 60)
    print("H3.59: ALOHA-Style Long-Horizon Manipulation")
    print("=" * 60)
    
    state_dim = 32
    action_dim = 14
    seq_lengths = [50, 100, 150, 200]
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        # Create tasks
        train_states, train_targets = create_aloha_task(
            batch_size=64, seq_len=seq_len, state_dim=state_dim, action_dim=action_dim
        )
        eval_states, eval_targets = create_aloha_task(
            batch_size=32, seq_len=seq_len, state_dim=state_dim, action_dim=action_dim
        )
        
        # Test concatenation baseline
        concat = ConcatBaseline(state_dim, action_dim).to(device)
        concat_mse = train_eval(concat, train_states, train_targets, eval_states, eval_targets)
        
        # Test attention
        attn = AttentionPolicy(state_dim, action_dim).to(device)
        attn_mse = train_eval(attn, train_states, train_targets, eval_states, eval_targets)
        
        improvement = (concat_mse - attn_mse) / (concat_mse + 1e-6) * 100
        print(f"  Concat MSE: {concat_mse:.6f}")
        print(f"  Attention MSE: {attn_mse:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results[seq_len] = {'concat': concat_mse, 'attn': attn_mse, 'improvement': improvement}
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    print(f"Average Improvement: {avg_improvement:+.1f}%")
    
    status = "SUPPORTED" if avg_improvement > 10 else ("MARGINAL" if avg_improvement > 0 else "REFUTED")
    print(f"Status: {status}")
    
    return results, status


if __name__ == "__main__":
    results, status = run()
    print(f"\nH3.59: {status}")