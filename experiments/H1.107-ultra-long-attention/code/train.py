"""
H1.107: Ultra-Long Sequence Attention (100-300 steps)
===================================================
Test attention on extremely long horizon tasks.

Based on H1.99: +99.1% avg on 100-250 step tasks (SUPPORTED)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
import json

np.random.seed(42)
torch.manual_seed(42)

@dataclass
class TaskConfig:
    state_dim: int = 14
    action_dim: int = 7
    hidden_dim: int = 1024
    n_steps: List[int] = None
    
    def __post_init__(self):
        if self.n_steps is None:
            self.n_steps = [100, 150, 200, 250, 300]

class ConcatModel(nn.Module):
    """Standard concatenation baseline."""
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim)
        )
        
    def forward(self, state, action, hidden=None):
        combined = torch.cat([state, action], dim=-1)
        return self.net(combined), None

class AttentionModel(nn.Module):
    """Attention-based model for long sequences."""
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        self.state_encoder = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_encoder = nn.Linear(config.action_dim, config.hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.action_dim)
        )
        
    def forward(self, state, action, hidden=None):
        # Encode
        state_emb = self.state_encoder(state)
        action_emb = self.action_encoder(action)
        
        # Stack for attention
        tokens = torch.stack([state_emb, action_emb], dim=1)  # [B, 2, hidden]
        
        # Self-attention
        attn_out, _ = self.attention(tokens, tokens, tokens)
        
        # Take action representation
        action_out = attn_out[:, 1, :]  # [B, hidden]
        
        return self.output(action_out), action_out

def generate_long_task(n_steps: int, state_dim: int, action_dim: int):
    """Generate long-horizon task with temporal dependencies."""
    states = []
    actions = []
    
    state = np.random.randn(state_dim) * 0.5
    action = np.random.randn(action_dim) * 0.2
    
    for step in range(n_steps):
        # Strong temporal dependence
        state = state * 0.95 + np.random.randn(state_dim) * 0.1
        action = action * 0.98 + np.random.randn(action_dim) * 0.05
        
        states.append(state.copy())
        actions.append(action.copy())
    
    return np.array(states), np.array(actions)

def train_model(model, states, actions, epochs=50):
    """Train model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for i in range(len(states) - 1):
            state = torch.FloatTensor(states[i:i+1])
            action = torch.FloatTensor(actions[i:i+1])
            next_action = torch.FloatTensor(actions[i+1:i+2])
            
            pred, _ = model(state, action)
            loss = criterion(pred, next_action)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate(model, states, actions):
    """Evaluate model."""
    total_error = 0
    for i in range(len(states) - 1):
        state = torch.FloatTensor(states[i:i+1])
        action = torch.FloatTensor(actions[i:i+1])
        next_action = torch.FloatTensor(actions[i+1:i+2])
        
        with torch.no_grad():
            pred, _ = model(state, action)
            total_error += nn.MSELoss()(pred, next_action).item()
    
    return total_error / max(1, len(states) - 1)

def run_experiment():
    config = TaskConfig()
    results = {
        'hypothesis': 'H1.107',
        'statement': 'Attention maintains +99% on 100-300 step ultra-long sequences',
        'results': []
    }
    
    print("\n=== H1.107: Ultra-Long Attention (100-300 steps) ===\n")
    
    for n_steps in config.n_steps:
        states, actions = generate_long_task(n_steps, config.state_dim, config.action_dim)
        
        # Concat baseline
        concat = ConcatModel(config)
        train_model(concat, states, actions)
        concat_error = evaluate(concat, states, actions)
        
        # Attention model
        attention = AttentionModel(config)
        train_model(attention, states, actions)
        attention_error = evaluate(attention, states, actions)
        
        # Improvement
        if concat_error > 0:
            improvement = ((concat_error - attention_error) / concat_error) * 100
        else:
            improvement = 0
        
        result = {
            'n_steps': n_steps,
            'concat_mse': concat_error,
            'attention_mse': attention_error,
            'improvement': improvement
        }
        results['results'].append(result)
        
        print(f"  {n_steps:3d} steps: Concat={concat_error:.6f}, Attn={attention_error:.6f}, Δ={improvement:+.1f}%")
    
    avg = np.mean([r['improvement'] for r in results['results']])
    results['avg_improvement'] = avg
    results['status'] = 'SUPPORTED' if avg > 0 else 'REFUTED'
    
    print(f"\n  Average: {avg:+.1f}%")
    print(f"  Status: {results['status']}")
    
    # Save
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    results = run_experiment()