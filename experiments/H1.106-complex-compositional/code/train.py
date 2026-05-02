"""
H1.106: Complex Compositional Multi-Step Tasks (20-50 steps)
===========================================================
Test unified architecture on extremely complex tasks requiring 
compositional reasoning over long horizons.

Based on H1.33: +86.8% avg on 20-40 step tasks (SUPPORTED)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple
import json

np.random.seed(42)
torch.manual_seed(42)

@dataclass
class TaskConfig:
    state_dim: int = 14
    action_dim: int = 7
    hidden_dim: int = 512
    n_steps: List[int] = None
    
    def __post_init__(self):
        if self.n_steps is None:
            self.n_steps = [20, 25, 30, 35, 40, 45, 50]

class UnifiedModel(nn.Module):
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.state_action_dim = config.state_dim + config.action_dim
        
        # Unified 512-dim representation
        self.physical_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim)
        )
        
    def forward(self, state, action, hidden=None):
        physical = self.physical_encoder(state)
        semantic = self.semantic_encoder(action)
        combined = torch.cat([physical, semantic], dim=-1)
        return self.fusion(combined), combined

class BaselineModel(nn.Module):
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

def generate_compositional_task(n_steps: int, state_dim: int, action_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate complex compositional task with multiple phases."""
    # Create multi-phase trajectory
    n_phases = 3
    steps_per_phase = n_steps // n_phases
    
    states = []
    actions = []
    
    for phase in range(n_phases):
        # Each phase has different dynamics
        base_state = np.random.randn(state_dim) * 0.5
        for step in range(steps_per_phase):
            # Compositional: phase-specific transformation
            phase_offset = phase * np.random.randn(state_dim) * 0.3
            state = base_state + phase_offset + np.random.randn(state_dim) * 0.1
            states.append(state)
            
            # Action also compositional
            action = np.random.randn(action_dim) * 0.2 + phase * 0.1
            actions.append(action)
    
    return np.array(states), np.array(actions)

def train_model(model, states, actions, epochs=100):
    """Train model on trajectory."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(states) - 1):
            state = torch.FloatTensor(states[i:i+1])
            action = torch.FloatTensor(actions[i:i+1])
            next_action = torch.FloatTensor(actions[i+1:i+2])
            
            pred, _ = model(state, action)
            loss = nn.MSELoss()(pred, next_action)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(total_loss / max(1, len(states) - 1))
    
    return losses

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
        'hypothesis': 'H1.106',
        'statement': 'Unified maintains advantage on 20-50 step complex compositional tasks',
        'results': []
    }
    
    print("\n=== H1.106: Complex Compositional 20-50 Steps ===\n")
    
    for n_steps in config.n_steps:
        # Generate task
        states, actions = generate_compositional_task(n_steps, config.state_dim, config.action_dim)
        
        # Train baseline
        baseline = BaselineModel(config)
        baseline_losses = train_model(baseline, states, actions)
        baseline_error = evaluate(baseline, states, actions)
        
        # Train unified
        unified = UnifiedModel(config)
        unified_losses = train_model(unified, states, actions)
        unified_error = evaluate(unified, states, actions)
        
        # Calculate improvement
        if baseline_error > 0:
            improvement = ((baseline_error - unified_error) / baseline_error) * 100
        else:
            improvement = 0
        
        result = {
            'n_steps': n_steps,
            'baseline_mse': baseline_error,
            'unified_mse': unified_error,
            'improvement': improvement
        }
        results['results'].append(result)
        
        print(f"  {n_steps:2d} steps: Baseline={baseline_error:.4f}, Unified={unified_error:.4f}, Δ={improvement:+.1f}%")
    
    # Calculate average
    avg_improvement = np.mean([r['improvement'] for r in results['results']])
    results['avg_improvement'] = avg_improvement
    results['status'] = 'SUPPORTED' if avg_improvement > 0 else 'REFUTED'
    
    print(f"\n  Average: {avg_improvement:+.1f}%")
    print(f"  Status: {results['status']}")
    
    # Save results
    with open('experiments/H1.106-complex-compositional/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    results = run_experiment()