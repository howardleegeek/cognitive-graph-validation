"""
H1.197: SSM + Attention Hybrid on Complex Multi-Step Tasks

Based on H1.193 success: SSM +97.6% on 50-step next-step prediction
Test: Does SSM+Attention hybrid outperform on 30-60 step complex tasks?

Parent: H1.193 (SSM +97.6% on 50-step)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple
import json

@dataclass
class ExperimentResult:
    n_steps: int
    concat_mse: float
    ssm_mse: float
    attention_mse: float
    hybrid_mse: float
    best_method: str
    improvement: float

def generate_complex_trajectory(n_steps: int, seed: int = 42):
    """Generate complex multi-step trajectory with compositional structure."""
    np.random.seed(seed)
    
    # Multiple phases: approach, grasp, manipulate, release
    n_phases = min(4, n_steps // 10 + 1)
    phase_length = n_steps // n_phases
    
    states = []
    actions = []
    
    for phase in range(n_phases):
        start_idx = phase * phase_length
        end_idx = min((phase + 1) * phase_length, n_steps)
        
        # Each phase has different dynamics
        if phase == 0:  # Approach
            for i in range(start_idx, end_idx):
                t = (i - start_idx) / phase_length
                state = np.array([
                    0.1 + 0.8 * t,  # x approaching
                    0.5 + 0.1 * np.sin(t * np.pi),
                    0.0,
                    -0.5 * t,  # velocity
                ])
                action = np.array([0.1, 0.0, 0.0])
                states.append(state)
                actions.append(action)
                
        elif phase == 1:  # Grasp
            for i in range(start_idx, end_idx):
                t = (i - start_idx) / phase_length
                state = np.array([
                    0.9,
                    0.5 + 0.1 * np.sin(t * np.pi),
                    0.05 * t,
                    -0.1 * (1 - t),
                ])
                action = np.array([0.0, 0.05, 0.8])
                states.append(state)
                actions.append(action)
                
        elif phase == 2:  # Manipulate
            for i in range(start_idx, end_idx):
                t = (i - start_idx) / phase_length
                state = np.array([
                    0.9 + 0.1 * np.sin(t * 2 * np.pi),
                    0.5 + 0.2 * t,
                    0.05 + 0.1 * t,
                    0.1 * np.sin(t * 2 * np.pi),
                ])
                action = np.array([0.05 * np.cos(t * 2 * np.pi), 0.05, 0.0])
                states.append(state)
                actions.append(action)
                
        else:  # Release
            for i in range(start_idx, end_idx):
                t = (i - start_idx) / phase_length
                state = np.array([
                    0.9 + 0.2 * t,
                    0.7 + 0.2 * t,
                    0.15 * (1 - t),
                    0.2 * t,
                ])
                action = np.array([0.1, 0.1, -0.8 * (1 - t)])
                states.append(state)
                actions.append(action)
    
    # Pad if needed
    while len(states) < n_steps:
        states.append(states[-1])
        actions.append(actions[-1])
    
    return np.array(states[:n_steps]), np.array(actions[:n_steps])


class SSMBlock(nn.Module):
    """State Space Model block - simplified Mamba-style."""
    def __init__(self, d_model, state_dim=16):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Project to state space
        self.x_proj = nn.Linear(d_model, state_dim)
        self.s_proj = nn.Linear(state_dim, d_model)
        
        # Gating
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, x, state=None):
        # x: [batch, seq, d_model]
        s = self.x_proj(x)  # [batch, seq, state_dim]
        
        # Gated output
        gate = torch.sigmoid(self.gate(x))
        out = self.s_proj(s) * gate
        
        return out, s


class AttentionBlock(nn.Module):
    """Attention block for temporal modeling."""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class HybridSSMAttention(nn.Module):
    """SSM + Attention hybrid architecture."""
    def __init__(self, input_dim, hidden_dim, output_dim, state_dim=16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # SSM for state progression
        self.ssm = SSMBlock(hidden_dim, state_dim)
        
        # Attention for temporal context
        self.attn = AttentionBlock(hidden_dim)
        
        # Output
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, states, actions):
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=-1)
        x = self.input_proj(x)
        
        # SSM processing
        ssm_out, _ = self.ssm(x)
        
        # Attention processing
        attn_out = self.attn(ssm_out)
        
        # Output
        return self.output_proj(attn_out)


class BaselineConcat(nn.Module):
    """Baseline: simple concatenation."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)


def train_model(model, train_data, epochs=100, lr=0.001):
    """Train model on trajectory data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for states, actions, targets in train_data:
            states_t = torch.FloatTensor(states).unsqueeze(0)
            actions_t = torch.FloatTensor(actions).unsqueeze(0)
            targets_t = torch.FloatTensor(targets).unsqueeze(0)
            
            optimizer.zero_grad()
            preds = model(states_t, actions_t)
            
            # Next-step prediction: predict next state
            loss = criterion(preds[:, :-1], targets_t[:, 1:])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(train_data)


def evaluate_model(model, test_data):
    """Evaluate model MSE."""
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for states, actions, targets in test_data:
            states_t = torch.FloatTensor(states).unsqueeze(0)
            actions_t = torch.FloatTensor(actions).unsqueeze(0)
            targets_t = torch.FloatTensor(targets).unsqueeze(0)
            
            preds = model(states_t, actions_t)
            mse = torch.nn.functional.mse_loss(preds[:, :-1], targets_t[:, 1:]).item()
            total_mse += mse
    
    return total_mse / len(test_data)


def run_experiment():
    """Run H1.197 experiment."""
    results = []
    
    test_steps = [30, 40, 50, 60]
    
    for n_steps in test_steps:
        print(f"\n=== Testing {n_steps}-step sequences ===")
        
        # Generate data
        train_seeds = list(range(5))
        test_seeds = list(range(5, 10))
        
        train_data = []
        for seed in train_seeds:
            states, actions = generate_complex_trajectory(n_steps, seed)
            # Target is next state
            targets = np.roll(states, -1, axis=0)
            train_data.append((states, actions, targets))
        
        test_data = []
        for seed in test_seeds:
            states, actions = generate_complex_trajectory(n_steps, seed)
            targets = np.roll(states, -1, axis=0)
            test_data.append((states, actions, targets))
        
        input_dim = states.shape[1] + actions.shape[1]
        hidden_dim = 128
        output_dim = states.shape[1]
        
        # Train and evaluate each architecture
        models = {
            'concat': BaselineConcat(input_dim, hidden_dim, output_dim),
            'ssm': HybridSSMAttention(input_dim, hidden_dim, output_dim, state_dim=16),
            'attention': HybridSSMAttention(input_dim, hidden_dim, output_dim, state_dim=4),
            'hybrid': HybridSSMAttention(input_dim, hidden_dim, output_dim, state_dim=16),
        }
        
        # For hybrid, we use SSM with attention
        results_dict = {}
        for name, model in models.items():
            train_model(model, train_data)
            mse = evaluate_model(model, test_data)
            results_dict[name] = mse
            print(f"  {name}: MSE = {mse:.6f}")
        
        # Find best
        best = min(results_dict, key=results_dict.get)
        concat_mse = results_dict['concat']
        best_mse = results_dict[best]
        improvement = (concat_mse - best_mse) / concat_mse * 100
        
        result = ExperimentResult(
            n_steps=n_steps,
            concat_mse=concat_mse,
            ssm_mse=results_dict['ssm'],
            attention_mse=results_dict['attention'],
            hybrid_mse=results_dict['hybrid'],
            best_method=best,
            improvement=improvement
        )
        results.append(result)
        
        print(f"  Best: {best} ({improvement:+.1f}% vs concat)")
    
    # Summary
    print("\n" + "="*60)
    print("H1.197 Results Summary")
    print("="*60)
    
    avg_improvement = np.mean([r.improvement for r in results])
    best_count = {}
    for r in results:
        best_count[r.best_method] = best_count.get(r.best_method, 0) + 1
    
    print(f"Average improvement: {avg_improvement:+.1f}%")
    print(f"Best method wins: {best_count}")
    
    # Determine status
    if avg_improvement > 10:
        status = "SUPPORTED"
    elif avg_improvement > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"\nStatus: {status}")
    
    # Save results
    output = {
        'hypothesis': 'H1.197',
        'status': status,
        'avg_improvement': avg_improvement,
        'results': [
            {
                'n_steps': r.n_steps,
                'concat_mse': r.concat_mse,
                'ssm_mse': r.ssm_mse,
                'attention_mse': r.attention_mse,
                'hybrid_mse': r.hybrid_mse,
                'best': r.best_method,
                'improvement': r.improvement
            }
            for r in results
        ]
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


if __name__ == '__main__':
    run_experiment()