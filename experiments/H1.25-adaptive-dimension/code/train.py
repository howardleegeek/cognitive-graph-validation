#!/usr/bin/env python3
"""
H1.25: Adaptive Dimension Allocation During Inference

Hypothesis: The optimal dimension allocation (physical/semantic ratio) may
vary by task complexity. Simple tasks may need different ratios than complex tasks.

Previous results:
- H4: 22% physical is optimal (static)
- H1.11-14: Larger dimensions = better
- H1.15-17: Graph helps complex tasks

This tests if we can dynamically adjust dimension allocation based on task complexity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveDimensionNetwork(nn.Module):
    """Network that adapts dimension allocation based on task complexity."""
    
    def __init__(self, state_dim=8, action_dim=4, max_hidden=512, n_objects=3):
        super().__init__()
        self.max_hidden = max_hidden
        self.n_objects = n_objects
        
        # Multiple branches with different allocations
        # 12%, 22%, 32%, 42% physical
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, int(max_hidden * physical_frac)),
                nn.ReLU(),
                nn.Linear(int(max_hidden * physical_frac), int(max_hidden * physical_frac)),
                nn.ReLU(),
                nn.Linear(int(max_hidden * physical_frac), state_dim)
            )
            for physical_frac in [0.12, 0.22, 0.32, 0.42]
        ])
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, states, actions, branch_weights=None):
        # Estimate task complexity
        complexity = self.complexity_estimator(states.mean(dim=1, keepdim=True))
        
        # Get predictions from all branches
        predictions = []
        for branch in self.branches:
            pred = branch(torch.cat([states, actions], dim=-1))
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [4, batch, n_objects, state_dim]
        
        if branch_weights is None:
            # Use complexity to weight branches
            # Low complexity -> use low physical (12%)
            # High complexity -> use high physical (42%)
            weights = torch.softmax(
                torch.tensor([0.2, 0.3, 0.5, 1.0]).to(DEVICE) * complexity.squeeze(), 
                dim=0
            )
        else:
            weights = branch_weights.to(DEVICE)
        
        # Weighted combination
        weighted_pred = (predictions * weights.view(4, 1, 1, 1)).sum(dim=0)
        
        return weighted_pred


def generate_task_data(n_samples=200, complexity=0.5):
    """Generate task data with varying complexity."""
    
    n_timesteps = int(5 + complexity * 15)  # 5-20 timesteps based on complexity
    
    states = []
    actions = []
    next_states = []
    
    for _ in range(n_samples):
        # Initial state
        state = np.random.randn(3, 8) * 0.3
        action = np.random.randn(3, 4) * 0.2
        
        state_traj = [state.copy()]
        action_traj = []
        next_state_traj = []
        
        for t in range(n_timesteps):
            # Random action
            action = np.random.randn(3, 4) * complexity
            action_traj.append(action.copy())
            
            # Physics
            next_state = state.copy()
            for obj in range(3):
                next_state[obj, :4] = state[obj, :4] + action[obj] * 0.1 - state[obj, :4] * 0.05
            
            state_traj.append(state.copy())
            next_state_traj.append(next_state.copy())
            state = next_state
        
        states.append(np.array(state_traj[:-1]))
        actions.append(np.array(action_traj))
        next_states.append(np.array(next_state_traj))
    
    return np.array(states), np.array(actions), np.array(next_states)


def train_adaptive():
    """Train and evaluate adaptive dimension allocation."""
    
    results = {
        "hypothesis": "H1.25",
        "statement": "Adaptive Dimension Allocation During Inference",
        "timestamp": datetime.now().isoformat(),
        "fixed_results": [],
        "adaptive_results": [],
        "complexities": []
    }
    
    for complexity in [0.2, 0.5, 0.8]:
        print(f"\n=== Testing complexity={complexity} ===")
        
        # Generate data
        states, actions, next_states = generate_task_data(n_samples=300, complexity=complexity)
        
        # Model with 22% fixed (from H4)
        class Fixed22Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(8 + 4, 112),
                    nn.ReLU(),
                    nn.Linear(112, 112),
                    nn.ReLU(),
                    nn.Linear(112, 8)
                )
            def forward(self, s, a, _=None):
                return self.net(torch.cat([s, a], dim=-1))
        
        # Adaptive model
        adaptive = AdaptiveDimensionNetwork(max_hidden=512).to(DEVICE)
        
        # Train fixed model
        fixed = Fixed22Network().to(DEVICE)
        opt_fixed = optim.Adam(fixed.parameters(), lr=0.001)
        
        # Train adaptive model
        opt_adaptive = optim.Adam(adaptive.parameters(), lr=0.001)
        
        dataset = TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(next_states)
        )
        
        for epoch in range(150):
            for s, a, ns in dataset:
                s, a, ns = s.to(DEVICE), a.to(DEVICE), ns.to(DEVICE)
                
                # Flatten for simplicity
                s_flat = s.view(s.size(0), -1)
                a_flat = a.view(a.size(0), -1)
                ns_flat = ns.view(ns.size(0), -1)
                
                # Fixed
                opt_fixed.zero_grad()
                pred_fixed = fixed(s_flat[:, :8], a_flat[:, :4])
                loss_fixed = nn.MSELoss()(pred_fixed, ns_flat[:, :8])
                loss_fixed.backward()
                opt_fixed.step()
                
                # Adaptive (sample average - reshape to [batch, n_objects, dims])
                opt_adaptive.zero_grad()
                # Simplified: just use first timestep for complexity estimation
                pred_adaptive = adaptive(s[:, 0], a[:, 0])
                loss_adaptive = nn.MSELoss()(pred_adaptive, ns[:, 0])
                loss_adaptive.backward()
                opt_adaptive.step()
        
        # Evaluate
        fixed.eval()
        adaptive.eval()
        
        fixed_errors = []
        adaptive_errors = []
        
        with torch.no_grad():
            for s, a, ns in dataset:
                s, a, ns = s.to(DEVICE), a.to(DEVICE), ns.to(DEVICE)
                s_flat = s.view(s.size(0), -1)
                a_flat = a.view(a.size(0), -1)
                ns_flat = ns.view(ns.size(0), -1)
                
                pred_fixed = fixed(s_flat[:, :8], a_flat[:, :4])
                pred_adaptive = adaptive(s[:, 0], a[:, 0])
                
                fixed_errors.append(nn.MSELoss()(pred_fixed, ns_flat[:, :8]).item())
                adaptive_errors.append(nn.MSELoss()(pred_adaptive, ns[:, 0]).item())
        
        fixed_mse = np.mean(fixed_errors)
        adaptive_mse = np.mean(adaptive_errors)
        improvement = (fixed_mse - adaptive_mse) / fixed_mse * 100
        
        print(f"Fixed (22%): {fixed_mse:.4f}, Adaptive: {adaptive_mse:.4f}, Delta: {improvement:+.1f}%")
        
        results["fixed_results"].append(fixed_mse)
        results["adaptive_results"].append(adaptive_mse)
        results["complexities"].append(complexity)
    
    avg_fixed = np.mean(results["fixed_results"])
    avg_adaptive = np.mean(results["adaptive_results"])
    avg_improvement = (avg_fixed - avg_adaptive) / avg_fixed * 100
    
    results["average_fixed"] = avg_fixed
    results["average_adaptive"] = avg_adaptive
    results["average_improvement"] = avg_improvement
    
    print(f"\n=== FINAL: Fixed {avg_fixed:.4f} vs Adaptive {avg_adaptive:.4f} ===")
    print(f"Average improvement: {avg_improvement:+.1f}%")
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = train_adaptive()
    
    status = "SUPPORTED" if results["average_improvement"] > 0 else "REFUTED"
    print(f"\nH1.25: {status} ({results['average_improvement']:+.1f}%)")