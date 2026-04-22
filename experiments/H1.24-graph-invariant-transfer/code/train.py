#!/usr/bin/env python3
"""
H1.24: Graph + Invariant Learning Combined for Cross-Dynamics Transfer

Hypothesis: Combining graph structure (H2.x temporal benefits) with invariant 
learning (H1.8 transfer benefits) might achieve BOTH temporal reasoning AND 
cross-dynamics transfer simultaneously.

Previous results:
- H1.8 (Invariant): +5.4% on transfer
- H2.3-6 (Graph): +56-75% on temporal reasoning
- H1.4 (Unified transfer): -56.7% (FAILS)

This tests if combining both approaches can solve both problems.
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

class InvariantGraphNetwork(nn.Module):
    """Combined invariant learning + graph structure."""
    
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128, message_passes=2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.message_passes = message_passes
        
        # Physical encoder (for invariant learning)
        self.physics_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph message passing layers
        self.graph_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(message_passes)
        ])
        
        # Invariant projector (bisimulation-inspired)
        self.invariant_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Dynamics predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, states, actions, adj_matrix=None):
        # Encode states
        h = self.state_encoder(states)
        
        # Graph message passing
        for i in range(self.message_passes):
            h_neighbors = torch.matmul(adj_matrix, h)
            messages = torch.cat([h, h_neighbors], dim=-1)
            h = torch.relu(self.graph_layers[i](messages))
        
        # Encode physics action
        physics_input = torch.cat([states, actions], dim=-1)
        physics_h = self.physics_encoder(physics_input)
        
        # Invariant projection (maximize similarity for similar outcomes)
        invariant_z = self.invariant_projector(physics_h)
        
        # Combine and predict next state
        combined = torch.cat([h, invariant_z], dim=-1)
        next_state_pred = self.dynamics_predictor(combined)
        
        return next_state_pred


def generate_task_data(n_samples=200, n_objects=3, dynamics_variation=0.3):
    """Generate task data with different dynamics."""
    
    friction = np.random.uniform(0.1, 0.5)
    mass = np.random.uniform(0.5, 1.5)
    
    states = []
    actions = []
    next_states = []
    
    for _ in range(n_samples):
        state = np.random.randn(n_objects, 8) * 0.5
        action = np.random.randn(n_objects, 4) * 0.3
        
        # Physics simulation with dynamics
        next_state = state.copy()
        for obj in range(n_objects):
            force = action[obj] / mass
            damping = friction * state[obj, :4]
            next_state[obj, :4] = state[obj, :4] + force - damping * dynamics_variation
            next_state[obj, 4:] = state[obj, 4:] + np.random.randn(4) * 0.1
        
        # Add noise
        state += np.random.randn(*state.shape) * 0.05
        next_state += np.random.randn(*next_state.shape) * 0.05
        
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
    
    return np.array(states), np.array(actions), np.array(next_states)


def train_invariant_graph():
    """Train and evaluate combined invariant + graph architecture."""
    
    results = {
        "hypothesis": "H1.24",
        "statement": "Graph + Invariant Learning for Cross-Dynamics Transfer",
        "timestamp": datetime.now().isoformat(),
        "source_domain": [],
        "target_domain": [],
        "baseline_transfer": [],
        "invariant_graph_transfer": []
    }
    
    # Test across multiple dynamics configurations
    dynamics_configs = [
        {"friction": 0.2, "mass": 1.0, "name": "baseline"},
        {"friction": 0.4, "mass": 1.5, "name": "high_friction_mass"},
        {"friction": 0.1, "mass": 0.6, "name": "low_friction_mass"},
        {"friction": 0.35, "mass": 1.2, "name": "mid_friction_mass"},
    ]
    
    for source_idx, source_config in enumerate(dynamics_configs[:2]):
        for target_idx, target_config in enumerate(dynamics_configs[2:], start=2):
            print(f"\n=== Source: {source_config['name']} -> Target: {target_config['name']} ===")
            
            # Generate source domain data
            source_states, source_actions, source_next = generate_task_data(
                n_samples=300, dynamics_variation=source_config["friction"]
            )
            
            # Generate target domain data
            target_states, target_actions, target_next = generate_task_data(
                n_samples=300, dynamics_variation=target_config["friction"]
            )
            
            # Baseline
            baseline_model = InvariantGraphNetwork(message_passes=0).to(DEVICE)
            
            # Use invariant + graph
            ig_model = InvariantGraphNetwork(message_passes=2).to(DEVICE)
            
            # Simple baseline (no invariant, no graph)
            class SimpleBaseline(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(8 + 4, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 8)
                    )
                def forward(self, s, a, _=None):
                    return self.net(torch.cat([s, a], dim=-1))
            
            baseline = SimpleBaseline().to(DEVICE)
            
            # Prepare data
            source_dataset = TensorDataset(
                torch.FloatTensor(source_states),
                torch.FloatTensor(source_actions),
                torch.FloatTensor(source_next)
            )
            target_dataset = TensorDataset(
                torch.FloatTensor(target_states),
                torch.FloatTensor(target_actions),
                torch.FloatTensor(target_next)
            )
            
            # Train baseline on source
            opt = optim.Adam(baseline.parameters(), lr=0.001)
            for epoch in range(200):
                for s, a, ns in source_dataset:
                    s, a, ns = s.to(DEVICE), a.to(DEVICE), ns.to(DEVICE)
                    opt.zero_grad()
                    pred = baseline(s, a)
                    loss = nn.MSELoss()(pred, ns)
                    loss.backward()
                    opt.step()
            
            # Train IG model on source
            opt_ig = optim.Adam(ig_model.parameters(), lr=0.001)
            adj = torch.ones(3, 3) * 0.5  # Fully connected graph
            for epoch in range(200):
                for s, a, ns in source_dataset:
                    s, a, ns = s.to(DEVICE), a.to(DEVICE), ns.to(DEVICE)
                    opt_ig.zero_grad()
                    pred = ig_model(s, a, adj.to(DEVICE))
                    loss = nn.MSELoss()(pred, ns)
                    loss.backward()
                    opt_ig.step()
            
            # Evaluate on target (zero-shot transfer)
            baseline.eval()
            ig_model.eval()
            
            baseline_errors = []
            ig_errors = []
            
            with torch.no_grad():
                for s, a, ns in target_dataset:
                    s, a, ns = s.to(DEVICE), a.to(DEVICE), ns.to(DEVICE)
                    pred_base = baseline(s, a)
                    pred_ig = ig_model(s, a, adj.to(DEVICE))
                    
                    baseline_errors.append(nn.MSELoss()(pred_base, ns).item())
                    ig_errors.append(nn.MSELoss()(pred_ig, ns).item())
            
            baseline_mse = np.mean(baseline_errors)
            ig_mse = np.mean(ig_errors)
            
            improvement = (baseline_mse - ig_mse) / baseline_mse * 100
            
            print(f"Baseline: {baseline_mse:.4f}, Invariant+Graph: {ig_mse:.4f}, Improvement: {improvement:.1f}%")
            
            results["baseline_transfer"].append(baseline_mse)
            results["invariant_graph_transfer"].append(ig_mse)
            results["source_domain"].append(source_config["name"])
            results["target_domain"].append(target_config["name"])
    
    avg_baseline = np.mean(results["baseline_transfer"])
    avg_ig = np.mean(results["invariant_graph_transfer"])
    avg_improvement = (avg_baseline - avg_ig) / avg_baseline * 100
    
    results["average_baseline"] = avg_baseline
    results["average_ig"] = avg_ig
    results["average_improvement"] = avg_improvement
    
    print(f"\n=== FINAL: Baseline {avg_baseline:.4f} vs IG {avg_ig:.4f} ===")
    print(f"Average improvement: {avg_improvement:.1f}%")
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = train_invariant_graph()
    
    status = "SUPPORTED" if results["average_improvement"] > 0 else "REFUTED"
    print(f"\nH1.24: {status} ({results['average_improvement']:+.1f}%)")