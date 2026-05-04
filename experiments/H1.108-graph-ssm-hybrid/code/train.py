#!/usr/bin/env python3
"""
H1.108: Graph + SSM Hybrid for Complex Temporal Tasks
=======================================================
Test combining graph structure with SSM for complex temporal reasoning.
Based on:
- H3.17: Graph+SSM +25% on temporal tasks (SUPPORTED)
- H1.102: Unified+SSM +29.8% on multi-step (SUPPORTED)
Goal: Test if combining both achieves even better results on complex tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import json

np.random.seed(42)
torch.manual_seed(42)


class GraphSSMModel(nn.Module):
    """Graph + SSM combined for temporal reasoning"""
    def __init__(self, obs_dim=32, hidden_dim=128, n_nodes=4, ssm_state=8):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = obs_dim // n_nodes
        self.ssm_state = ssm_state
        self.hidden_per_node = hidden_dim // n_nodes
        
        # Node encoders
        self.node_encoders = nn.ModuleList([
            nn.Linear(self.node_dim, self.hidden_per_node) for _ in range(n_nodes)
        ])
        
        # Graph message passing (per node)
        self.graph_layers = nn.ModuleList([
            nn.Linear(self.hidden_per_node, self.hidden_per_node) for _ in range(3)
        ])
        
        # SSM processing
        self.ssm_A = nn.Parameter(torch.randn(ssm_state, ssm_state) * 0.01)
        self.ssm_proj = nn.Linear(hidden_dim, ssm_state)
        self.ssm_out = nn.Linear(ssm_state, hidden_dim)
        
        # Output
        self.output = nn.Linear(hidden_dim, obs_dim)
    
    def forward(self, x):
        # x: (B, T, obs_dim)
        B, T, _ = x.shape
        
        # Reshape to nodes
        x_nodes = x.view(B, T, self.n_nodes, self.node_dim)
        
        # Process each node
        node_h = []
        for i in range(self.n_nodes):
            h = self.node_encoders[i](x_nodes[:, :, i])
            node_h.append(h)
        
        # Graph message passing
        for layer in self.graph_layers:
            new_node_h = []
            for i in range(self.n_nodes):
                # Aggregate messages from other nodes
                msg = sum(node_h[j] for j in range(self.n_nodes) if j != i)
                combined = node_h[i] + msg
                new_node_h.append(torch.relu(layer(combined)))
            node_h = new_node_h
        
        # Concatenate node representations
        h = torch.cat(node_h, dim=-1)  # (B, T, hidden_dim)
        
        # SSM temporal processing
        ssm_h = torch.zeros(B, self.ssm_state, device=x.device)
        for t in range(T):
            inp = self.ssm_proj(h[:, t])
            ssm_h = torch.matmul(ssm_h, self.ssm_A.t()) + inp
        
        ssm_out = self.ssm_out(ssm_h)
        
        return self.output(ssm_out)


class BaselineModel(nn.Module):
    """Simple MLP baseline"""
    def __init__(self, obs_dim=32, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def forward(self, x):
        return self.net(x[:, -1, :])


def generate_complex_temporal_data(n_samples, seq_len, obs_dim, n_steps):
    """Generate complex multi-step temporal data"""
    X, Y = [], []
    
    for _ in range(n_samples):
        # Initial state
        state = torch.randn(seq_len, obs_dim) * 0.1
        
        # Multi-step transitions with temporal dependencies
        for step in range(n_steps):
            # Action affects state with delay
            action = torch.randn(seq_len, 4) * 0.1
            transition = state * 0.4 + action[:, :1] * 0.2
            
            # Add temporal dependency
            if step > 0:
                transition = transition + state[:, :1] * 0.1
            
            state = state + transition + torch.randn(seq_len, obs_dim) * 0.02
        
        # Target is final state
        y = state[-1] + torch.randn(obs_dim) * 0.05
        X.append(state)
        Y.append(y)
    
    return torch.stack(X), torch.stack(Y)


def main():
    print("=" * 60)
    print("H1.108: Graph + SSM Hybrid for Complex Temporal Tasks")
    print("=" * 60)
    
    results = {}
    obs_dim = 32
    n_samples = 150
    
    for n_steps in [5, 8, 12, 15]:
        print(f"\n--- {n_steps} steps ---")
        
        # Generate data
        X, Y = generate_complex_temporal_data(n_samples, 10, obs_dim, n_steps)
        
        # Split
        split = int(0.8 * n_samples)
        X_train, Y_train = X[:split], Y[:split]
        X_test, Y_test = X[split:], Y[split:]
        
        # Baseline
        torch.manual_seed(42)
        baseline = BaselineModel(obs_dim, 128)
        opt = torch.optim.Adam(baseline.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(100):
            pred = baseline(X_train)
            loss = criterion(pred, Y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        baseline.eval()
        with torch.no_grad():
            baseline_mse = criterion(baseline(X_test), Y_test).item()
        
        # Graph+SSM
        torch.manual_seed(42)
        graph_ssm = GraphSSMModel(obs_dim, 128, 4, 8)
        opt = torch.optim.Adam(graph_ssm.parameters(), lr=0.01)
        
        for epoch in range(100):
            pred = graph_ssm(X_train)
            loss = criterion(pred, Y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        graph_ssm.eval()
        with torch.no_grad():
            graph_ssm_mse = criterion(graph_ssm(X_test), Y_test).item()
        
        improvement = (baseline_mse - graph_ssm_mse) / baseline_mse * 100
        
        print(f"  Baseline: {baseline_mse:.4f}, Graph+SSM: {graph_ssm_mse:.4f}, Δ={improvement:+.1f}%")
        results[n_steps] = {
            "baseline_mse": baseline_mse,
            "graph_ssm_mse": graph_ssm_mse,
            "improvement": improvement
        }
    
    # Summary
    avg_improvement = np.mean([r["improvement"] for r in results.values()])
    status = "SUPPORTED" if avg_improvement > 10 else "MARGINAL" if avg_improvement > 0 else "REFUTED"
    
    print(f"\n=== Summary ===")
    print(f"Average Improvement: {avg_improvement:+.1f}%")
    print(f"Status: {status}")
    
    # Save
    with open("results.json", "w") as f:
        json.dump({"results": results, "avg_improvement": avg_improvement, "status": status}, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()