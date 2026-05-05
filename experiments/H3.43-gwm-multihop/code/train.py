#!/usr/bin/env python3
"""H3.43: Multi-hop Message Passing (GWM-style)"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)

def generate_data(n_samples, n_timesteps, seed=42):
    np.random.seed(seed)
    state_dim = 64
    action_dim = 16
    states = np.random.randn(n_samples, n_timesteps, state_dim).astype(np.float32)
    actions = np.random.randn(n_samples, n_timesteps, action_dim).astype(np.float32)
    targets = np.tanh(states @ np.random.randn(state_dim, state_dim).T * 0.1)
    targets += 0.1 * np.roll(targets, 1, axis=1)
    targets = targets.astype(np.float32)
    return torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(targets)

class GraphLayer(nn.Module):
    def __init__(self, dim, n_hops):
        super().__init__()
        self.n_hops = n_hops
        self.edge_mlp = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, dim))
        
    def forward(self, x):
        h = x
        for _ in range(self.n_hops):
            h_shifted = torch.roll(h, 1, dims=1)
            h_shifted[:, 0, :] = 0
            combined = torch.cat([h, h_shifted], dim=-1)
            msg = self.edge_mlp(combined)
            h = h + 0.1 * msg
        return h

class Model(nn.Module):
    def __init__(self, dim=256, n_hops=2, use_graph=True):
        super().__init__()
        self.encoder = nn.Linear(80, dim)
        self.graph = GraphLayer(dim, n_hops) if use_graph else nn.Identity()
        self.decoder = nn.Linear(dim, 64)
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        x = self.graph(x)
        return self.decoder(x)

def run_experiment(n_hops, n_timesteps=20, epochs=100):
    train_s, train_a, train_t = generate_data(50, n_timesteps)
    val_s, val_a, val_t = generate_data(20, n_timesteps+1, seed=43)
    
    baseline = Model(n_hops=0, use_graph=False)
    graph = Model(n_hops=n_hops, use_graph=True)
    
    opt_b = torch.optim.Adam(baseline.parameters(), lr=0.001)
    opt_g = torch.optim.Adam(graph.parameters(), lr=0.001)
    crit = nn.MSELoss()
    
    for _ in range(epochs):
        opt_b.zero_grad()
        loss = crit(baseline(train_s, train_a), train_t)
        loss.backward()
        opt_b.step()
        
        opt_g.zero_grad()
        loss = crit(graph(train_s, train_a), train_t)
        loss.backward()
        opt_g.step()
    
    baseline.eval()
    graph.eval()
    
    with torch.no_grad():
        base_mse = crit(baseline(val_s, val_a), val_t).item()
        graph_mse = crit(graph(val_s, val_a), val_t).item()
    
    return base_mse, graph_mse

def main():
    print("H3.43: Multi-hop Message Passing (GWM-style)")
    print("=" * 50)
    
    results = {}
    for n_hops in [1, 2, 3]:
        base_mse, graph_mse = run_experiment(n_hops)
        results[n_hops] = (base_mse, graph_mse)
        imp = (base_mse - graph_mse) / base_mse * 100 if base_mse > 0 else 0
        print(f"  {n_hops} hops: baseline={base_mse:.4f}, graph={graph_mse:.4f}, improvement={imp:+.1f}%")
    
    avg_imp = sum((r[0] - r[1]) / r[0] * 100 for r in results.values() if r[0] > 0) / len(results)
    print(f"\nAverage improvement: {avg_imp:+.1f}%")
    print(f"Status: {'SUPPORTED' if avg_imp > 0 else 'REFUTED' if avg_imp < -10 else 'INCONCLUSIVE'}")
    return avg_imp

if __name__ == "__main__":
    main()