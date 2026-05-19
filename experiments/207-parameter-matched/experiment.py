#!/usr/bin/env python3
"""
H1.441: Parameter-Matched Architecture Comparison
==================================================
Hypothesis: GraphCG's diminishing advantage with complexity is due to 
parameter count mismatch. Test with adaptive node count.
MINIMAL VERSION - for quick iteration.
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class GraphCG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=4, n_nodes=6, n_layers=1):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_embed = nn.Parameter(torch.randn(n_nodes, hidden_dim) * 0.1)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.graph_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.graph_layers.append(nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True))
            self.graph_layers.append(nn.LayerNorm(hidden_dim))
            self.graph_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        h = self.input_proj(x)
        node_feats = self.node_embed.unsqueeze(0).expand(batch_size, -1, -1).clone()
        input_injection = h.unsqueeze(1).expand(-1, self.n_nodes, -1)
        node_feats = node_feats + input_injection * 0.1
        
        for i in range(0, len(self.graph_layers), 3):
            attn_out, _ = self.graph_layers[i](node_feats, node_feats, node_feats)
            node_feats = self.graph_layers[i+1](attn_out + node_feats)
            ff_out = self.graph_layers[i+2](node_feats)
            node_feats = self.graph_layers[i+1](ff_out + node_feats)
        
        h = node_feats.mean(dim=1)
        return self.output_proj(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_data(n_samples, n_objects, n_steps, seed=42):
    set_seed(seed)
    data = []
    for _ in range(n_samples):
        positions = np.random.randn(n_objects, 2) * 2.0
        for _ in range(n_steps):
            dx, dy = np.random.randn() * 0.5, np.random.randn() * 0.5
            angle = np.random.randn() * 0.3
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            new_positions = positions.copy()
            for i in range(n_objects):
                x, y = positions[i]
                new_positions[i] = [x * cos_a - y * sin_a + dx, x * sin_a + y * cos_a + dy]
            positions = new_positions
        data.append((positions.flatten(), positions[0].flatten()))
    return data


def train_and_eval(mlp, gc, train_data, val_data, test_data, epochs=30):
    """Train both models and evaluate."""
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_t = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in train_data]
    val_t = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in val_data]
    test_t = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in test_data]
    
    # Train MLP
    for epoch in range(epochs):
        mlp.train()
        for x_t, y_t in train_t:
            optimizer.zero_grad()
            loss = criterion(mlp(x_t.unsqueeze(0)).squeeze(0), y_t)
            loss.backward()
            optimizer.step()
    
    # Train GraphCG
    optimizer = torch.optim.Adam(gc.parameters(), lr=0.001)
    for epoch in range(epochs):
        gc.train()
        for x_t, y_t in train_t:
            optimizer.zero_grad()
            loss = criterion(gc(x_t.unsqueeze(0)).squeeze(0), y_t)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    mlp.eval()
    gc.eval()
    
    mlp_loss = sum(criterion(mlp(x_t.unsqueeze(0)).squeeze(0), y_t).item() for x_t, y_t in test_t) / len(test_t)
    gc_loss = sum(criterion(gc(x_t.unsqueeze(0)).squeeze(0), y_t).item() for x_t, y_t in test_t) / len(test_t)
    
    return mlp_loss, gc_loss


def main():
    print("=" * 60)
    print("H1.441: Parameter-Matched Architecture Comparison")
    print("=" * 60)
    
    complexity_levels = [
        {"level": 1, "objects": 2, "steps": 5},
        {"level": 2, "objects": 4, "steps": 10},
        {"level": 3, "objects": 6, "steps": 15},
        {"level": 4, "objects": 8, "steps": 20},
    ]
    
    n_samples = 400
    train_ratio = 0.7
    val_ratio = 0.15
    
    all_results = []
    
    for comp in complexity_levels:
        level = comp["level"]
        n_objects = comp["objects"]
        n_steps = comp["steps"]
        input_dim = 2 * n_objects
        
        print(f"\nLevel {level}: {n_objects} objects, {n_steps} steps")
        
        # Generate data
        data = generate_data(n_samples, n_objects, n_steps, seed=100+level)
        random.shuffle(data)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train+n_val]
        test_data = data[n_train+n_val:]
        
        # Create models
        mlp = MLP(input_dim=input_dim, hidden_dim=64, output_dim=2, n_layers=2)
        n_nodes = min(n_objects + 2, 10)
        gc = GraphCG(input_dim=input_dim, hidden_dim=64, output_dim=2, n_heads=4, n_nodes=n_nodes, n_layers=2)
        
        mlp_loss, gc_loss = train_and_eval(mlp, gc, train_data, val_data, test_data, epochs=30)
        
        improvement = ((mlp_loss - gc_loss) / mlp_loss) * 100
        print(f"  MLP={mlp_loss:.6f}, GraphCG={gc_loss:.6f}, Improvement={improvement:+.1f}%")
        
        all_results.append({
            "level": level,
            "n_objects": n_objects,
            "mlp_mse": mlp_loss,
            "graphcg_mse": gc_loss,
            "improvement": improvement,
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for r in all_results:
        print(f"  Level {r['level']}: {r['improvement']:+.1f}%")
    
    improvements = [r['improvement'] for r in all_results]
    avg = np.mean(improvements)
    slope = np.polyfit([1,2,3,4], improvements, 1)[0]
    
    print(f"\nAvg: {avg:+.1f}%, Trend: {slope:+.1f}%/level")
    
    if avg > 0 and slope > -10:
        conclusion = "SUPPORTED"
        key = f"GraphCG maintains {avg:+.1f}% advantage with stable trend"
    elif avg > 0:
        conclusion = "PARTIALLY_SUPPORTED"
        key = f"GraphCG shows {avg:+.1f}% but advantage diminishes ({slope:+.1f}%/level)"
    else:
        conclusion = "REFUTED"
        key = f"MLP outperforms GraphCG by {-avg:.1f}%"
    
    print(f"Conclusion: {conclusion}")
    print(f"Key: {key}")
    
    output = {
        "experiment_id": "H1.441",
        "round": 207,
        "conclusion": conclusion,
        "key_insight": key,
        "results": all_results,
        "summary": {"avg_improvement": avg, "trend_slope": slope}
    }
    
    with open("experiment_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nSaved to experiment_results.json")
    return output


if __name__ == "__main__":
    main()
