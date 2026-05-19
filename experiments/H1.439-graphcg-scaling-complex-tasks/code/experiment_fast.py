#!/usr/bin/env python3
"""
H1.439 - GraphCG Scaling: Fast version
Test GraphCG vs MLP on tasks with increasing complexity.
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Model Definitions (Simplified)
# ============================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimpleGraphCG(nn.Module):
    """Simplified GraphCG for faster training."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, n_passes=3, n_nodes=6):
        super().__init__()
        self.n_passes = n_passes
        self.n_nodes = n_nodes
        
        # Project input to node embeddings
        self.node_proj = nn.Linear(input_dim, n_nodes * hidden_dim)
        
        # Simple message passing (no learnable adjacency)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(n_nodes * hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize nodes
        nodes = self.node_proj(x).view(batch_size, self.n_nodes, -1)
        
        # Message passing iterations
        for _ in range(self.n_passes):
            # Compute all-pair messages
            messages = []
            for i in range(self.n_nodes):
                node_i = nodes[:, i:i+1, :]
                node_i_expanded = node_i.expand(-1, self.n_nodes, -1)
                
                # Concatenate and compute message
                pairs = torch.cat([node_i_expanded, nodes], dim=-1)
                msg_i = self.message_mlp(pairs)
                
                # Average messages from all nodes
                aggregated_msg_i = torch.mean(msg_i, dim=1, keepdim=True)
                messages.append(aggregated_msg_i)
            
            messages = torch.cat(messages, dim=1)
            
            # Update nodes (simple residual)
            nodes = nodes + messages
        
        # Flatten and project to output
        nodes_flat = nodes.view(batch_size, -1)
        return self.output_proj(nodes_flat)

# ============================================================
# Task Generators (Simplified)
# ============================================================

def generate_task_complexity_level(level, n_samples=1000):
    """
    Generate tasks with increasing complexity:
    level 1: 2 objects, 5 steps
    level 2: 4 objects, 10 steps  
    level 3: 6 objects, 15 steps
    level 4: 8 objects, 20 steps
    """
    n_objects = 2 * level
    seq_len = 5 * level
    
    # Generate object properties
    properties = np.random.randn(n_samples, n_objects, 3) * 1.0
    
    # Generate simple transformation
    transform = np.random.randn(3, 3) * 0.5
    bias = np.random.randn(3) * 0.2
    
    # Apply transformation seq_len times
    current = properties.copy()
    for _ in range(seq_len):
        current_flat = current.reshape(n_samples * n_objects, 3)
        transformed = current_flat @ transform.T + bias
        current = transformed.reshape(n_samples, n_objects, 3)
        current = np.tanh(current)  # Non-linearity
    
    # Target: final state of first object
    targets = current[:, 0, :]
    
    # Input: initial properties
    features = properties.reshape(n_samples, -1)
    
    # Add task complexity indicator
    complexity_feat = np.ones((n_samples, 1)) * level
    features = np.concatenate([features, complexity_feat], axis=1)
    
    return torch.FloatTensor(features), torch.FloatTensor(targets)

# ============================================================
# Training
# ============================================================

def train_and_evaluate(model, train_X, train_y, test_X, test_y, epochs=10, lr=1e-3):
    """Quick training and evaluation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Quick training
    for epoch in range(epochs):
        model.train()
        pred = model(train_X)
        loss = F.mse_loss(pred, train_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(test_X)
        test_mse = F.mse_loss(test_pred, test_y).item()
    
    return test_mse

def main():
    print("="*60)
    print("H1.439 - GraphCG Scaling (Fast Version)")
    print("="*60)
    print("\nTesting if GraphCG advantage increases with task complexity")
    
    results = {}
    
    # Test across 4 complexity levels
    for level in range(1, 5):
        print(f"\n{'='*40}")
        print(f"Complexity Level {level}: {2*level} objects, {5*level} steps")
        print(f"{'='*40}")
        
        # Generate data
        X, y = generate_task_complexity_level(level, n_samples=800)
        
        # Split
        n_train = 600
        n_test = 200
        
        indices = torch.randperm(X.size(0))
        train_X, train_y = X[indices[:n_train]], y[indices[:n_train]]
        test_X, test_y = X[indices[n_train:n_train+n_test]], y[indices[n_train:n_train+n_test]]
        
        input_dim = train_X.size(1)
        output_dim = train_y.size(1)
        
        # Models
        mlp = BaselineMLP(input_dim, hidden_dim=64, output_dim=output_dim)
        graphcg = SimpleGraphCG(input_dim, hidden_dim=64, output_dim=output_dim, n_passes=3, n_nodes=6)
        
        # Train and evaluate
        print("Training MLP...")
        mlp_mse = train_and_evaluate(mlp, train_X, train_y, test_X, test_y, epochs=10)
        
        print("Training GraphCG...")
        graphcg_mse = train_and_evaluate(graphcg, train_X, train_y, test_X, test_y, epochs=10)
        
        # Calculate improvement
        improvement = ((graphcg_mse - mlp_mse) / mlp_mse) * 100
        
        results[level] = {
            "n_objects": 2 * level,
            "seq_len": 5 * level,
            "mlp_mse": mlp_mse,
            "graphcg_mse": graphcg_mse,
            "improvement_pct": improvement,
            "input_dim": input_dim,
            "output_dim": output_dim
        }
        
        print(f"  MLP MSE: {mlp_mse:.6f}")
        print(f"  GraphCG MSE: {graphcg_mse:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    # Analyze trend
    print("\n" + "="*60)
    print("TREND ANALYSIS")
    print("="*60)
    
    levels = list(results.keys())
    improvements = [results[level]["improvement_pct"] for level in levels]
    
    print("\nImprovement by complexity level:")
    for level in levels:
        imp = results[level]["improvement_pct"]
        print(f"  Level {level} ({results[level]['n_objects']} objects, {results[level]['seq_len']} steps): {imp:+.1f}%")
    
    # Calculate trend
    if len(levels) > 1:
        slope, intercept = np.polyfit(levels, improvements, 1)
        print(f"\nTrend slope: {slope:.2f}% per complexity level")
        
        # Statistical test (simple)
        if slope < -1.0:  # Strong negative trend
            trend = "STRONG_NEGATIVE"
            conclusion = "GraphCG advantage STRONGLY INCREASES with complexity"
        elif slope < -0.5:
            trend = "MODERATE_NEGATIVE" 
            conclusion = "GraphCG advantage moderately increases with complexity"
        elif slope < 0:
            trend = "WEAK_NEGATIVE"
            conclusion = "GraphCG advantage weakly increases with complexity"
        elif slope < 0.5:
            trend = "WEAK_POSITIVE"
            conclusion = "GraphCG advantage weakly decreases with complexity"
        else:
            trend = "STRONG_POSITIVE"
            conclusion = "GraphCG advantage decreases with complexity"
        
        print(f"Trend: {trend}")
        print(f"Conclusion: {conclusion}")
    
    # Save results
    output = {
        "experiment_id": "H1.439-fast",
        "description": "GraphCG Scaling: Fast test of complexity trend",
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "trend_analysis": {
            "levels": levels,
            "improvements": improvements,
            "slope": float(slope) if len(levels) > 1 else None,
            "intercept": float(intercept) if len(levels) > 1 else None,
            "trend": trend if len(levels) > 1 else "N/A",
            "conclusion": conclusion if len(levels) > 1 else "N/A"
        },
        "key_findings": {
            "tested_complexity_levels": f"1-{max(levels)}",
            "objects_range": f"{results[1]['n_objects']}-{results[max(levels)]['n_objects']}",
            "sequence_range": f"{results[1]['seq_len']}-{results[max(levels)]['seq_len']}",
            "avg_improvement": float(np.mean(improvements)) if improvements else 0,
            "best_improvement": float(min(improvements)) if improvements else 0,
            "worst_improvement": float(max(improvements)) if improvements else 0
        }
    }
    
    output_path = Path(__file__).parent / "results_fast.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Final conclusion
    print("\n" + "="*60)
    print("FINAL CONCLUSION")
    print("="*60)
    
    avg_imp = np.mean(improvements)
    if avg_imp < -5:
        print(f"SUPPORTED: GraphCG outperforms MLP by {-avg_imp:.1f}% on average")
        if 'slope' in locals() and slope < -0.5:
            print(f"STRONGLY SUPPORTED: Advantage increases with complexity (slope: {slope:.2f}%/level)")
        elif 'slope' in locals() and slope < 0:
            print(f"PARTIALLY SUPPORTED: Outperforms but trend is weak (slope: {slope:.2f}%/level)")
        else:
            print("PARTIALLY SUPPORTED: Outperforms but doesn't scale clearly with complexity")
    elif avg_imp < 0:
        print(f"WEAKLY SUPPORTED: GraphCG slightly better ({-avg_imp:.1f}%)")
    else:
        print(f"REFUTED: GraphCG performs worse than MLP (+{avg_imp:.1f}%)")
    
    return output

if __name__ == "__main__":
    main()