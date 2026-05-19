#!/usr/bin/env python3
"""
H1.440 - Robust GraphCG Scaling Test - Full version
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from pathlib import Path
import time
import statistics
from collections import defaultdict

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Model Definitions
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


class RobustGraphCG(nn.Module):
    """More robust GraphCG implementation."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, n_passes=3, n_nodes=6):
        super().__init__()
        self.n_passes = n_passes
        self.n_nodes = n_nodes
        
        # Project input to node embeddings
        self.node_proj = nn.Linear(input_dim, n_nodes * hidden_dim)
        
        # Message passing layers
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
            nodes = nodes + 0.1 * messages
        
        # Flatten and project to output
        nodes_flat = nodes.view(batch_size, -1)
        return self.output_proj(nodes_flat)

# ============================================================
# Task Generators
# ============================================================

def generate_stable_task(n_objects, seq_length, n_samples=1000):
    """
    Generate stable tasks.
    """
    input_dim = n_objects * 4 + seq_length * 2
    
    X_list = []
    y_list = []
    
    for _ in range(n_samples):
        # Initialize objects
        objects = []
        for i in range(n_objects):
            x = np.random.uniform(i/(n_objects+1), (i+1)/(n_objects+1))
            y = np.random.uniform(0.2, 0.8)
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            objects.extend([x, y, dx, dy])
        
        # Generate transformation sequence
        transformations = []
        current_transform = np.array([0.0, 0.0])
        
        for step in range(seq_length):
            delta = np.random.uniform(-0.2, 0.2, size=2) * (1.0 / (step + 1))
            current_transform = current_transform * 0.8 + delta * 0.2
            transformations.extend(current_transform.tolist())
        
        # Combine features
        features = objects + transformations
        X_list.append(features)
        
        # Compute target
        obj_x, obj_y = objects[0], objects[1]
        total_dx, total_dy = 0.0, 0.0
        
        for i in range(0, len(transformations), 2):
            if i < len(transformations):
                total_dx += transformations[i] * 0.5
                total_dy += transformations[i+1] * 0.5
        
        final_x = obj_x + total_dx
        final_x = np.clip(final_x, 0.0, 1.0)
        
        y_list.append([final_x])
    
    return torch.FloatTensor(np.array(X_list)), torch.FloatTensor(np.array(y_list))

# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
    """Train model with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return best_val_loss

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = F.mse_loss(y_pred, y_test).item()
    return mse

# ============================================================
# Main Experiment
# ============================================================

def run_full_experiment():
    """Run full scaling experiment."""
    print("=" * 80)
    print("H1.440 - Robust GraphCG Scaling Test (Full)")
    print("=" * 80)
    
    # Experiment configuration
    complexity_levels = [
        {"n_objects": 2, "seq_length": 5, "name": "Level 1 (Simple)"},
        {"n_objects": 4, "seq_length": 10, "name": "Level 2 (Medium)"},
        {"n_objects": 6, "seq_length": 15, "name": "Level 3 (Complex)"},
        {"n_objects": 8, "seq_length": 20, "name": "Level 4 (Very Complex)"},
    ]
    
    n_trials = 5
    n_samples = 1500
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    results = {
        "experiment_id": "H1.440",
        "description": "Robust GraphCG scaling test with multiple trials",
        "config": {
            "n_trials": n_trials,
            "n_samples": n_samples,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "complexity_levels": len(complexity_levels),
            "mlp_hidden_dim": 64,
            "graphcg_hidden_dim": 64,
            "graphcg_n_passes": 3,
            "graphcg_n_nodes": 6,
        },
        "results": {}
    }
    
    all_improvements = []
    
    for level_idx, level_config in enumerate(complexity_levels):
        n_objects = level_config["n_objects"]
        seq_length = level_config["seq_length"]
        level_name = level_config["name"]
        
        print(f"\n{'='*60}")
        print(f"Testing {level_name}: {n_objects} objects, {seq_length} steps")
        print(f"{'='*60}")
        
        mlp_losses = []
        graphcg_losses = []
        improvements = []
        
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            
            # Generate data with different random seed for each trial
            np.random.seed(42 + trial * 100 + level_idx * 1000)
            torch.manual_seed(42 + trial * 100 + level_idx * 1000)
            
            # Generate dataset
            X, y = generate_stable_task(n_objects, seq_length, n_samples)
            
            # Split data
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            indices = torch.randperm(n_samples)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Normalize data
            X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
            X_std = torch.where(X_std == 0, torch.ones_like(X_std), X_std)
            X_train = (X_train - X_mean) / X_std
            X_val = (X_val - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            
            y_mean, y_std = y_train.mean(dim=0), y_train.std(dim=0)
            y_std = torch.where(y_std == 0, torch.ones_like(y_std), y_std)
            y_train = (y_train - y_mean) / y_std
            y_val = (y_val - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std
            
            # Define input dimension
            input_dim = n_objects * 4 + seq_length * 2
            
            # Train MLP
            print("  Training MLP...")
            mlp = BaselineMLP(input_dim=input_dim, hidden_dim=64, output_dim=1)
            mlp_val_loss = train_model(mlp, X_train, y_train, X_val, y_val, epochs=100)
            mlp_test_loss = evaluate_model(mlp, X_test, y_test)
            mlp_losses.append(mlp_test_loss)
            
            # Train GraphCG
            print("  Training GraphCG...")
            graphcg = RobustGraphCG(input_dim=input_dim, hidden_dim=64, output_dim=1, 
                                   n_passes=3, n_nodes=min(6, n_objects))
            graphcg_val_loss = train_model(graphcg, X_train, y_train, X_val, y_val, epochs=100)
            graphcg_test_loss = evaluate_model(graphcg, X_test, y_test)
            graphcg_losses.append(graphcg_test_loss)
            
            # Calculate improvement
            improvement = ((graphcg_test_loss - mlp_test_loss) / mlp_test_loss) * 100
            improvements.append(improvement)
            
            print(f"    MLP MSE: {mlp_test_loss:.6f}, GraphCG MSE: {graphcg_test_loss:.6f}, "
                  f"Improvement: {improvement:+.2f}%")
        
        # Calculate statistics for this level
        avg_mlp = statistics.mean(mlp_losses)
        avg_graphcg = statistics.mean(graphcg_losses)
        avg_improvement = statistics.mean(improvements)
        std_improvement = statistics.stdev(improvements) if len(improvements) > 1 else 0
        
        # Store results
        level_key = f"level_{level_idx+1}"
        results["results"][level_key] = {
            "n_objects": n_objects,
            "seq_length": seq_length,
            "mlp_losses": mlp_losses,
            "graphcg_losses": graphcg_losses,
            "improvements": improvements,
            "avg_mlp_mse": avg_mlp,
            "avg_graphcg_mse": avg_graphcg,
            "avg_improvement": avg_improvement,
            "std_improvement": std_improvement,
            "mlp_best": min(mlp_losses),
            "mlp_worst": max(mlp_losses),
            "graphcg_best": min(graphcg_losses),
            "graphcg_worst": max(graphcg_losses),
        }
        
        all_improvements.append(avg_improvement)
        
        print(f"\n{level_name} Summary:")
        print(f"  MLP MSE: {avg_mlp:.6f} (best: {min(mlp_losses):.6f}, worst: {max(mlp_losses):.6f})")
        print(f"  GraphCG MSE: {avg_graphcg:.6f} (best: {min(graphcg_losses):.6f}, worst: {max(graphcg_losses):.6f})")
        print(f"  Average Improvement: {avg_improvement:+.2f}% ± {std_improvement:.2f}%")
        print(f"  Improvement Range: [{min(improvements):+.2f}%, {max(improvements):+.2f}%]")
    
    # Calculate overall statistics
    results["summary"] = {
        "overall_avg_improvement": statistics.mean(all_improvements),
        "overall_std_improvement": statistics.stdev(all_improvements) if len(all_improvements) > 1 else 0,
        "improvement_trend": all_improvements,
    }
    
    # Calculate trend
    complexity_indices = list(range(1, len(complexity_levels) + 1))
    if len(all_improvements) > 1:
        x = np.array(complexity_indices)
        y = np.array(all_improvements)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        results["summary"]["trend_slope"] = float(m)
        results["summary"]["trend_intercept"] = float(c)
        
        # Classify trend
        if m < -1.0:
            trend_class = "STRONG_NEGATIVE"
        elif m < -0.5:
            trend_class = "MODERATE_NEGATIVE"
        elif m < -0.1:
            trend_class = "WEAK_NEGATIVE"
        elif m < 0.1:
            trend_class = "FLAT"
        elif m < 0.5:
            trend_class = "WEAK_POSITIVE"
        elif m < 1.0:
            trend_class = "MODERATE_POSITIVE"
        else:
            trend_class = "STRONG_POSITIVE"
        results["summary"]["trend_class"] = trend_class
    else:
        results["summary"]["trend_slope"] = 0.0
        results["summary"]["trend_class"] = "INSUFFICIENT_DATA"
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Overall Average Improvement: {results['summary']['overall_avg_improvement']:+.2f}%")
    print(f"Overall Std Improvement: {results['summary']['overall_std_improvement']:.2f}%")
    print(f"Trend Slope: {results['summary'].get('trend_slope', 0):+.2f}% per complexity level")
    print(f"Trend Classification: {results['summary'].get('trend_class', 'N/A')}")
    
    # Save results
    results_file = Path(__file__).parent / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    results = run_full_experiment()