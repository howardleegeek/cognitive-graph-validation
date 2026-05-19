#!/usr/bin/env python3
"""
H1.440 - Robust GraphCG Scaling Test
Design more stable tasks and run proper experiment with 3+ trials per complexity level
to get statistically significant results on whether GraphCG advantage scales with task complexity.
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class RobustGraphCG(nn.Module):
    """More robust GraphCG implementation with better stability."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, n_passes=3, n_nodes=6):
        super().__init__()
        self.n_passes = n_passes
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        
        # Project input to node embeddings with layer norm for stability
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, n_nodes * hidden_dim),
            nn.LayerNorm(n_nodes * hidden_dim)
        )
        
        # Message passing layers with residual connections
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Node update with residual
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(n_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable adjacency with softmax normalization
        self.adjacency = nn.Parameter(torch.randn(n_nodes, n_nodes))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize nodes
        nodes = self.node_proj(x).view(batch_size, self.n_nodes, self.hidden_dim)
        
        # Normalize adjacency
        adj = F.softmax(self.adjacency, dim=-1)
        
        # Message passing iterations
        for _ in range(self.n_passes):
            # Compute messages between all pairs of nodes
            messages = []
            for i in range(self.n_nodes):
                node_i = nodes[:, i:i+1, :]  # [B, 1, D]
                node_i_expanded = node_i.expand(-1, self.n_nodes, -1)  # [B, N, D]
                
                # Concatenate node_i with all other nodes
                pairs = torch.cat([node_i_expanded, nodes], dim=-1)  # [B, N, 2D]
                msg_i = self.message_mlp(pairs)  # [B, N, D]
                
                # Weight messages by adjacency
                weighted_msg_i = msg_i * adj[i:i+1, :].unsqueeze(0)  # [B, N, D]
                aggregated_msg_i = torch.sum(weighted_msg_i, dim=1, keepdim=True)  # [B, 1, D]
                messages.append(aggregated_msg_i)
            
            messages = torch.cat(messages, dim=1)  # [B, N, D]
            
            # Update nodes with residual connection
            node_inputs = torch.cat([nodes, messages], dim=-1)
            node_updates = self.update_mlp(node_inputs)
            nodes = nodes + 0.1 * node_updates  # Small step size for stability
        
        # Flatten and project to output
        nodes_flat = nodes.reshape(batch_size, -1)
        return self.output_proj(nodes_flat)

# ============================================================
# Task Generators (More Stable)
# ============================================================

def generate_stable_task(n_objects, seq_length, n_samples=1000):
    """
    Generate more stable tasks with better conditioning.
    
    Args:
        n_objects: Number of objects in the scene
        seq_length: Length of transformation sequence
        n_samples: Number of samples to generate
    
    Returns:
        X: Input features [n_samples, input_dim]
        y: Target outputs [n_samples, 1]
    """
    # Each object has position (x, y) and velocity (dx, dy)
    # More stable initialization with controlled variance
    input_dim = n_objects * 4 + seq_length * 2  # Object states + transformation sequence
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Initialize objects with controlled variance
        objects = []
        for i in range(n_objects):
            # Position in [0, 1] with spacing
            x = np.random.uniform(i/(n_objects+1), (i+1)/(n_objects+1))
            y = np.random.uniform(0.2, 0.8)
            # Small velocities
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            objects.extend([x, y, dx, dy])
        
        # Generate transformation sequence with smooth transitions
        transformations = []
        current_transform = np.array([0.0, 0.0])
        
        for step in range(seq_length):
            # Smooth transformations with momentum
            delta = np.random.uniform(-0.2, 0.2, size=2) * (1.0 / (step + 1))
            current_transform = current_transform * 0.8 + delta * 0.2
            transformations.extend(current_transform.tolist())
        
        # Combine features
        features = objects + transformations
        X.append(features)
        
        # Compute target: final position of first object after transformations
        # More stable computation with bounded effects
        obj_x, obj_y = objects[0], objects[1]
        total_dx, total_dy = 0.0, 0.0
        
        for i in range(0, len(transformations), 2):
            if i < len(transformations):
                total_dx += transformations[i] * 0.5  # Reduced effect
                total_dy += transformations[i+1] * 0.5
        
        final_x = obj_x + total_dx
        final_y = obj_y + total_dy
        
        # Bound output to prevent extreme values
        final_x = np.clip(final_x, 0.0, 1.0)
        final_y = np.clip(final_y, 0.0, 1.0)
        
        y.append([final_x])
    
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
    """Train model with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

def run_experiment():
    """Run robust scaling experiment with multiple trials."""
    print("=" * 80)
    print("H1.440 - Robust GraphCG Scaling Test")
    print("=" * 80)
    
    # Experiment configuration
    complexity_levels = [
        {"n_objects": 2, "seq_length": 5, "name": "Level 1 (Simple)"},
        {"n_objects": 4, "seq_length": 10, "name": "Level 2 (Medium)"},
        {"n_objects": 6, "seq_length": 15, "name": "Level 3 (Complex)"},
        {"n_objects": 8, "seq_length": 20, "name": "Level 4 (Very Complex)"},
    ]
    
    n_trials = 5  # Increased from 1 to 5 for statistical significance
    n_samples = 2000  # More samples for better statistics
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
            
            # Normalize data for stability
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
            mlp_val_loss = train_model(mlp, X_train, y_train, X_val, y_val, epochs=150)
            mlp_test_loss = evaluate_model(mlp, X_test, y_test)
            mlp_losses.append(mlp_test_loss)
            
            # Train GraphCG
            print("  Training GraphCG...")
            graphcg = RobustGraphCG(input_dim=input_dim, hidden_dim=64, output_dim=1, 
                                   n_passes=3, n_nodes=min(6, n_objects))
            graphcg_val_loss = train_model(graphcg, X_train, y_train, X_val, y_val, epochs=150)
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
    
    # Calculate trend (does improvement increase with complexity?)
    complexity_indices = list(range(1, len(complexity_levels) + 1))
    if len(all_improvements) > 1:
        # Simple linear regression for trend
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
    results = run_experiment()