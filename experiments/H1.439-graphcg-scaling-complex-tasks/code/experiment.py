#!/usr/bin/env python3
"""
H1.439 - GraphCG Scaling: Does advantage increase with task complexity?
Test GraphCG vs MLP on tasks with 6+ objects and longer sequences (20+ timesteps)
to test if graph structure advantage compounds with problem complexity.
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
import time

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Model Definitions
# ============================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=32):
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


class GraphCG(nn.Module):
    """Graph Cognitive Graph with explicit message passing."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=32, n_passes=3, n_nodes=8):
        super().__init__()
        self.n_passes = n_passes
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        
        # Project input to node embeddings
        self.node_proj = nn.Linear(input_dim, n_nodes * hidden_dim)
        
        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update layers
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(n_nodes * hidden_dim, output_dim)
        
        # Learnable adjacency matrix (fully connected graph)
        self.adjacency = nn.Parameter(torch.ones(n_nodes, n_nodes) / n_nodes)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize nodes
        nodes = self.node_proj(x).view(batch_size, self.n_nodes, self.hidden_dim)
        
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
                
                # Weight messages by adjacency (broadcast correctly)
                # adjacency[i, :] has shape [N], need [1, N, 1] for broadcasting
                adj_i = self.adjacency[i:i+1, :].unsqueeze(0).unsqueeze(-1)  # [1, 1, N, 1]
                adj_i = adj_i.expand(batch_size, 1, self.n_nodes, 1)  # [B, 1, N, 1]
                msg_i_expanded = msg_i.unsqueeze(1)  # [B, 1, N, D]
                
                weighted_msg_i = msg_i_expanded * adj_i  # [B, 1, N, D]
                aggregated_msg_i = torch.sum(weighted_msg_i, dim=2)  # [B, 1, D]
                messages.append(aggregated_msg_i)
            
            messages = torch.cat(messages, dim=1)  # [B, N, D]
            
            # Update nodes
            node_updates = self.update_mlp(torch.cat([nodes, messages], dim=-1))
            nodes = nodes + node_updates  # Residual connection
        
        # Flatten and project to output
        nodes_flat = nodes.view(batch_size, -1)
        return self.output_proj(nodes_flat)

# ============================================================
# Task Generators
# ============================================================

def generate_complex_relational_task(n_samples=5000, n_objects=6, seq_len=20, noise_std=0.1):
    """
    Generate complex relational reasoning task with 6+ objects.
    Task: Predict whether object A is closer to B than C is to D,
    with additional constraints about object relationships.
    """
    # Generate random object positions in 3D space
    positions = np.random.randn(n_samples, n_objects, 3) * 2.0
    
    # Generate random queries: (A, B, C, D) indices
    queries = np.random.randint(0, n_objects, size=(n_samples, 4))
    
    # Compute distances
    dist_AB = np.linalg.norm(
        positions[np.arange(n_samples), queries[:, 0]] - 
        positions[np.arange(n_samples), queries[:, 1]], axis=1
    )
    dist_CD = np.linalg.norm(
        positions[np.arange(n_samples), queries[:, 2]] - 
        positions[np.arange(n_samples), queries[:, 3]], axis=1
    )
    
    # Target: 1 if A closer to B than C to D, else 0
    targets = (dist_AB < dist_CD).astype(np.float32)
    
    # Flatten positions and add query indices as one-hot
    positions_flat = positions.reshape(n_samples, -1)
    queries_onehot = np.zeros((n_samples, n_objects * 4))
    for i in range(4):
        queries_onehot[np.arange(n_samples), queries[:, i] + i * n_objects] = 1
    
    # Combine features
    features = np.concatenate([positions_flat, queries_onehot], axis=1)
    
    # Add noise
    features = features + np.random.randn(*features.shape) * noise_std
    
    return torch.FloatTensor(features), torch.FloatTensor(targets).unsqueeze(1)

def generate_complex_compositional_task(n_samples=5000, n_objects=6, seq_len=20, noise_std=0.1):
    """
    Generate complex compositional task with 6+ objects and longer sequences.
    Task: Apply a sequence of transformations to object properties.
    """
    # Generate initial object properties (position, velocity, mass)
    properties = np.random.randn(n_samples, n_objects, 5) * 1.0
    
    # Generate transformation sequences
    transformations = []
    for _ in range(seq_len):
        # Each transformation is a simple linear operation
        transform_matrix = np.random.randn(5, 5) * 0.5
        transform_bias = np.random.randn(5) * 0.2
        transformations.append((transform_matrix, transform_bias))
    
    # Apply transformations sequentially
    current = properties.copy()
    for transform_matrix, transform_bias in transformations:
        # Reshape for batch matrix multiplication
        current_reshaped = current.reshape(n_samples * n_objects, 5)
        transformed = current_reshaped @ transform_matrix.T + transform_bias
        current = transformed.reshape(n_samples, n_objects, 5)
        
        # Add some non-linearity
        current = np.tanh(current)
    
    # Target: final properties of first object
    targets = current[:, 0, :]  # [n_samples, 5]
    
    # Input: initial properties + transformation parameters
    # Flatten initial properties
    features_flat = properties.reshape(n_samples, -1)
    
    # Add transformation sequence as context
    transform_params = []
    for transform_matrix, transform_bias in transformations:
        transform_params.append(transform_matrix.flatten())
        transform_params.append(transform_bias)
    
    transform_params = np.concatenate(transform_params)
    # Repeat for each sample
    transform_features = np.tile(transform_params, (n_samples, 1))
    
    # Combine
    features = np.concatenate([features_flat, transform_features], axis=1)
    
    # Add noise
    features = features + np.random.randn(*features.shape) * noise_std
    
    return torch.FloatTensor(features), torch.FloatTensor(targets)

def generate_multi_step_manipulation_task(n_samples=5000, n_objects=6, seq_len=25, noise_std=0.1):
    """
    Generate multi-step robot manipulation task with 6+ objects and 25+ timesteps.
    Simulates pushing multiple objects to target positions.
    """
    # Initial object positions
    init_positions = np.random.randn(n_samples, n_objects, 3) * 2.0
    
    # Target positions for each object
    target_positions = np.random.randn(n_samples, n_objects, 3) * 2.0
    
    # Generate action sequence (simplified: push forces)
    actions = np.random.randn(n_samples, seq_len, 3) * 0.5
    
    # Simulate physics: objects move toward targets with some noise
    # and are affected by push actions
    current_positions = init_positions.copy()
    
    # Store trajectory
    trajectory = []
    for t in range(seq_len):
        # Compute forces: attraction to targets + random pushes
        attraction_forces = (target_positions - current_positions) * 0.1
        
        # Apply push action (affects all objects slightly)
        push_force = actions[:, t:t+1, :] * 0.2  # [B, 1, 3]
        push_force = np.repeat(push_force, n_objects, axis=1)  # [B, N, 3]
        
        # Update positions
        current_positions = current_positions + attraction_forces + push_force
        
        # Add some friction/damping
        current_positions = current_positions * 0.95
        
        trajectory.append(current_positions.copy())
    
    # Final positions
    final_positions = current_positions
    
    # Target: final positions of all objects
    targets = final_positions.reshape(n_samples, -1)
    
    # Input: initial positions + target positions + action sequence
    init_flat = init_positions.reshape(n_samples, -1)
    target_flat = target_positions.reshape(n_samples, -1)
    actions_flat = actions.reshape(n_samples, -1)
    
    features = np.concatenate([init_flat, target_flat, actions_flat], axis=1)
    
    # Add noise
    features = features + np.random.randn(*features.shape) * noise_std
    
    return torch.FloatTensor(features), torch.FloatTensor(targets)

# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_X, train_y, val_X, val_y, epochs=20, lr=3e-4, batch_size=64):
    """Train a model and return validation MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_samples = train_X.size(0)
    best_val_mse = float('inf')
    
    for epoch in range(epochs):
        # Shuffle
        indices = torch.randperm(n_samples)
        train_X_shuffled = train_X[indices]
        train_y_shuffled = train_y[indices]
        
        model.train()
        epoch_loss = 0.0
        
        for i in range(0, n_samples, batch_size):
            batch_X = train_X_shuffled[i:i+batch_size]
            batch_y = train_y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
        
        epoch_loss /= n_samples
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_mse = F.mse_loss(val_pred, val_y).item()
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
        
        scheduler.step()
    
    return best_val_mse

def run_experiment(task_name, data_generator, n_trials=3, epochs=20):
    """Run experiment comparing MLP vs GraphCG on a task."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    
    results = {}
    
    for trial in range(n_trials):
        print(f"\nTrial {trial+1}/{n_trials}")
        
        # Generate data
        X, y = data_generator()
        n_samples = X.size(0)
        
        # Split
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        train_X, train_y = X[train_idx], y[train_idx]
        val_X, val_y = X[val_idx], y[val_idx]
        test_X, test_y = X[val_idx], y[val_idx]  # Using val as test for simplicity
        
        input_dim = train_X.size(1)
        output_dim = train_y.size(1)
        
        # Models to test
        models = {
            "MLP-128": BaselineMLP(input_dim, hidden_dim=128, output_dim=output_dim),
            "GraphCG-128-3p": GraphCG(input_dim, hidden_dim=128, output_dim=output_dim, n_passes=3, n_nodes=8),
            "GraphCG-128-6p": GraphCG(input_dim, hidden_dim=128, output_dim=output_dim, n_passes=6, n_nodes=8),
            "GraphCG-256-3p": GraphCG(input_dim, hidden_dim=256, output_dim=output_dim, n_passes=3, n_nodes=8),
        }
        
        for name, model in models.items():
            print(f"  Training {name}...")
            start_time = time.time()
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            
            # Train
            val_mse = train_model(
                model, train_X, train_y, val_X, val_y,
                epochs=epochs, lr=3e-4, batch_size=64
            )
            
            train_time = time.time() - start_time
            
            # Test
            model.eval()
            with torch.no_grad():
                test_pred = model(test_X)
                test_mse = F.mse_loss(test_pred, test_y).item()
            
            if name not in results:
                results[name] = {
                    "mse_values": [],
                    "train_times": [],
                    "n_params": n_params
                }
            
            results[name]["mse_values"].append(test_mse)
            results[name]["train_times"].append(train_time)
    
    # Compute statistics
    summary = {}
    for name, data in results.items():
        mse_values = np.array(data["mse_values"])
        summary[name] = {
            "mean_mse": float(np.mean(mse_values)),
            "std_mse": float(np.std(mse_values)),
            "mean_train_time": float(np.mean(data["train_times"])),
            "n_params": data["n_params"]
        }
        print(f"  {name}: MSE = {summary[name]['mean_mse']:.6f} ± {summary[name]['std_mse']:.6f}")
    
    return summary

def main():
    print("="*60)
    print("H1.439 - GraphCG Scaling: Does advantage increase with task complexity?")
    print("="*60)
    print("\nTesting GraphCG vs MLP on tasks with 6+ objects and 20+ timesteps")
    print("to see if graph structure advantage compounds with problem complexity.")
    
    all_results = {}
    
    # Task 1: Complex relational reasoning (6 objects)
    print("\n" + "="*60)
    print("Task 1: Complex Relational Reasoning (6 objects)")
    print("="*60)
    all_results["complex_relational_6obj"] = run_experiment(
        "Complex Relational (6 objects)",
        lambda: generate_complex_relational_task(n_samples=2000, n_objects=6, seq_len=20),
        n_trials=3,
        epochs=15
    )
    
    # Task 2: Complex compositional reasoning (6 objects, 20 steps)
    print("\n" + "="*60)
    print("Task 2: Complex Compositional Reasoning (6 objects, 20 steps)")
    print("="*60)
    all_results["complex_compositional_6obj_20step"] = run_experiment(
        "Complex Compositional (6 objects, 20 steps)",
        lambda: generate_complex_compositional_task(n_samples=2000, n_objects=6, seq_len=20),
        n_trials=3,
        epochs=15
    )
    
    # Task 3: Multi-step manipulation (6 objects, 25 steps)
    print("\n" + "="*60)
    print("Task 3: Multi-step Manipulation (6 objects, 25 steps)")
    print("="*60)
    all_results["multi_step_manipulation_6obj_25step"] = run_experiment(
        "Multi-step Manipulation (6 objects, 25 steps)",
        lambda: generate_multi_step_manipulation_task(n_samples=2000, n_objects=6, seq_len=25),
        n_trials=3,
        epochs=15
    )
    
    # Task 4: Even more complex (8 objects, 30 steps)
    print("\n" + "="*60)
    print("Task 4: Extreme Complexity (8 objects, 30 steps)")
    print("="*60)
    all_results["extreme_complexity_8obj_30step"] = run_experiment(
        "Extreme Complexity (8 objects, 30 steps)",
        lambda: generate_multi_step_manipulation_task(n_samples=2000, n_objects=8, seq_len=30),
        n_trials=3,
        epochs=15
    )
    
    # Summary and analysis
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    summary = {}
    mlp_baseline = "MLP-128"
    
    for task_name, task_results in all_results.items():
        print(f"\n{task_name.replace('_', ' ').title()}:")
        mlp_mse = task_results[mlp_baseline]["mean_mse"]
        
        for model_name, model_results in task_results.items():
            if model_name == mlp_baseline:
                continue
                
            mse = model_results["mean_mse"]
            diff_pct = ((mse - mlp_mse) / mlp_mse) * 100
            print(f"  {model_name}: MSE={mse:.6f} ({diff_pct:+.1f}% vs MLP-128)")
            
            if model_name not in summary:
                summary[model_name] = []
            summary[model_name].append(diff_pct)
    
    # Compute average improvement across tasks
    print("\n" + "="*60)
    print("AVERAGE IMPROVEMENT ACROSS TASKS")
    print("="*60)
    
    avg_improvements = {}
    for model_name, diffs in summary.items():
        avg_improvement = np.mean(diffs)
        std_improvement = np.std(diffs)
        avg_improvements[model_name] = {
            "mean": avg_improvement,
            "std": std_improvement
        }
        print(f"{model_name}: {avg_improvement:.1f}% ± {std_improvement:.1f}%")
    
    # Save results
    output = {
        "experiment_id": "H1.439",
        "description": "GraphCG Scaling: Does advantage increase with task complexity?",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_trials": 3,
            "epochs": 15,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "tasks": [
                "complex_relational_6obj",
                "complex_compositional_6obj_20step", 
                "multi_step_manipulation_6obj_25step",
                "extreme_complexity_8obj_30step"
            ]
        },
        "results": all_results,
        "summary": summary,
        "avg_improvements": avg_improvements,
        "key_findings": {
            "mlp_baseline": mlp_baseline,
            "best_model": min(avg_improvements.items(), key=lambda x: x[1]["mean"])[0] if avg_improvements else "N/A",
            "best_improvement": min([v["mean"] for v in avg_improvements.values()]) if avg_improvements else 0,
            "complexity_trend": "To be analyzed"  # Will fill after results
        }
    }
    
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Determine conclusion
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if avg_improvements:
        best_avg = min([v["mean"] for v in avg_improvements.values()])
        
        # Check if GraphCG shows increasing advantage with complexity
        graphcg_128_3p = avg_improvements.get("GraphCG-128-3p", {"mean": 0})
        graphcg_128_6p = avg_improvements.get("GraphCG-128-6p", {"mean": 0})
        graphcg_256_3p = avg_improvements.get("GraphCG-256-3p", {"mean": 0})
        
        # Get per-task improvements for trend analysis
        task_names = list(all_results.keys())
        improvements_by_complexity = []
        
        for model_name in ["GraphCG-128-3p", "GraphCG-128-6p", "GraphCG-256-3p"]:
            if model_name in summary:
                model_diffs = summary[model_name]
                improvements_by_complexity.append({
                    "model": model_name,
                    "diffs": model_diffs,
                    "trend": np.polyfit(range(len(model_diffs)), model_diffs, 1)[0]  # slope
                })
        
        print(f"\nBest average improvement: {best_avg:.1f}%")
        
        if best_avg < -5:  # GraphCG outperforms MLP by at least 5%
            print("SUPPORTED: GraphCG outperforms MLP on complex tasks.")
            
            # Check if advantage increases with complexity
            if improvements_by_complexity:
                avg_slope = np.mean([imp["trend"] for imp in improvements_by_complexity])
                if avg_slope < -1:  # Negative slope means improvement gets better with complexity
                    print(f"STRONGLY SUPPORTED: GraphCG advantage increases with task complexity (slope: {avg_slope:.2f}%/task)")
                else:
                    print(f"PARTIALLY SUPPORTED: GraphCG outperforms but advantage doesn't clearly increase with complexity")
        else:
            print("INCONCLUSIVE: GraphCG doesn't show clear advantage on complex tasks.")
    else:
        print("ERROR: No results to analyze.")
    
    return output

if __name__ == "__main__":
    main()