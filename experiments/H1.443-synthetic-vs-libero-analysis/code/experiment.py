#!/usr/bin/env python3
"""
H1.443 - Synthetic vs LIBERO Task Discrepancy Analysis

CRITICAL QUESTION: Why does GraphCG succeed on synthetic tasks (+29.1% H1.441) 
but fail on LIBERO tasks (-39.8% H1.442)?

Hypothesis: The task type (transformation prediction vs action prediction) is 
the key factor, not the data structure.

Experiment Design:
1. Test GraphCG vs MLP on "LIBERO-style transformation prediction" 
   (predict next state from current state + action)
2. Test GraphCG vs MLP on "synthetic-style action prediction" 
   (predict action from state sequence)
3. Compare results to isolate which factor drives the performance difference

If task type is key:
- GraphCG should excel at transformation prediction regardless of data source
- GraphCG should struggle with action prediction regardless of data source

If data structure is key:
- GraphCG should excel on synthetic data regardless of task type
- GraphCG should struggle on LIBERO data regardless of task type
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
import pickle
from collections import defaultdict

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Model Definitions
# ============================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline matching GraphCG parameter count."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
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


class GraphCG(nn.Module):
    """
    Graph-style Cognitive Graph with message passing.
    Uses explicit graph structure for relational reasoning.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=7, n_passes=3, n_nodes=6):
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
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project to nodes
        nodes = self.node_proj(x).view(batch_size, self.n_nodes, self.hidden_dim)
        
        # Message passing
        for _ in range(self.n_passes):
            # Compute all pairwise messages
            messages = []
            for i in range(self.n_nodes):
                msg_i = []
                for j in range(self.n_nodes):
                    if i != j:
                        # Message from j to i
                        msg = torch.cat([nodes[:, i], nodes[:, j]], dim=-1)
                        msg = self.message_mlp(msg)
                        msg_i.append(msg)
                # Aggregate messages
                if msg_i:
                    agg_msg = torch.stack(msg_i, dim=1).mean(dim=1)
                else:
                    agg_msg = torch.zeros(batch_size, self.hidden_dim, device=x.device)
                messages.append(agg_msg)
            
            # Update nodes with residual
            new_nodes = []
            for i in range(self.n_nodes):
                update_input = torch.cat([nodes[:, i], messages[i]], dim=-1)
                update = self.update_mlp(update_input)
                new_nodes.append(nodes[:, i] + update)
            nodes = torch.stack(new_nodes, dim=1)
        
        # Aggregate and output
        graph_out = nodes.view(batch_size, -1)
        return self.output_proj(graph_out)


# ============================================================
# Task Generators
# ============================================================

def generate_synthetic_transformation_task(n_samples, n_objects=4, n_steps=5):
    """
    Synthetic transformation prediction task (like H1.441).
    Predict final state after transformation sequence.
    
    Input: initial positions + transformation parameters
    Output: final positions
    """
    # Object positions (x, y, z) for each object
    positions = np.random.randn(n_samples, n_objects, 3).astype(np.float32)
    
    # Transformation parameters (rotation, translation)
    transforms = np.random.randn(n_samples, n_steps, 6).astype(np.float32)
    
    # Apply transformations to get final positions
    final_positions = positions.copy()
    for s in range(n_steps):
        # Simple transformation: add translation, apply rotation
        translation = transforms[:, s, :3]
        rotation = transforms[:, s, 3:]
        
        # Apply translation
        final_positions = final_positions + translation[:, np.newaxis, :]
        
        # Apply rotation (simplified - rotate around z-axis)
        for b in range(n_samples):
            angle = np.linalg.norm(rotation[b])
            if angle > 0:
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                for o in range(n_objects):
                    x_rot = final_positions[b, o, 0] * cos_a - final_positions[b, o, 1] * sin_a
                    y_rot = final_positions[b, o, 0] * sin_a + final_positions[b, o, 1] * cos_a
                    final_positions[b, o, 0] = x_rot
                    final_positions[b, o, 1] = y_rot
    
    # Flatten input: initial positions + transforms
    X = np.concatenate([
        positions.reshape(n_samples, -1),
        transforms.reshape(n_samples, -1)
    ], axis=1)
    
    # Output: final positions
    y = final_positions.reshape(n_samples, -1)
    
    return X, y


def generate_synthetic_action_task(n_samples, n_objects=4, n_steps=5):
    """
    Synthetic action prediction task (LIBERO-style but with synthetic data).
    Predict action from state sequence.
    
    Input: sequence of states
    Output: action to take
    """
    # State sequence: positions over time
    states = np.random.randn(n_samples, n_steps, n_objects, 3).astype(np.float32)
    
    # Action: delta between last two states + noise
    actions = states[:, -1] - states[:, -2] + 0.1 * np.random.randn(n_samples, n_objects, 3).astype(np.float32)
    
    # Flatten input: state sequence
    X = states.reshape(n_samples, -1)
    
    # Output: action
    y = actions.reshape(n_samples, -1)
    
    return X, y


def generate_libero_transformation_task(n_samples, n_objects=4):
    """
    LIBERO-style transformation prediction task.
    Predict next state from current state + action.
    
    Input: current state + action
    Output: next state
    """
    # State: object positions + gripper state
    state_dim = n_objects * 3 + 7  # positions + gripper pose
    action_dim = 7  # gripper action
    
    # Current state
    current_state = np.random.randn(n_samples, state_dim).astype(np.float32) * 0.5
    
    # Action
    action = np.random.randn(n_samples, action_dim).astype(np.float32) * 0.3
    
    # Next state: current state + action effect + noise
    # Action affects gripper and nearby objects
    action_effect = np.zeros((n_samples, state_dim), dtype=np.float32)
    action_effect[:, -7:] = action  # Gripper moves
    # Objects affected by gripper motion (simplified)
    for i in range(n_objects):
        action_effect[:, i*3:(i+1)*3] = action[:, :3] * 0.1  # All objects slightly affected
    
    next_state = current_state + action_effect + 0.05 * np.random.randn(n_samples, state_dim).astype(np.float32)
    
    # Input: current state + action
    X = np.concatenate([current_state, action], axis=1)
    
    # Output: next state
    y = next_state
    
    return X, y


def generate_libero_action_task(n_samples, n_objects=4):
    """
    LIBERO-style action prediction task (like H1.442).
    Predict action from state sequence.
    
    Input: state sequence
    Output: action
    """
    # State: object positions + gripper state
    state_dim = n_objects * 3 + 7
    
    # State sequence (3 timesteps)
    states = np.random.randn(n_samples, 3, state_dim).astype(np.float32) * 0.5
    
    # Action: delta to move toward goal + noise
    goal = np.random.randn(n_samples, state_dim).astype(np.float32) * 0.3
    action = (goal - states[:, -1])[:, -7:] * 0.5 + 0.1 * np.random.randn(n_samples, 7).astype(np.float32)
    
    # Input: state sequence
    X = states.reshape(n_samples, -1)
    
    # Output: action
    y = action
    
    return X, y


# ============================================================
# Training Functions
# ============================================================

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=64, lr=3e-4):
    """Train model and return test MSE."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t)
        mse = criterion(pred, y_test_t).item()
    
    return mse


def run_comparison(X, y, task_name, n_trials=3, epochs=50, hidden_dim=64, n_passes=3, n_nodes=6):
    """Run MLP vs GraphCG comparison on a task."""
    results = {
        'task': task_name,
        'n_samples': X.shape[0],
        'input_dim': X.shape[1],
        'output_dim': y.shape[1],
        'trials': []
    }
    
    for trial in range(n_trials):
        # Split data
        n_train = int(0.7 * X.shape[0])
        idx = np.random.permutation(X.shape[0])
        X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
        y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]
        
        # Train MLP
        mlp = BaselineMLP(X.shape[1], hidden_dim, y.shape[1])
        mlp_mse = train_model(mlp, X_train, y_train, X_test, y_test, epochs)
        
        # Train GraphCG
        graphcg = GraphCG(X.shape[1], hidden_dim, y.shape[1], n_passes, n_nodes)
        graphcg_mse = train_model(graphcg, X_train, y_train, X_test, y_test, epochs)
        
        improvement = (mlp_mse - graphcg_mse) / mlp_mse * 100
        
        results['trials'].append({
            'trial': trial + 1,
            'mlp_mse': mlp_mse,
            'graphcg_mse': graphcg_mse,
            'improvement': improvement
        })
    
    # Compute averages
    results['avg_mlp_mse'] = np.mean([t['mlp_mse'] for t in results['trials']])
    results['avg_graphcg_mse'] = np.mean([t['graphcg_mse'] for t in results['trials']])
    results['avg_improvement'] = np.mean([t['improvement'] for t in results['trials']])
    
    return results


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 80)
    print("H1.443 - Synthetic vs LIBERO Task Discrepancy Analysis")
    print("=" * 80)
    
    all_results = {
        'experiment_id': 'H1.443',
        'description': 'Analyze why GraphCG succeeds on synthetic but fails on LIBERO',
        'config': {
            'n_trials': 3,
            'epochs': 50,
            'batch_size': 64,
            'hidden_dim': 64,
            'n_passes': 3,
            'n_nodes': 6
        },
        'results': {}
    }
    
    n_samples = 400
    n_objects = 4
    n_steps = 5
    
    # Task 1: Synthetic Transformation Prediction (like H1.441)
    print("\n" + "=" * 60)
    print("Task 1: Synthetic Transformation Prediction")
    print("Predict final state after transformation sequence")
    print("=" * 60)
    
    X, y = generate_synthetic_transformation_task(n_samples, n_objects, n_steps)
    results_1 = run_comparison(X, y, 'synthetic_transformation')
    all_results['results']['synthetic_transformation'] = results_1
    
    print(f"MLP MSE: {results_1['avg_mlp_mse']:.4f}")
    print(f"GraphCG MSE: {results_1['avg_graphcg_mse']:.4f}")
    print(f"Improvement: {results_1['avg_improvement']:+.1f}%")
    
    # Task 2: Synthetic Action Prediction (LIBERO-style with synthetic data)
    print("\n" + "=" * 60)
    print("Task 2: Synthetic Action Prediction")
    print("Predict action from state sequence (LIBERO-style with synthetic data)")
    print("=" * 60)
    
    X, y = generate_synthetic_action_task(n_samples, n_objects, n_steps)
    results_2 = run_comparison(X, y, 'synthetic_action')
    all_results['results']['synthetic_action'] = results_2
    
    print(f"MLP MSE: {results_2['avg_mlp_mse']:.4f}")
    print(f"GraphCG MSE: {results_2['avg_graphcg_mse']:.4f}")
    print(f"Improvement: {results_2['avg_improvement']:+.1f}%")
    
    # Task 3: LIBERO Transformation Prediction (transformation task with LIBERO-style data)
    print("\n" + "=" * 60)
    print("Task 3: LIBERO Transformation Prediction")
    print("Predict next state from current state + action")
    print("=" * 60)
    
    X, y = generate_libero_transformation_task(n_samples, n_objects)
    results_3 = run_comparison(X, y, 'libero_transformation')
    all_results['results']['libero_transformation'] = results_3
    
    print(f"MLP MSE: {results_3['avg_mlp_mse']:.4f}")
    print(f"GraphCG MSE: {results_3['avg_graphcg_mse']:.4f}")
    print(f"Improvement: {results_3['avg_improvement']:+.1f}%")
    
    # Task 4: LIBERO Action Prediction (like H1.442)
    print("\n" + "=" * 60)
    print("Task 4: LIBERO Action Prediction")
    print("Predict action from state sequence (H1.442 style)")
    print("=" * 60)
    
    X, y = generate_libero_action_task(n_samples, n_objects)
    results_4 = run_comparison(X, y, 'libero_action')
    all_results['results']['libero_action'] = results_4
    
    print(f"MLP MSE: {results_4['avg_mlp_mse']:.4f}")
    print(f"GraphCG MSE: {results_4['avg_graphcg_mse']:.4f}")
    print(f"Improvement: {results_4['avg_improvement']:+.1f}%")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'Task':<35} {'MLP MSE':>10} {'GraphCG MSE':>12} {'Improvement':>12}")
    print("-" * 60)
    for task_name, task_results in all_results['results'].items():
        print(f"{task_name:<35} {task_results['avg_mlp_mse']:>10.4f} {task_results['avg_graphcg_mse']:>12.4f} {task_results['avg_improvement']:>+11.1f}%")
    print("-" * 60)
    
    # Determine key factor
    synth_trans = all_results['results']['synthetic_transformation']['avg_improvement']
    synth_action = all_results['results']['synthetic_action']['avg_improvement']
    libero_trans = all_results['results']['libero_transformation']['avg_improvement']
    libero_action = all_results['results']['libero_action']['avg_improvement']
    
    print("\nKey Factor Analysis:")
    print("-" * 60)
    
    # If task type is key: transformation tasks should show similar patterns
    trans_avg = (synth_trans + libero_trans) / 2
    action_avg = (synth_action + libero_action) / 2
    
    # If data source is key: synthetic tasks should show similar patterns
    synth_avg = (synth_trans + synth_action) / 2
    libero_avg = (libero_trans + libero_action) / 2
    
    print(f"Transformation tasks avg improvement: {trans_avg:+.1f}%")
    print(f"Action tasks avg improvement: {action_avg:+.1f}%")
    print(f"Synthetic data avg improvement: {synth_avg:+.1f}%")
    print(f"LIBERO data avg improvement: {libero_avg:+.1f}%")
    
    task_type_diff = abs(trans_avg - action_avg)
    data_source_diff = abs(synth_avg - libero_avg)
    
    print(f"\nTask type difference: {task_type_diff:.1f} percentage points")
    print(f"Data source difference: {data_source_diff:.1f} percentage points")
    
    if task_type_diff > data_source_diff:
        conclusion = "TASK_TYPE_DOMINANT"
        print("\n>>> CONCLUSION: Task type is the dominant factor")
        print("    GraphCG excels at transformation prediction regardless of data source")
        print("    GraphCG struggles with action prediction regardless of data source")
    else:
        conclusion = "DATA_SOURCE_DOMINANT"
        print("\n>>> CONCLUSION: Data source is the dominant factor")
        print("    GraphCG excels on synthetic data regardless of task type")
        print("    GraphCG struggles on LIBERO data regardless of task type")
    
    all_results['analysis'] = {
        'trans_avg_improvement': trans_avg,
        'action_avg_improvement': action_avg,
        'synth_avg_improvement': synth_avg,
        'libero_avg_improvement': libero_avg,
        'task_type_diff': task_type_diff,
        'data_source_diff': data_source_diff,
        'conclusion': conclusion
    }
    
    # Save results
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()