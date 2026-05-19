#!/usr/bin/env python3
"""
H1.440 - Robust GraphCG Scaling Test - Fixed version
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
# Model Definitions (Simplified for debugging)
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
        
        # Simple message passing
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
# Task Generators (Fixed)
# ============================================================

def generate_stable_task_fixed(n_objects, seq_length, n_samples=1000):
    """
    Generate more stable tasks with better conditioning - FIXED VERSION.
    """
    # Each object has position (x, y) and velocity (dx, dy)
    input_dim = n_objects * 4 + seq_length * 2
    
    X_list = []
    y_list = []
    
    for sample_idx in range(n_samples):
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
    
    X_tensor = torch.FloatTensor(np.array(X_list))
    y_tensor = torch.FloatTensor(np.array(y_list))
    
    print(f"Generated dataset: X shape = {X_tensor.shape}, y shape = {y_tensor.shape}")
    return X_tensor, y_tensor

# ============================================================
# Training and Evaluation
# ============================================================

def train_model_simple(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
    """Simple training function."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

def evaluate_model_simple(model, X_test, y_test):
    """Simple evaluation function."""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = F.mse_loss(y_pred, y_test).item()
    return mse

# ============================================================
# Main Experiment (Simplified)
# ============================================================

def run_simple_experiment():
    """Run simplified experiment for debugging."""
    print("=" * 80)
    print("H1.440 - Robust GraphCG Scaling Test (Simplified)")
    print("=" * 80)
    
    # Test with just one complexity level first
    n_objects = 2
    seq_length = 5
    n_samples = 100
    n_trials = 2
    
    print(f"\nTesting: {n_objects} objects, {seq_length} steps")
    print(f"Samples: {n_samples}, Trials: {n_trials}")
    
    mlp_losses = []
    graphcg_losses = []
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Generate data
        np.random.seed(42 + trial * 100)
        torch.manual_seed(42 + trial * 100)
        
        X, y = generate_stable_task_fixed(n_objects, seq_length, n_samples)
        
        # Simple split (80% train, 20% test)
        n_train = int(n_samples * 0.8)
        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Define input dimension
        input_dim = n_objects * 4 + seq_length * 2
        
        # Train MLP
        print("  Training MLP...")
        mlp = BaselineMLP(input_dim=input_dim, hidden_dim=32, output_dim=1)
        mlp_loss = train_model_simple(mlp, X_train, y_train, X_test, y_test, epochs=30)
        mlp_test_loss = evaluate_model_simple(mlp, X_test, y_test)
        mlp_losses.append(mlp_test_loss)
        
        # Train GraphCG
        print("  Training GraphCG...")
        graphcg = SimpleGraphCG(input_dim=input_dim, hidden_dim=32, output_dim=1, 
                               n_passes=2, n_nodes=min(4, n_objects))
        graphcg_loss = train_model_simple(graphcg, X_train, y_train, X_test, y_test, epochs=30)
        graphcg_test_loss = evaluate_model_simple(graphcg, X_test, y_test)
        graphcg_losses.append(graphcg_test_loss)
        
        print(f"    MLP MSE: {mlp_test_loss:.6f}, GraphCG MSE: {graphcg_test_loss:.6f}")
    
    # Calculate statistics
    avg_mlp = statistics.mean(mlp_losses)
    avg_graphcg = statistics.mean(graphcg_losses)
    improvement = ((avg_graphcg - avg_mlp) / avg_mlp) * 100
    
    print(f"\nSummary:")
    print(f"  Average MLP MSE: {avg_mlp:.6f}")
    print(f"  Average GraphCG MSE: {avg_graphcg:.6f}")
    print(f"  Improvement: {improvement:+.2f}%")
    
    return {
        "mlp_losses": mlp_losses,
        "graphcg_losses": graphcg_losses,
        "avg_mlp": avg_mlp,
        "avg_graphcg": avg_graphcg,
        "improvement": improvement
    }

if __name__ == "__main__":
    results = run_simple_experiment()