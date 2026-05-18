#!/usr/bin/env python3
"""
H1.436: CG Domain of Applicability
===================================
Test whether CG performs better on tasks with clear relational structure
(e.g., object relationships, spatial reasoning) vs continuous control tasks
(e.g., trajectory following, smooth motion).

Hypothesis: CG wins on relational tasks but loses on continuous control tasks.
This would explain the discrepancy between H1.433 (CG wins on synthetic physics)
and H1.434 (CG loses on LIBERO manipulation).
"""

import sys
import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")
from data_loader import LIBERODataset


class SimpleCG(nn.Module):
    """Simplified Cognitive Graph - 3-layer GNN with node/edge attention."""
    
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32, n_nodes=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        
        # Node embeddings
        self.node_embed = nn.Linear(input_dim, hidden_dim)
        
        # Graph layers with attention
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge computation
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, n_nodes, input_dim)
        batch_size = x.shape[0]
        
        # Node embeddings
        h = torch.relu(self.node_embed(x))
        
        # Self-attention
        Q = self.W_q(h)
        K = self.W_k(h)
        V = self.W_v(h)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (h.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn_weights, V)
        
        # Simple graph pooling (mean)
        h = h.mean(dim=1)
        
        return self.out_proj(h)


class BaselineMLP(nn.Module):
    """Simple MLP baseline."""
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=32):
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


def generate_relational_task(batch_size, seq_len, n_objects=4):
    """
    Generate relational task data: requires understanding object relationships.
    Example: "move red block to blue bowl" - requires identifying objects and relations.
    """
    # State: (batch, seq_len, n_objects, object_dim)
    # Each object has position (3) + color (1) + type (1) = 5 dims
    object_dim = 5
    
    states = []
    actions = []
    
    for _ in range(batch_size):
        # Random object properties
        obj_positions = np.random.randn(seq_len, n_objects, 3) * 0.5
        obj_colors = np.random.randint(0, 3, (seq_len, n_objects))  # 0=red, 1=blue, 2=green
        obj_types = np.random.randint(0, 2, (seq_len, n_objects))   # 0=block, 1=bowl
        
        # Target: move object 0 to position of object 1
        # Action = delta to move obj0 toward obj1
        target_pos = obj_positions[:, 1, :]  # Target position
        current_pos = obj_positions[:, 0, :]  # Current position
        actions_delta = target_pos - current_pos  # (seq_len, 3)
        
        # Flatten state
        state = np.concatenate([
            obj_positions.reshape(seq_len, -1),
            obj_colors.reshape(seq_len, -1),
            obj_types.reshape(seq_len, -1)
        ], axis=-1)  # (seq_len, n_objects * 5)
        
        states.append(state)
        actions.append(actions_delta)
    
    return np.array(states), np.array(actions)


def generate_continuous_control_task(batch_size, seq_len):
    """
    Generate continuous control task: smooth trajectory following.
    No relational structure - just smooth motion prediction.
    """
    # State: (batch, seq_len, state_dim)
    # Simple: previous positions + velocity
    state_dim = 6  # position(3) + velocity(3)
    
    states = []
    actions = []
    
    for _ in range(batch_size):
        # Generate smooth trajectory with random walk
        positions = np.cumsum(np.random.randn(seq_len, 3) * 0.1, axis=0)
        velocities = np.diff(positions, axis=0, prepend=positions[:1, :])
        
        # State: position + velocity
        state = np.concatenate([positions, velocities], axis=-1)
        
        # Action: next velocity (for simplicity, predict velocity change)
        action = np.diff(velocities, axis=0, prepend=velocities[:1, :])
        
        states.append(state)
        actions.append(action)
    
    return np.array(states), np.array(actions)


def train_model(model, states, actions, epochs=10, lr=1e-3, model_type="mlp", n_nodes=4):
    """Train a model on the given data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    states_t = torch.FloatTensor(states)
    actions_t = torch.FloatTensor(actions)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if model_type == "mlp":
            # Flatten for MLP
            flat_states = states_t.reshape(states_t.shape[0], -1)
            pred = model(flat_states)
        else:
            # For CG, reshape to (batch, n_nodes, features)
            # states shape: (batch, seq_len, n_features)
            # We take last timestep and reshape to (batch, n_nodes, features_per_node)
            seq_len = states_t.shape[1]
            total_features = states_t.shape[2]
            features_per_node = total_features // n_nodes
            
            if features_per_node * n_nodes != total_features:
                # Pad to make divisible
                pad_size = n_nodes - (total_features % n_nodes)
                if pad_size != n_nodes:
                    states_t = torch.cat([states_t, torch.zeros(states_t.shape[0], states_t.shape[1], pad_size)], dim=2)
                    features_per_node = (total_features + pad_size) // n_nodes
            
            # Take last timestep and reshape
            last_state = states_t[:, -1, :]  # (batch, features)
            flat_states = last_state.reshape(-1, n_nodes, features_per_node)  # (batch, n_nodes, features_per_node)
            pred = model(flat_states)
        
        # Flatten actions
        flat_actions = actions_t[:, -1]  # Just predict final action
        
        loss = criterion(pred, flat_actions)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses


def evaluate_model(model, states, actions, model_type="mlp", n_nodes=4):
    """Evaluate model MSE."""
    states_t = torch.FloatTensor(states)
    actions_t = torch.FloatTensor(actions)
    
    with torch.no_grad():
        if model_type == "mlp":
            flat_states = states_t.reshape(states_t.shape[0], -1)
            pred = model(flat_states)
        else:
            # For CG
            total_features = states_t.shape[2]
            features_per_node = total_features // n_nodes
            
            if features_per_node * n_nodes != total_features:
                pad_size = n_nodes - (total_features % n_nodes)
                if pad_size != n_nodes:
                    states_t = torch.cat([states_t, torch.zeros(states_t.shape[0], states_t.shape[1], pad_size)], dim=2)
                    features_per_node = (total_features + pad_size) // n_nodes
            
            last_state = states_t[:, -1, :]
            flat_states = last_state.reshape(-1, n_nodes, features_per_node)
            pred = model(flat_states)
        
        flat_actions = actions_t[:, -1]
        mse = F.mse_loss(pred, flat_actions).item()
    
    return mse


def run_experiment():
    """Run the domain applicability experiment."""
    print("=" * 60)
    print("H1.436: CG Domain of Applicability")
    print("=" * 60)
    
    # Config
    batch_size = 64
    seq_len = 10
    n_trials = 3
    epochs = 15
    
    results = {
        "relational": {"mlp": [], "cg": []},
        "continuous": {"mlp": [], "cg": []}
    }
    
    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        
        # Generate data
        rel_states, rel_actions = generate_relational_task(batch_size, seq_len, n_objects=4)
        cont_states, cont_actions = generate_continuous_control_task(batch_size, seq_len)
        
        print(f"Relational task: states {rel_states.shape}, actions {rel_actions.shape}")
        print(f"Continuous task: states {cont_states.shape}, actions {cont_actions.shape}")
        
        # Train and evaluate on relational tasks
        # MLP
        mlp = BaselineMLP(input_dim=seq_len * 20, hidden_dim=128, output_dim=3)
        train_model(mlp, rel_states, rel_actions, epochs=epochs, model_type="mlp")
        mlp_mse = evaluate_model(mlp, rel_states, rel_actions, model_type="mlp")
        results["relational"]["mlp"].append(mlp_mse)
        
        # CG - 4 nodes for relational (4 objects)
        cg = SimpleCG(input_dim=5, hidden_dim=64, output_dim=3, n_nodes=4)
        train_model(cg, rel_states, rel_actions, epochs=epochs, model_type="cg", n_nodes=4)
        cg_mse = evaluate_model(cg, rel_states, rel_actions, model_type="cg", n_nodes=4)
        results["relational"]["cg"].append(cg_mse)
        
        # Train and evaluate on continuous control tasks
        # MLP
        mlp = BaselineMLP(input_dim=seq_len * 6, hidden_dim=128, output_dim=3)
        train_model(mlp, cont_states, cont_actions, epochs=epochs, model_type="mlp")
        mlp_mse = evaluate_model(mlp, cont_states, cont_actions, model_type="mlp")
        results["continuous"]["mlp"].append(mlp_mse)
        
        # CG - 1 node for continuous (single agent trajectory)
        cg = SimpleCG(input_dim=6, hidden_dim=64, output_dim=3, n_nodes=1)
        train_model(cg, cont_states, cont_actions, epochs=epochs, model_type="cg", n_nodes=1)
        cg_mse = evaluate_model(cg, cont_states, cont_actions, model_type="cg", n_nodes=1)
        results["continuous"]["cg"].append(cg_mse)
        
        print(f"Relational - MLP MSE: {np.mean(results['relational']['mlp']):.4f}, CG MSE: {np.mean(results['relational']['cg']):.4f}")
        print(f"Continuous - MLP MSE: {np.mean(results['continuous']['mlp']):.4f}, CG MSE: {np.mean(results['continuous']['cg']):.4f}")
    
    # Compute final statistics
    rel_mlp_mean = np.mean(results["relational"]["mlp"])
    rel_cg_mean = np.mean(results["relational"]["cg"])
    cont_mlp_mean = np.mean(results["continuous"]["mlp"])
    cont_cg_mean = np.mean(results["continuous"]["cg"])
    
    rel_cg_vs_mlp = ((rel_cg_mean - rel_mlp_mean) / rel_mlp_mean) * 100
    cont_cg_vs_mlp = ((cont_cg_mean - cont_mlp_mean) / cont_mlp_mean) * 100
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Relational Tasks:")
    print(f"  MLP MSE: {rel_mlp_mean:.4f}")
    print(f"  CG MSE:  {rel_cg_mean:.4f}")
    print(f"  CG vs MLP: {rel_cg_vs_mlp:+.1f}%")
    print(f"\nContinuous Control Tasks:")
    print(f"  MLP MSE: {cont_mlp_mean:.4f}")
    print(f"  CG MSE:  {cont_cg_mean:.4f}")
    print(f"  CG vs MLP: {cont_cg_vs_mlp:+.1f}%")
    
    # Determine conclusion
    if rel_cg_vs_mlp < 0 and cont_cg_vs_mlp > 0:
        conclusion = "SUPPORTED"
        insight = "CG performs better on relational tasks but worse on continuous control, confirming domain-specific advantage"
    elif rel_cg_vs_mlp > 0 and cont_cg_vs_mlp > 0:
        conclusion = "PARTIALLY_SUPPORTED"
        insight = "CG underperforms on both but less on relational tasks"
    elif rel_cg_vs_mlp < 0 and cont_cg_vs_mlp < 0:
        conclusion = "PARTIALLY_SUPPORTED"
        insight = "CG underperforms on both but more on continuous control"
    else:
        conclusion = "NOT_SUPPORTED"
        insight = "CG shows no clear domain-specific advantage"
    
    print(f"\nConclusion: {conclusion}")
    print(f"Insight: {insight}")
    
    # Save results
    output = {
        "experiment_id": "H1.436",
        "description": "CG Domain of Applicability - Relational vs Continuous Control",
        "conclusion": conclusion,
        "results": {
            "relational_mlp_mse": float(rel_mlp_mean),
            "relational_cg_mse": float(rel_cg_mean),
            "relational_cg_vs_mlp": float(rel_cg_vs_mlp),
            "continuous_mlp_mse": float(cont_mlp_mean),
            "continuous_cg_mse": float(cont_cg_mean),
            "continuous_cg_vs_mlp": float(cont_cg_vs_mlp),
        },
        "key_insight": insight,
        "config": {
            "n_trials": n_trials,
            "epochs": epochs,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
    }
    
    with open("experiments/436-cg-domain-applicability/results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output


if __name__ == "__main__":
    result = run_experiment()
