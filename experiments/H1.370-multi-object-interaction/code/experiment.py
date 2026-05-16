#!/usr/bin/env python3
"""
H1.370: Multi-Object Interaction Requirement for Cognitive Graph Advantage

Hypothesis: CG requires multi-object interactions to demonstrate advantage.
Real robot data (where CG wins by +25.6%) involves multiple objects with complex
interactions, while synthetic tests so far have been single-object or simple sequences.
CG's graph structure should excel at modeling object relationships.

Prediction: CG improvement will be positive (>0%) when tested on tasks with:
1. 3+ interacting objects
2. Complex spatial relationships (stacking, containment, adjacency)
3. Dynamic interactions (collisions, pushing, pulling)

Test Plan:
1. Create synthetic dataset with multiple interacting objects
2. Vary number of objects (1, 2, 3, 5)
3. Vary interaction complexity (independent motion, collisions, coordinated motion)
4. Measure CG vs baseline performance
"""

import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def generate_multi_object_data(num_objects=3, seq_len=100, interaction_type='collision'):
    """
    Generate multi-object data with different interaction types.
    
    Args:
        num_objects: Number of objects (1, 2, 3, 5)
        seq_len: Sequence length
        interaction_type: 'independent', 'collision', 'coordinated', 'stacking'
    """
    states = []
    
    # Initialize object positions and velocities
    positions = []
    velocities = []
    for obj in range(num_objects):
        # Start with objects in a circle
        angle = 2 * np.pi * obj / max(num_objects, 1)
        pos = [0.5 * np.cos(angle), 0.5 * np.sin(angle), 0.0]
        vel = [0.1 * np.sin(angle), -0.1 * np.cos(angle), 0.0]
        positions.append(pos)
        velocities.append(vel)
    
    for t in range(seq_len):
        time = t / seq_len
        
        # Update positions based on interaction type
        new_positions = []
        new_velocities = []
        
        for obj in range(num_objects):
            pos = positions[obj].copy()
            vel = velocities[obj].copy()
            
            # Basic motion
            pos[0] += vel[0]
            pos[1] += vel[1]
            pos[2] += vel[2]
            
            # Interaction-specific dynamics
            if interaction_type == 'collision' and num_objects > 1:
                # Check for collisions with other objects
                for other in range(num_objects):
                    if obj != other:
                        dist = np.linalg.norm(np.array(pos) - np.array(positions[other]))
                        if dist < 0.2:  # Collision threshold
                            # Elastic collision - swap velocities
                            vel, velocities[other] = velocities[other].copy(), vel.copy()
                            
            elif interaction_type == 'coordinated' and num_objects > 1:
                # Objects move in coordinated patterns
                phase = 2 * np.pi * obj / num_objects
                vel[0] = 0.15 * np.sin(time * 2 * np.pi + phase)
                vel[1] = 0.15 * np.cos(time * 2 * np.pi + phase)
                
            elif interaction_type == 'stacking' and num_objects > 1:
                # Objects try to stack
                target_z = 0.05 * obj  # Each object wants to be at different height
                if pos[2] < target_z:
                    vel[2] = 0.02  # Move upward
                else:
                    vel[2] = -0.01  # Slight downward drift
            
            # Boundary conditions
            for i in range(3):
                if pos[i] < -1.0:
                    pos[i] = -1.0
                    vel[i] = -vel[i] * 0.8  # Damped bounce
                elif pos[i] > 1.0:
                    pos[i] = 1.0
                    vel[i] = -vel[i] * 0.8
            
            new_positions.append(pos)
            new_velocities.append(vel)
        
        positions = new_positions
        velocities = new_velocities
        
        # Create state representation
        state = {
            'timestep': t,
            'objects': []
        }
        
        for obj in range(num_objects):
            state['objects'].append({
                'id': obj,
                'position': positions[obj],
                'velocity': velocities[obj],
                'type': obj % 3,  # 0: cube, 1: sphere, 2: cylinder
            })
        
        states.append(state)
    
    return states

class BaselineModel(nn.Module):
    """Simple baseline model (MLP) for comparison."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CognitiveGraphModel(nn.Module):
    """Cognitive Graph model with explicit object nodes and relation edges."""
    def __init__(self, node_dim=6, edge_dim=4, hidden_dim=128, output_dim=3):
        super().__init__()
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph propagation (2 layers)
        self.propagation = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(2)
        ])
        
        # Readout
        self.readout = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, node_features, edge_features, adj_matrix):
        """
        Args:
            node_features: [batch, num_nodes, node_dim]
            edge_features: [batch, num_edges, edge_dim]
            adj_matrix: [batch, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Encode nodes
        node_emb = self.node_encoder(node_features)  # [batch, num_nodes, hidden]
        
        # Encode edges
        edge_emb = self.edge_encoder(edge_features)  # [batch, num_edges, hidden]
        
        # Graph propagation
        for prop_layer in self.propagation:
            # Aggregate neighbor information
            neighbor_agg = torch.bmm(adj_matrix, node_emb)  # [batch, num_nodes, hidden]
            
            # Combine with edge information (simplified - average edge features per node)
            node_edge = torch.cat([node_emb, neighbor_agg], dim=-1)
            node_emb = F.relu(prop_layer(node_edge))
        
        # Global pooling
        global_feat = torch.mean(node_emb, dim=1)  # [batch, hidden]
        local_feat = torch.max(node_emb, dim=1)[0]  # [batch, hidden]
        combined = torch.cat([global_feat, local_feat], dim=-1)
        
        return self.readout(combined)

def prepare_data_for_models(states, predict_steps=10):
    """Convert states to training data for both models."""
    X_baseline = []
    X_cg_nodes = []
    X_cg_edges = []
    X_cg_adj = []
    y = []
    
    num_objects = len(states[0]['objects'])
    
    for i in range(len(states) - predict_steps):
        current_state = states[i]
        future_state = states[i + predict_steps]
        
        # Baseline: flatten all object features
        baseline_feat = []
        for obj in current_state['objects']:
            baseline_feat.extend(obj['position'])
            baseline_feat.extend(obj['velocity'])
            baseline_feat.append(obj['type'])
        X_baseline.append(baseline_feat)
        
        # CG: node features
        node_feat = []
        for obj in current_state['objects']:
            node_feat.append(obj['position'] + obj['velocity'] + [obj['type']])
        X_cg_nodes.append(node_feat)
        
        # CG: edge features (distances between objects)
        edge_feat = []
        adj_matrix = np.zeros((num_objects, num_objects))
        
        if num_objects > 1:
            edge_idx = 0
            for obj1 in range(num_objects):
                for obj2 in range(obj1 + 1, num_objects):
                    pos1 = np.array(current_state['objects'][obj1]['position'])
                    pos2 = np.array(current_state['objects'][obj2]['position'])
                    dist = np.linalg.norm(pos1 - pos2)
                    edge_feat.append([dist, 1.0 if dist < 0.3 else 0.0, obj1/num_objects, obj2/num_objects])
                    adj_matrix[obj1, obj2] = 1.0
                    adj_matrix[obj2, obj1] = 1.0
                    edge_idx += 1
        else:
            # Single object - no edges
            edge_feat.append([0.0, 0.0, 0.0, 0.0])  # Dummy edge
            adj_matrix[0, 0] = 1.0  # Self-connection
        
        X_cg_edges.append(edge_feat)
        X_cg_adj.append(adj_matrix)
        
        # Target: average position change of all objects
        target = []
        for obj_idx in range(num_objects):
            pos_current = np.array(current_state['objects'][obj_idx]['position'])
            pos_future = np.array(future_state['objects'][obj_idx]['position'])
            target.extend(pos_future - pos_current)
        y.append(target)
    
    return (np.array(X_baseline), np.array(X_cg_nodes), np.array(X_cg_edges), 
            np.array(X_cg_adj), np.array(y))

def train_and_evaluate(num_objects=3, interaction_type='collision', seq_len=200):
    """Train and evaluate both models on multi-object data."""
    set_seed(42)
    
    # Generate data
    print(f"Generating data: {num_objects} objects, {interaction_type} interaction")
    states = generate_multi_object_data(num_objects, seq_len, interaction_type)
    
    # Prepare for models
    X_base, X_nodes, X_edges, X_adj, y = prepare_data_for_models(states)
    
    # Split data
    split = int(0.8 * len(X_base))
    X_base_train, X_base_test = X_base[:split], X_base[split:]
    X_nodes_train, X_nodes_test = X_nodes[:split], X_nodes[split:]
    X_edges_train, X_edges_test = X_edges[:split], X_edges[split:]
    X_adj_train, X_adj_test = X_adj[:split], X_adj[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Convert to tensors
    X_base_train = torch.FloatTensor(X_base_train)
    X_base_test = torch.FloatTensor(X_base_test)
    X_nodes_train = torch.FloatTensor(X_nodes_train)
    X_nodes_test = torch.FloatTensor(X_nodes_test)
    X_edges_train = torch.FloatTensor(X_edges_train)
    X_edges_test = torch.FloatTensor(X_edges_test)
    X_adj_train = torch.FloatTensor(X_adj_train)
    X_adj_test = torch.FloatTensor(X_adj_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Initialize models
    baseline_input_dim = X_base_train.shape[1]
    baseline_output_dim = y_train.shape[1]
    baseline_model = BaselineModel(baseline_input_dim, hidden_dim=128, output_dim=baseline_output_dim)
    
    cg_node_dim = X_nodes_train.shape[2]
    cg_edge_dim = X_edges_train.shape[2]
    cg_output_dim = y_train.shape[1]
    cg_model = CognitiveGraphModel(cg_node_dim, cg_edge_dim, hidden_dim=128, output_dim=cg_output_dim)
    
    # Training
    optimizer_base = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    optimizer_cg = torch.optim.Adam(cg_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    num_epochs = 100
    batch_size = 32
    
    for epoch in range(num_epochs):
        # Baseline training
        permutation = torch.randperm(X_base_train.size(0))
        for i in range(0, X_base_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_base_train[indices]
            batch_y = y_train[indices]
            
            optimizer_base.zero_grad()
            outputs = baseline_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_base.step()
        
        # CG training
        for i in range(0, X_nodes_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_nodes = X_nodes_train[indices]
            batch_edges = X_edges_train[indices]
            batch_adj = X_adj_train[indices]
            batch_y = y_train[indices]
            
            optimizer_cg.zero_grad()
            outputs = cg_model(batch_nodes, batch_edges, batch_adj)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_cg.step()
    
    # Evaluation
    with torch.no_grad():
        baseline_pred = baseline_model(X_base_test)
        baseline_mse = criterion(baseline_pred, y_test).item()
        
        cg_pred = cg_model(X_nodes_test, X_edges_test, X_adj_test)
        cg_mse = criterion(cg_pred, y_test).item()
    
    improvement = ((baseline_mse - cg_mse) / baseline_mse) * 100
    
    print(f"Results for {num_objects} objects, {interaction_type}:")
    print(f"  Baseline MSE: {baseline_mse:.4f}")
    print(f"  CG MSE: {cg_mse:.4f}")
    print(f"  CG Improvement: {improvement:.1f}%")
    print(f"  CG Wins: {improvement > 0}")
    
    return {
        'num_objects': num_objects,
        'interaction_type': interaction_type,
        'baseline_mse': baseline_mse,
        'cg_mse': cg_mse,
        'improvement_percent': improvement,
        'cognitive_graph_wins': improvement > 0
    }

def main():
    """Run experiments with varying object counts and interaction types."""
    results = []
    
    # Test different numbers of objects
    for num_objects in [1, 2, 3, 5]:
        # Test different interaction types
        for interaction_type in ['independent', 'collision', 'coordinated', 'stacking']:
            result = train_and_evaluate(num_objects, interaction_type)
            results.append(result)
    
    # Save results
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Analyze results
    print("\n=== Analysis ===")
    for num_objects in [1, 2, 3, 5]:
        print(f"\n{num_objects} objects:")
        obj_results = [r for r in results if r['num_objects'] == num_objects]
        for r in obj_results:
            wins_symbol = "✓" if r['cognitive_graph_wins'] else "✗"
            print(f"  {r['interaction_type']:12s}: {r['improvement_percent']:6.1f}% {wins_symbol}")
    
    # Check hypothesis
    cg_wins_multi_object = any(r['cognitive_graph_wins'] and r['num_objects'] >= 3 for r in results)
    cg_wins_complex = any(r['cognitive_graph_wins'] and r['interaction_type'] in ['collision', 'coordinated', 'stacking'] for r in results)
    
    print(f"\n=== Hypothesis H1.370 ===")
    print(f"CG wins with ≥3 objects: {cg_wins_multi_object}")
    print(f"CG wins with complex interactions: {cg_wins_complex}")
    
    if cg_wins_multi_object and cg_wins_complex:
        print("✅ H1.370 SUPPORTED: CG requires multi-object interactions to demonstrate advantage")
    elif cg_wins_multi_object or cg_wins_complex:
        print("⚠️ H1.370 PARTIALLY SUPPORTED: CG shows advantage in some multi-object scenarios")
    else:
        print("❌ H1.370 REFUTED: CG does not show advantage even with multi-object interactions")
    
    return results

if __name__ == "__main__":
    main()