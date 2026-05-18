#!/usr/bin/env python3
"""
H1.420: Per-Object Cognitive Graph Structure

Hypothesis: CG benefits from finer-grained node structure (per-object nodes 
instead of single physical blob). Per-object CG will match or exceed GraphAttn 
performance on permanence task.

Comparison:
(a) Current 2-node CG (physical + semantic)
(b) Per-object CG with N+1 nodes (N object nodes + 1 semantic node)
(c) Hybrid with object-level physical nodes + unified semantic node
(d) GraphAttn baseline (from H1.419)

Key insight from H1.419: GraphAttn (+5.28%) beat CG (-5.31%) on object permanence.
This suggests object-level graph structure matters for physical reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Physical Grounding Task Generators (from H1.419)
# ============================================================================

def generate_collision_data(n_samples=3000, n_objects=5, seq_len=10):
    """
    Collision prediction: predict whether objects will collide.
    Input: positions and velocities of N objects over seq_len timesteps
    Output: binary collision prediction
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random initial positions (spread out)
        positions = np.random.uniform(-2, 2, (n_objects, 2))
        # Random velocities
        velocities = np.random.uniform(-0.5, 0.5, (n_objects, 2))
        
        # Simulate trajectory
        trajectory = [positions.copy()]
        for t in range(seq_len - 1):
            positions = positions + velocities * 0.1
            trajectory.append(positions.copy())
        
        # Check for collision (any two objects within threshold)
        final_positions = trajectory[-1]
        collision = False
        for i in range(n_objects):
            for j in range(i+1, n_objects):
                dist = np.linalg.norm(final_positions[i] - final_positions[j])
                if dist < 0.3:
                    collision = True
                    break
            if collision:
                break
        
        # Flatten trajectory for input
        X.append(np.array(trajectory).flatten())
        y.append(np.array([float(collision)]))  # Shape: [1]
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def generate_permanence_data(n_samples=3000, n_objects=5, seq_len=10):
    """
    Object permanence: predict which objects disappeared.
    Input: positions of N objects over seq_len timesteps (some may disappear)
    Output: binary mask of which objects are still present
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random initial positions
        positions = np.random.uniform(-2, 2, (n_objects, 2))
        
        # Random disappearance times for some objects
        disappear_times = np.full(n_objects, seq_len)  # All present by default
        n_disappear = np.random.randint(0, n_objects)  # 0 to n_objects-1 disappear
        if n_disappear > 0:
            disappear_indices = np.random.choice(n_objects, n_disappear, replace=False)
            for idx in disappear_indices:
                disappear_times[idx] = np.random.randint(1, seq_len-1)
        
        # Generate trajectory
        trajectory = []
        present_mask = []
        for t in range(seq_len):
            frame = positions.copy()
            mask = np.ones(n_objects)
            for i in range(n_objects):
                if t >= disappear_times[i]:
                    frame[i] = np.array([0.0, 0.0])  # Disappeared
                    mask[i] = 0.0
            trajectory.append(frame)
            present_mask.append(mask)
        
        # Input: trajectory flattened
        X.append(np.array(trajectory).flatten())
        # Output: final presence mask
        y.append(np.array(present_mask[-1]))  # Shape: [n_objects]
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def generate_spatial_data(n_samples=3000, n_objects=5, seq_len=10):
    """
    Spatial reasoning: predict relative positions between objects.
    Input: positions of N objects over seq_len timesteps
    Output: relative position vectors between object pairs
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random positions
        positions = np.random.uniform(-2, 2, (n_objects, 2))
        
        # Random velocities
        velocities = np.random.uniform(-0.3, 0.3, (n_objects, 2))
        
        # Generate trajectory
        trajectory = []
        for t in range(seq_len):
            trajectory.append(positions.copy())
            positions = positions + velocities * 0.1
        
        # Input: trajectory flattened
        X.append(np.array(trajectory).flatten())
        
        # Output: relative positions between all pairs (final frame)
        final_pos = trajectory[-1]
        rel_positions = []
        for i in range(n_objects):
            for j in range(n_objects):
                if i != j:
                    rel_positions.append(final_pos[j] - final_pos[i])
        y.append(np.array(rel_positions).flatten())
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ============================================================================
# Model Definitions
# ============================================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
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


class TwoNodeCG(nn.Module):
    """Original 2-node CG (physical + semantic)"""
    def __init__(self, input_dim, output_dim, physical_dim=48, semantic_dim=96):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        total_dim = physical_dim + semantic_dim
        
        # Encoders
        self.physical_encoder = nn.Linear(input_dim, physical_dim)
        self.semantic_encoder = nn.Linear(input_dim, semantic_dim)
        
        # GNN layer for cross-modal attention
        self.gnn = nn.Linear(total_dim, total_dim)
        
        # Decoder
        self.decoder = nn.Linear(total_dim, output_dim)
    
    def forward(self, x):
        # Encode to two nodes
        phys = self.physical_encoder(x)  # [B, physical_dim]
        sem = self.semantic_encoder(x)   # [B, semantic_dim]
        
        # Concatenate for processing
        combined = torch.cat([phys, sem], dim=-1)  # [B, total_dim]
        updated = F.relu(self.gnn(combined))
        
        # Decode
        return self.decoder(updated)


class PerObjectCG(nn.Module):
    """
    Per-object CG: N object nodes + 1 semantic node
    Each object has its own physical node
    """
    def __init__(self, input_dim, output_dim, n_objects=5, object_dim=32, semantic_dim=96):
        super().__init__()
        self.n_objects = n_objects
        self.object_dim = object_dim
        self.semantic_dim = semantic_dim
        
        # Per-object encoders
        self.object_encoders = nn.ModuleList([
            nn.Linear(input_dim, object_dim) for _ in range(n_objects)
        ])
        
        # Semantic encoder
        self.semantic_encoder = nn.Linear(input_dim, semantic_dim)
        
        # GNN layers
        node_dim = object_dim + semantic_dim
        self.gnn1 = nn.Linear(node_dim, node_dim)
        self.gnn2 = nn.Linear(node_dim, node_dim)
        
        # Decoder
        self.decoder = nn.Linear(node_dim, output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode each object
        object_nodes = []
        for i, encoder in enumerate(self.object_encoders):
            obj_node = encoder(x)  # [B, object_dim]
            object_nodes.append(obj_node)
        
        # Semantic node
        sem_node = self.semantic_encoder(x)  # [B, semantic_dim]
        
        # Stack all nodes: N object nodes + 1 semantic node
        all_nodes = object_nodes + [sem_node]  # List of [B, dim]
        
        # Process through GNN with cross-attention
        # Concatenate object nodes with semantic for each
        processed_nodes = []
        for i, node in enumerate(all_nodes):
            if i < self.n_objects:
                # Object node: concatenate with semantic
                combined = torch.cat([node, sem_node], dim=-1)
            else:
                # Semantic node: concatenate with mean of objects
                obj_mean = torch.mean(torch.stack(object_nodes), dim=0)
                combined = torch.cat([obj_mean, node], dim=-1)
            
            # GNN processing
            updated = F.relu(self.gnn1(combined))
            updated = F.relu(self.gnn2(updated))
            processed_nodes.append(updated)
        
        # Aggregate: mean of all nodes
        aggregated = torch.mean(torch.stack(processed_nodes), dim=0)
        
        return self.decoder(aggregated)


class HybridCG(nn.Module):
    """
    Hybrid CG: Object-level physical nodes + unified semantic node
    Combines benefits of per-object structure with unified semantic space
    """
    def __init__(self, input_dim, output_dim, n_objects=5, object_dim=24, semantic_dim=96):
        super().__init__()
        self.n_objects = n_objects
        self.object_dim = object_dim
        self.semantic_dim = semantic_dim
        
        # Shared object encoder (processes all objects)
        self.object_encoder = nn.Linear(input_dim, object_dim * n_objects)
        
        # Semantic encoder
        self.semantic_encoder = nn.Linear(input_dim, semantic_dim)
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Object to semantic projection
        self.obj_to_semantic = nn.Linear(object_dim, semantic_dim)
        
        # Decoder
        self.decoder = nn.Linear(semantic_dim * 2, output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode objects
        obj_encoded = self.object_encoder(x)  # [B, object_dim * n_objects]
        obj_nodes = obj_encoded.view(batch_size, self.n_objects, self.object_dim)
        
        # Project objects to semantic space
        obj_semantic = self.obj_to_semantic(obj_nodes)  # [B, n_objects, semantic_dim]
        
        # Semantic node
        sem_node = self.semantic_encoder(x)  # [B, semantic_dim]
        sem_node_expanded = sem_node.unsqueeze(1)  # [B, 1, semantic_dim]
        
        # Cross-attention: semantic attends to objects
        attended, _ = self.cross_attn(
            query=sem_node_expanded,
            key=obj_semantic,
            value=obj_semantic
        )
        
        # Combine attended object info with semantic
        combined = torch.cat([attended.squeeze(1), sem_node], dim=-1)
        
        return self.decoder(combined)


class GraphAttnBaseline(nn.Module):
    """
    Graph Attention baseline (from H1.419)
    Object-level graph with attention between nodes
    """
    def __init__(self, input_dim, output_dim, n_objects=5, node_dim=32):
        super().__init__()
        self.n_objects = n_objects
        self.node_dim = node_dim
        
        # Node encoder - takes first 2 coords per object
        self.node_encoder = nn.Linear(2, node_dim)
        
        # Graph attention layers
        self.gat1 = nn.MultiheadAttention(node_dim, num_heads=4, batch_first=True)
        self.gat2 = nn.MultiheadAttention(node_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Linear(node_dim * n_objects, output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input to per-object features
        # Assume input is organized by timesteps, need to extract per-object info
        obj_features = x.view(batch_size, -1)[:, :self.n_objects * 2]  # First 2 coords per object
        obj_features = obj_features.view(batch_size, self.n_objects, 2)
        
        # Encode each object
        nodes = self.node_encoder(obj_features)  # [B, n_objects, node_dim]
        
        # Graph attention
        attn1, _ = self.gat1(nodes, nodes, nodes)
        nodes = F.relu(nodes + attn1)
        
        attn2, _ = self.gat2(nodes, nodes, nodes)
        nodes = F.relu(nodes + attn2)
        
        # Flatten and decode
        flat = nodes.view(batch_size, -1)
        return self.decoder(flat)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, batch_size=128):
    """Train a model and return training history"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(X_train))
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_loss / n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test)
        y_test_t = torch.tensor(y_test)
        pred = model(X_test_t)
        mse = F.mse_loss(pred, y_test_t).item()
        
        # For classification tasks (collision), compute accuracy
        if y_test.shape[1] == 1:
            pred_binary = (pred.squeeze() > 0.5).float()
            y_binary = y_test_t.squeeze()
            accuracy = (pred_binary == y_binary).float().mean().item()
        else:
            accuracy = None
        
        # MAE for regression
        mae = F.l1_loss(pred, y_test_t).item()
    
    return {'mse': mse, 'mae': mae, 'accuracy': accuracy}


def run_experiment(task_name, generate_data_fn, n_objects=5, seq_len=10, 
                   n_samples=3000, epochs=50, lr=0.001, batch_size=128):
    """Run full experiment for a task"""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    
    # Generate data
    X, y = generate_data_fn(n_samples=n_samples, n_objects=n_objects, seq_len=seq_len)
    
    print(f"Data shapes: X={X.shape}, y={y.shape}")
    
    # Split
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    results = {}
    
    # Models to test
    models = {
        'Baseline MLP': BaselineMLP(input_dim, output_dim, hidden_dim=128),
        '2-Node CG': TwoNodeCG(input_dim, output_dim, physical_dim=48, semantic_dim=96),
        'Per-Object CG': PerObjectCG(input_dim, output_dim, n_objects=n_objects, 
                                      object_dim=32, semantic_dim=96),
        'Hybrid CG': HybridCG(input_dim, output_dim, n_objects=n_objects,
                              object_dim=24, semantic_dim=96),
        'GraphAttn': GraphAttnBaseline(input_dim, output_dim, n_objects=n_objects, node_dim=32)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        history = train_model(model, X_train, y_train, X_val, y_val,
                             epochs=epochs, lr=lr, batch_size=batch_size)
        metrics = evaluate_model(model, X_test, y_test)
        
        results[name] = {
            'test_mse': metrics['mse'],
            'test_mae': metrics['mae'],
            'accuracy': metrics['accuracy'],
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
        
        print(f"  Test MSE: {metrics['mse']:.6f}")
        print(f"  Test MAE: {metrics['mae']:.6f}")
        if metrics['accuracy'] is not None:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    return results


def main():
    print("="*70)
    print("H1.420: Per-Object Cognitive Graph Structure Experiment")
    print("="*70)
    print("\nHypothesis: Per-object CG will match or exceed GraphAttn on permanence task")
    print("Key comparison: Object-level graph structure vs unified CG representation")
    
    all_results = {}
    
    # Run experiments on all three tasks
    all_results['collision'] = run_experiment(
        "Collision Prediction",
        generate_collision_data,
        n_objects=5, seq_len=10, n_samples=3000, epochs=50
    )
    
    all_results['permanence'] = run_experiment(
        "Object Permanence",
        generate_permanence_data,
        n_objects=5, seq_len=10, n_samples=3000, epochs=50
    )
    
    all_results['spatial'] = run_experiment(
        "Spatial Reasoning",
        generate_spatial_data,
        n_objects=5, seq_len=10, n_samples=3000, epochs=50
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for task_name, task_results in all_results.items():
        print(f"\n{task_name.upper()}:")
        baseline_mse = task_results['Baseline MLP']['test_mse']
        for model_name, metrics in task_results.items():
            improvement = (baseline_mse - metrics['test_mse']) / baseline_mse * 100
            print(f"  {model_name}: MSE={metrics['test_mse']:.6f} ({improvement:+.2f}% vs baseline)")
    
    # Save results
    results_path = os.path.dirname(os.path.abspath(__file__)) + '/results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    # Check permanence task specifically
    perm_results = all_results['permanence']
    baseline_mse = perm_results['Baseline MLP']['test_mse']
    two_node_mse = perm_results['2-Node CG']['test_mse']
    per_obj_mse = perm_results['Per-Object CG']['test_mse']
    hybrid_mse = perm_results['Hybrid CG']['test_mse']
    graphattn_mse = perm_results['GraphAttn']['test_mse']
    
    two_node_improvement = (baseline_mse - two_node_mse) / baseline_mse * 100
    per_obj_improvement = (baseline_mse - per_obj_mse) / baseline_mse * 100
    hybrid_improvement = (baseline_mse - hybrid_mse) / baseline_mse * 100
    graphattn_improvement = (baseline_mse - graphattn_mse) / baseline_mse * 100
    
    print(f"\nPermanence Task (key test for H1.420):")
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  2-Node CG: {two_node_mse:.6f} ({two_node_improvement:+.2f}%)")
    print(f"  Per-Object CG: {per_obj_mse:.6f} ({per_obj_improvement:+.2f}%)")
    print(f"  Hybrid CG: {hybrid_mse:.6f} ({hybrid_improvement:+.2f}%)")
    print(f"  GraphAttn: {graphattn_mse:.6f} ({graphattn_improvement:+.2f}%)")
    
    if per_obj_improvement >= graphattn_improvement:
        print("\n✅ H1.420 SUPPORTED: Per-Object CG matches or exceeds GraphAttn!")
    elif hybrid_improvement >= graphattn_improvement:
        print("\n🔸 H1.420 PARTIALLY SUPPORTED: Hybrid CG matches GraphAttn!")
    elif per_obj_improvement > two_node_improvement:
        print(f"\n🔸 H1.420 PARTIALLY SUPPORTED: Per-Object CG ({per_obj_improvement:+.2f}%) outperforms 2-Node CG ({two_node_improvement:+.2f}%), but not GraphAttn ({graphattn_improvement:+.2f}%)")
    else:
        print("\n❌ H1.420 REFUTED: GraphAttn still outperforms CG variants")
    
    return all_results


if __name__ == "__main__":
    main()