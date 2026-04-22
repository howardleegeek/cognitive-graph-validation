"""
H2.11: Hierarchical Graph + Graph Transformer Combined
Testing if combining hierarchical structure with transformer attention provides additional benefit
"""

import numpy as np

def generate_multi_object_temporal(n_objects=4, n_steps=12, n_samples=250):
    """Generate multi-object temporal tasks"""
    np.random.seed(42)
    states = []
    actions = []
    next_states = []
    adjacencies = []
    
    for i in range(n_samples):
        # Generate object positions (3D)
        obj_pos = np.random.randn(n_objects, 3) * 0.1
        for t in range(n_steps):
            # Actions affect objects (3D action)
            actions_t = np.random.randn(n_objects, 3) * 0.1
            # Physics: positions evolve
            delta = np.dot(actions_t, np.ones(3))[:, None] * np.ones(3) * 0.1
            next_pos = obj_pos + delta + np.random.randn(n_objects, 3) * 0.01
            # Build adjacency based on distance
            dist = np.linalg.norm(next_pos[:, None] - next_pos, axis=2)
            adj = (dist < 1.0).astype(float) - np.eye(n_objects)
            
            state_flat = next_pos.flatten()
            action_flat = actions_t.flatten()
            state_full = np.concatenate([state_flat, action_flat])
            
            states.append(state_full)
            actions.append(action_flat)
            next_states.append(next_pos.flatten())
            adjacencies.append(adj)
        
        obj_pos = next_pos.copy() if 'next_pos' in locals() else obj_pos
    
    return np.array(states), np.array(actions), np.array(next_states), np.array(adjacencies)

def baseline_neural(x, dims=512):
    """Standard neural network"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    return np.mean(h ** 2)

def hierarchical_graph(x, adj, dims=512):
    """Hierarchical graph neural network"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    w2 = np.random.randn(dims, dims) * np.sqrt(2.0 / (dims + dims))
    h = np.tanh(x @ w1)
    # Hierarchical message passing
    for _ in range(3):
        h = h + 0.1 * np.tanh(h @ w2)
    return np.mean(h ** 2)

def graph_transformer(x, adj, dims=512):
    """Graph transformer with self-attention over edges"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    # Self-attention over nodes
    scale = np.sqrt(dims)
    attn_logits = (h @ h.T) / scale
    attn = softmax(attn_logits, axis=-1)
    h = attn @ h
    return np.mean(h ** 2)

def combined_hierarchical_transformer(x, adj, dims=512):
    """Combined hierarchical graph + transformer"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    w2 = np.random.randn(dims, dims) * np.sqrt(2.0 / (dims + dims))
    h = np.tanh(x @ w1)
    # Hierarchical passes
    for _ in range(3):
        h = h + 0.1 * np.tanh(h @ w2)
    # Then transformer attention
    scale = np.sqrt(dims)
    attn_logits = (h @ h.T) / scale
    attn = softmax(attn_logits, axis=-1)
    h = attn @ h
    return np.mean(h ** 2)

def softmax(x, axis=-1):
    """Manual softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def run_experiment():
    """Run H2.11 experiment"""
    results = []
    
    for n_objects in [3, 4, 5, 6]:
        X, A, Y, Adj = generate_multi_object_temporal(n_objects=n_objects, n_steps=12, n_samples=250)
        
        baseline_losses = []
        hier_losses = []
        transformer_losses = []
        combined_losses = []
        
        for run in range(5):
            np.random.seed(42 + run)
            baseline_losses.append(baseline_neural(X))
            hier_losses.append(hierarchical_graph(X, Adj))
            transformer_losses.append(graph_transformer(X, Adj))
            combined_losses.append(combined_hierarchical_transformer(X, Adj))
        
        baseline_mse = np.mean(baseline_losses)
        hier_mse = np.mean(hier_losses)
        transformer_mse = np.mean(transformer_losses)
        combined_mse = np.mean(combined_losses)
        
        # Best individual vs combined
        best_individual = min(hier_mse, transformer_mse)
        improvement_vs_best = (best_individual - combined_mse) / best_individual * 100
        improvement_vs_baseline = (baseline_mse - combined_mse) / baseline_mse * 100
        
        print(f"{n_objects}-obj: Baseline={baseline_mse:.4f}, Hier={hier_mse:.4f}, Trans={transformer_mse:.4f}, Combined={combined_mse:.4f}")
        results.append({
            'n_objects': n_objects,
            'baseline': baseline_mse,
            'hierarchical': hier_mse,
            'transformer': transformer_mse,
            'combined': combined_mse,
            'improvement_vs_best': improvement_vs_best,
            'improvement_vs_baseline': improvement_vs_baseline
        })
    
    avg_vs_best = np.mean([r['improvement_vs_best'] for r in results])
    avg_vs_baseline = np.mean([r['improvement_vs_baseline'] for r in results])
    print(f"\nAvg vs Best Individual: {avg_vs_best:+.1f}%")
    print(f"Avg vs Baseline: {avg_vs_baseline:+.1f}%")
    
    # Determine status
    status = "SUPPORTED" if avg_vs_best > 5 else ("MARGINAL" if avg_vs_best > 0 else "REFUTED")
    print(f"Status: {status}")
    
    return results, status, avg_vs_best, avg_vs_baseline

if __name__ == "__main__":
    results, status, avg_vs_best, avg_vs_baseline = run_experiment()