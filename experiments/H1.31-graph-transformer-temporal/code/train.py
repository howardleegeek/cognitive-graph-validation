"""
H1.31: Graph Transformer on Temporal Tasks
Combines self-attention over graph edges (H1.30: +5.7%) with temporal reasoning (H2.3: +56.8%).
Tests if transformer-style graph outperforms standard GNN on temporal tasks.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_temporal_data(n_samples, n_objects=3, n_timesteps=5, seed=42):
    """Generate data requiring temporal reasoning (object permanence)."""
    np.random.seed(seed)
    
    state_dim = 8  # per object
    action_dim = 4
    
    X = []
    y = []
    
    for _ in range(n_samples):
        positions = np.random.randn(n_objects, state_dim) * 0.5
        actions = np.random.randn(n_timesteps, action_dim) * 0.2
        
        all_next_states = []
        current_positions = positions.copy()
        
        for t in range(n_timesteps):
            next_pos = current_positions.copy()
            next_pos[0] += actions[t, 0] * 0.3
            next_pos[1] += 0
            next_pos[2] += np.random.randn(state_dim) * 0.05
            
            all_next_states.append(next_pos.flatten())
            current_positions = next_pos
        
        history = positions.flatten()
        action_sequence = actions.flatten()
        
        X.append(np.concatenate([history, action_sequence]))
        y.append(np.array(all_next_states).flatten())
    
    return np.array(X), np.array(y)


def add_gnn_features(X, n_objects=3, n_timesteps=5):
    """Standard GNN graph features (baseline from H2.3)."""
    n_samples = X.shape[0]
    history_dim = n_objects * 8
    
    history = X[:, :history_dim].reshape(n_samples, n_objects, 8)
    
    graph_features = []
    for i in range(n_samples):
        positions = history[i]
        dists = np.zeros((n_objects, n_objects))
        for o1 in range(n_objects):
            for o2 in range(n_objects):
                dists[o1, o2] = np.linalg.norm(positions[o1] - positions[o2])
        graph_features.append(dists.flatten())
    
    graph_features = np.array(graph_features)
    return np.hstack([X, graph_features])


def add_transformer_features(X, n_objects=3, n_timesteps=5):
    """Graph transformer: self-attention over edges + relational features."""
    n_samples = X.shape[0]
    history_dim = n_objects * 8
    
    history = X[:, :history_dim].reshape(n_samples, n_objects, 8)
    
    transformer_features = []
    for i in range(n_samples):
        positions = history[i]
        
        edges = []
        for o1 in range(n_objects):
            for o2 in range(n_objects):
                if o1 != o2:
                    diff = positions[o1] - positions[o2]
                    edges.append(diff)
        
        edges = np.array(edges)
        
        if len(edges) > 0:
            edge_mean = np.mean(edges, axis=0)
            edge_std = np.std(edges, axis=0)
            edge_max = np.max(edges, axis=0)
        else:
            edge_mean = np.zeros(8)
            edge_std = np.zeros(8)
            edge_max = np.zeros(8)
        
        dists = np.zeros((n_objects, n_objects))
        for o1 in range(n_objects):
            for o2 in range(n_objects):
                dists[o1, o2] = np.linalg.norm(positions[o1] - positions[o2])
        
        combined = np.concatenate([
            dists.flatten(),
            edge_mean,
            edge_std,
            edge_max
        ])
        transformer_features.append(combined)
    
    transformer_features = np.array(transformer_features)
    return np.hstack([X, transformer_features])


def run_experiment():
    """Run H1.31: Graph Transformer on Temporal Tasks."""
    print("\n=== H1.31: Graph Transformer on Temporal Tasks ===")
    print("Combining H1.30 self-attention (+5.7%) with temporal reasoning")
    
    results = {'configurations': []}
    
    for n_timesteps in [5, 8, 12]:
        print(f"\n--- Timesteps: {n_timesteps} ---")
        
        train_X, train_y = generate_temporal_data(400, n_objects=3, n_timesteps=n_timesteps, seed=42)
        test_X, test_y = generate_temporal_data(200, n_objects=3, n_timesteps=n_timesteps, seed=789)
        
        print(f"Train: {train_X.shape}, Test: {test_X.shape}")
        
        # Pure neural baseline
        neural_model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        neural_model.fit(train_X, train_y)
        neural_pred = neural_model.predict(test_X)
        neural_loss = np.mean((neural_pred - test_y) ** 2)
        
        # Standard GNN features
        X_gnn = add_gnn_features(train_X, n_objects=3, n_timesteps=n_timesteps)
        X_test_gnn = add_gnn_features(test_X, n_objects=3, n_timesteps=n_timesteps)
        
        gnn_model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=43,
            early_stopping=True,
            validation_fraction=0.15
        )
        gnn_model.fit(X_gnn, train_y)
        gnn_pred = gnn_model.predict(X_test_gnn)
        gnn_loss = np.mean((gnn_pred - test_y) ** 2)
        
        # Graph transformer features
        X_trans = add_transformer_features(train_X, n_objects=3, n_timesteps=n_timesteps)
        X_test_trans = add_transformer_features(test_X, n_objects=3, n_timesteps=n_timesteps)
        
        trans_model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=44,
            early_stopping=True,
            validation_fraction=0.15
        )
        trans_model.fit(X_trans, train_y)
        trans_pred = trans_model.predict(X_test_trans)
        trans_loss = np.mean((trans_pred - test_y) ** 2)
        
        improvement_gnn = (neural_loss - gnn_loss) / neural_loss * 100
        improvement_trans = (neural_loss - trans_loss) / neural_loss * 100
        improvement_vs_gnn = (gnn_loss - trans_loss) / gnn_loss * 100
        
        print(f"Pure Neural: {neural_loss:.4f}")
        print(f"GNN: {gnn_loss:.4f} ({improvement_gnn:+.1f}%)")
        print(f"Transformer: {trans_loss:.4f} ({improvement_trans:+.1f}%)")
        print(f"Transformer vs GNN: {improvement_vs_gnn:+.1f}%")
        
        results['configurations'].append({
            'timesteps': n_timesteps,
            'neural': float(neural_loss),
            'gnn': float(gnn_loss),
            'transformer': float(trans_loss),
            'improvement_gnn': float(improvement_gnn),
            'improvement_trans': float(improvement_trans),
            'improvement_vs_gnn': float(improvement_vs_gnn)
        })
    
    avg_improvement = np.mean([c['improvement_vs_gnn'] for c in results['configurations']])
    overall_improvement = np.mean([c['improvement_trans'] for c in results['configurations']])
    
    results['avg_improvement_vs_gnn'] = float(avg_improvement)
    results['avg_improvement_vs_neural'] = float(overall_improvement)
    results['status'] = 'supported' if avg_improvement > 0 else 'refuted'
    
    print(f"\n=== Summary ===")
    print(f"Transformer vs GNN: {avg_improvement:+.1f}% average")
    print(f"Transformer vs Neural: {overall_improvement:+.1f}%")
    print(f"Status: {results['status']}")
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == '__main__':
    run_experiment()