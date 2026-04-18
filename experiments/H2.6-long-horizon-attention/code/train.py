"""
H2.6: Graph with Attention for Very Long Horizons
Tests if graph + light attention helps on 20+ step temporal tasks.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_long_horizon_data(n_samples, n_timesteps=20, n_objects=4, seed=42):
    """Generate data requiring very long-horizon temporal reasoning."""
    np.random.seed(seed)
    
    state_dim = 6
    
    X = []
    y = []
    
    for _ in range(n_samples):
        positions = np.random.randn(n_objects, state_dim) * 0.5
        actions = np.random.randn(n_timesteps, 4) * 0.2
        
        all_next_states = []
        current = positions.copy()
        
        for t in range(n_timesteps):
            next_pos = current.copy()
            
            for obj in range(n_objects):
                rel_action = np.sum(actions[t, :min(4, obj+1)])
                next_pos[obj] += rel_action * 0.15 + np.random.randn(state_dim) * 0.01
            
            all_next_states.append(next_pos.flatten())
            current = next_pos
        
        history = positions.flatten()
        actions_flat = actions.flatten()
        
        X.append(np.concatenate([history, actions_flat]))
        y.append(np.array(all_next_states).flatten())
    
    return np.array(X), np.array(y)


def add_graph_attention_features(X, n_objects=4, n_timesteps=20):
    n_samples = X.shape[0]
    history_dim = n_objects * 6
    
    history = X[:, :history_dim].reshape(n_samples, n_objects, 6)
    
    features = []
    for i in range(n_samples):
        pos = history[i]
        
        dists = np.zeros((n_objects, n_objects))
        for o1 in range(n_objects):
            for o2 in range(n_objects):
                dists[o1, o2] = np.linalg.norm(pos[o1] - pos[o2])
        
        avg_dist = np.mean(dists)
        max_dist = np.max(dists)
        
        attention_weights = np.exp(-dists / (avg_dist + 1e-6))
        attention_weights = attention_weights / (attention_weights.sum() + 1e-6)
        
        graph_feat = np.concatenate([
            dists.flatten(),
            [avg_dist, max_dist],
            attention_weights.flatten()
        ])
        
        features.append(graph_feat)
    
    return np.hstack([X, np.array(features)])


def run_experiment():
    print("\n=== H2.6: Graph + Attention on Very Long Horizons (20 steps) ===")
    
    train_X, train_y = generate_long_horizon_data(800, n_timesteps=20, n_objects=4, seed=42)
    test_X, test_y = generate_long_horizon_data(200, n_timesteps=20, n_objects=4, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"Predicting {train_y.shape[1]} values (4 objects x 20 timesteps x 6 dims)")
    
    neural_model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
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
    
    X_graph = add_graph_attention_features(train_X)
    X_test_graph = add_graph_attention_features(test_X)
    
    graph_model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=43,
        early_stopping=True,
        validation_fraction=0.15
    )
    graph_model.fit(X_graph, train_y)
    graph_pred = graph_model.predict(X_test_graph)
    graph_loss = np.mean((graph_pred - test_y) ** 2)
    
    improvement = (neural_loss - graph_loss) / neural_loss * 100
    
    print(f"\n=== Results ===")
    print(f"Pure Neural: {neural_loss:.4f}")
    print(f"Graph+Attention: {graph_loss:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    result = {
        'neural_loss': float(neural_loss),
        'graph_attention_loss': float(graph_loss),
        'improvement_percent': float(improvement),
        'n_timesteps': 20,
        'status': 'supported' if improvement > 0 else 'refuted'
    }
    
    print(f"\n=== H2.6 Result ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()