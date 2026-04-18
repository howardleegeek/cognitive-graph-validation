"""
H2.4: Explicit Graph on Longer Temporal Horizons
Tests if explicit graph structure helps with 10+ timestep temporal reasoning.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_long_temporal_data(n_samples, n_timesteps=12, n_objects=4, seed=42):
    """Generate data requiring long-horizon temporal reasoning."""
    np.random.seed(seed)
    
    state_dim = 8
    
    X = []
    y = []
    
    for _ in range(n_samples):
        positions = np.random.randn(n_objects, state_dim) * 0.5
        
        actions = np.random.randn(n_timesteps, 4) * 0.2
        
        all_next_states = []
        current_positions = positions.copy()
        
        for t in range(n_timesteps):
            next_pos = current_positions.copy()
            
            for obj in range(n_objects):
                if obj == 0:
                    next_pos[obj] += actions[t, 0] * 0.2
                elif obj == 1:
                    next_pos[obj] += actions[t, 1] * 0.15
                elif obj == 2:
                    next_pos[obj] += (actions[t, 0] + actions[t, 1]) * 0.1
                else:
                    next_pos[obj] += 0
            
            all_next_states.append(next_pos.flatten())
            current_positions = next_pos
        
        history = positions.flatten()
        action_sequence = actions.flatten()
        
        X.append(np.concatenate([history, action_sequence]))
        y.append(np.array(all_next_states).flatten())
    
    return np.array(X), np.array(y)


def add_graph_features(X, n_objects=4):
    n_samples = X.shape[0]
    n_timesteps = 12
    
    history_dim = n_objects * 8
    action_dim_total = n_timesteps * 4
    
    history = X[:, :history_dim].reshape(n_samples, n_objects, 8)
    
    graph_features = []
    for i in range(n_samples):
        positions = history[i]
        dists = np.zeros((n_objects, n_objects))
        for o1 in range(n_objects):
            for o2 in range(n_objects):
                dists[o1, o2] = np.linalg.norm(positions[o1] - positions[o2])
        graph_features.append(dists.flatten())
    
    return np.hstack([X, np.array(graph_features)])


def run_experiment():
    print("\n=== H2.4: Explicit Graph on Long Temporal Horizons (12 steps) ===")
    
    train_X, train_y = generate_long_temporal_data(600, n_timesteps=12, n_objects=4, seed=42)
    test_X, test_y = generate_long_temporal_data(200, n_timesteps=12, n_objects=4, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"Predicting {train_y.shape[1]} values (4 objects x 12 timesteps x 8 dims)")
    
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
    
    X_with_graph = add_graph_features(train_X)
    X_test_with_graph = add_graph_features(test_X)
    
    graph_model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=43,
        early_stopping=True,
        validation_fraction=0.15
    )
    graph_model.fit(X_with_graph, train_y)
    graph_pred = graph_model.predict(X_test_with_graph)
    graph_loss = np.mean((graph_pred - test_y) ** 2)
    
    improvement = (neural_loss - graph_loss) / neural_loss * 100
    
    print(f"\n=== Results ===")
    print(f"Pure Neural: {neural_loss:.4f}")
    print(f"Graph-Enhanced: {graph_loss:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    result = {
        'neural_loss': float(neural_loss),
        'graph_loss': float(graph_loss),
        'improvement_percent': float(improvement),
        'status': 'supported' if improvement > 0 else 'refuted'
    }
    
    print(f"\n=== H2.4 Result ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()