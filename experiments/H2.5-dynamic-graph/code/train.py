"""
H2.5: Graph with Dynamic Relationships
Tests if graph structure with time-varying relationships helps.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_dynamic_rel_data(n_samples, n_timesteps=8, n_objects=5, seed=42):
    """Generate data with dynamic object relationships over time."""
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
                next_pos[obj] += rel_action * 0.2
            
            all_next_states.append(next_pos.flatten())
            current = next_pos
        
        history = positions.flatten()
        actions_flat = actions.flatten()
        
        X.append(np.concatenate([history, actions_flat]))
        y.append(np.array(all_next_states).flatten())
    
    return np.array(X), np.array(y)


def add_dynamic_graph_features(X, n_objects=5, n_timesteps=8):
    n_samples = X.shape[0]
    history_dim = n_objects * 6
    
    history = X[:, :history_dim].reshape(n_samples, n_objects, 6)
    
    features = []
    for i in range(n_samples):
        pos = history[i]
        
        static_dists = np.zeros((n_objects, n_objects))
        for o1 in range(n_objects):
            for o2 in range(n_objects):
                static_dists[o1, o2] = np.linalg.norm(pos[o1] - pos[o2])
        
        dynamic_features = static_dists.flatten()
        
        features.append(dynamic_features)
    
    return np.hstack([X, np.array(features)])


def run_experiment():
    print("\n=== H2.5: Graph with Dynamic Relationships (8 objects, 8 steps) ===")
    
    train_X, train_y = generate_dynamic_rel_data(600, n_timesteps=8, n_objects=5, seed=42)
    test_X, test_y = generate_dynamic_rel_data(200, n_timesteps=8, n_objects=5, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"Predicting {train_y.shape[1]} values")
    
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
    
    X_graph = add_dynamic_graph_features(train_X)
    X_test_graph = add_dynamic_graph_features(test_X)
    
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
    print(f"Dynamic Graph: {graph_loss:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    result = {
        'neural_loss': float(neural_loss),
        'graph_loss': float(graph_loss),
        'improvement_percent': float(improvement),
        'status': 'supported' if improvement > 0 else 'refuted'
    }
    
    print(f"\n=== H2.5 Result ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()