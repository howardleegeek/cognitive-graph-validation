"""
H2.3: Explicit Graph on Temporal Reasoning
Tests if explicit graph structure helps with temporal reasoning (object permanence).
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
        # Initial positions of objects
        positions = np.random.randn(n_objects, state_dim) * 0.5
        
        # Actions across timesteps
        actions = np.random.randn(n_timesteps, action_dim) * 0.2
        
        # Physics: objects move based on actions
        all_next_states = []
        current_positions = positions.copy()
        
        for t in range(n_timesteps):
            # Object 0 follows action, object 1 is static, object 2 has noise
            next_pos = current_positions.copy()
            next_pos[0] += actions[t, 0] * 0.3
            next_pos[1] += 0  # static
            next_pos[2] += np.random.randn(state_dim) * 0.05  # random
            
            all_next_states.append(next_pos.flatten())
            current_positions = next_pos
        
        # State includes all historical positions (temporal reasoning required)
        history = positions.flatten()  # initial positions
        action_sequence = actions.flatten()
        
        X.append(np.concatenate([history, action_sequence]))
        y.append(np.array(all_next_states).flatten())  # predict all future states
    
    return np.array(X), np.array(y)


def run_experiment():
    """Run H2.3 explicit graph on temporal reasoning."""
    print("\n=== H2.3: Explicit Graph on Temporal Reasoning ===")
    
    train_X, train_y = generate_temporal_data(400, n_objects=3, n_timesteps=5, seed=42)
    test_X, test_y = generate_temporal_data(200, n_objects=3, n_timesteps=5, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"Predicting {train_y.shape[1]} values (3 objects x 5 timesteps x 8 dims)")
    
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
    
    # Graph-enhanced (simulate GNN by adding graph structure features)
    # Add explicit relationship features between objects
    def add_graph_features(X, n_objects=3):
        n_samples = X.shape[0]
        n_timesteps = 5
        action_dim = 4
        
        history_dim = n_objects * 8
        action_dim_total = n_timesteps * action_dim
        
        history = X[:, :history_dim].reshape(n_samples, n_objects, 8)
        actions = X[:, history_dim:history_dim + action_dim_total].reshape(n_samples, n_timesteps, action_dim)
        
        # Compute pairwise distances between objects (graph edges)
        graph_features = []
        for i in range(n_samples):
            positions = history[i]  # (3, 8)
            dists = np.zeros((n_objects, n_objects))
            for o1 in range(n_objects):
                for o2 in range(n_objects):
                    dists[o1, o2] = np.linalg.norm(positions[o1] - positions[o2])
            graph_features.append(dists.flatten())
        
        graph_features = np.array(graph_features)
        return np.hstack([X, graph_features])
    
    X_with_graph = add_graph_features(train_X)
    X_test_with_graph = add_graph_features(test_X)
    
    graph_model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
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
    
    print(f"\n=== H2.3 Result ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()