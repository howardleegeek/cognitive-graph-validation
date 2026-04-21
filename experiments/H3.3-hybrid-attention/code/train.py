"""
H3.3: Hybrid architecture - concat for simple tasks, attention for complex tasks
Hypothesis: Dynamic switching based on task complexity
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import sys


def generate_temporal_data(n_samples, n_timesteps=8, seed=42):
    np.random.seed(seed)
    state_dim = 24
    action_dim = 8
    X, y = [], []
    for _ in range(n_samples):
        states = []
        actions = []
        state = np.random.randn(state_dim) * 0.5
        for t in range(n_timesteps):
            action = np.random.randn(action_dim) * 0.2
            action_expanded = np.tile(action, 3)[:state_dim]
            next_state = state + action_expanded * 0.15 + np.random.randn(state_dim) * 0.02
            states.append(state)
            actions.append(action)
            state = next_state
        X.append(np.concatenate([s for s in states] + actions))
        y.append(state)
    return np.array(X), np.array(y)


def generate_complex_data(n_samples, n_timesteps=16, seed=42):
    """Generate more complex multi-step data"""
    np.random.seed(seed)
    state_dim = 24
    action_dim = 8
    X, y = [], []
    for _ in range(n_samples):
        states = []
        actions = []
        state = np.random.randn(state_dim) * 0.5
        # Add some nonlinear dynamics
        for t in range(n_timesteps):
            action = np.random.randn(action_dim) * 0.2
            action_expanded = np.tile(action, 3)[:state_dim]
            # Add coupling
            next_state = state + action_expanded * 0.15 + np.sin(state) * 0.05 + np.random.randn(state_dim) * 0.02
            states.append(state)
            actions.append(action)
            state = next_state
        X.append(np.concatenate([s for s in states] + actions))
        y.append(state)
    return np.array(X), np.array(y)


def run_baseline(X, y, test_X, test_y):
    model = MLPRegressor(hidden_layer_sizes=(1024, 512), activation='relu',
                       solver='adam', max_iter=500, random_state=42,
                       early_stopping=True, validation_fraction=0.15)
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_concat(X, y, test_X, test_y, hidden_size=2048):
    model = MLPRegressor(hidden_layer_sizes=(hidden_size, hidden_size//2), activation='relu',
                       solver='adam', max_iter=500, random_state=42,
                       early_stopping=True, validation_fraction=0.15, alpha=0.001)
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_with_graph_features(X, y, test_X, test_y, hidden_size=2048):
    """Simulate attention/graph features by adding temporal differences"""
    n_samples = X.shape[0]
    seq_len = X.shape[1] // 24
    state_dim = 24
    
    X_enhanced = []
    for i in range(n_samples):
        sample = X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        enhanced = np.concatenate([sample, diffs_padded.flatten()])
        X_enhanced.append(enhanced)
    X_enhanced = np.array(X_enhanced)
    
    test_enhanced = []
    for i in range(test_X.shape[0]):
        sample = test_X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        enhanced = np.concatenate([sample, diffs_padded.flatten()])
        test_enhanced.append(enhanced)
    test_enhanced = np.array(test_enhanced)
    
    model = MLPRegressor(hidden_layer_sizes=(hidden_size+256, hidden_size//2), activation='relu',
                        solver='adam', max_iter=600, random_state=42,
                        early_stopping=True, validation_fraction=0.15, alpha=0.01)
    model.fit(X_enhanced, y)
    pred = model.predict(test_enhanced)
    return np.mean((pred - test_y) ** 2)


def run_hybrid(X_simple, y_simple, test_X_simple, test_y_simple,
              X_complex, y_complex, test_X_complex, test_y_complex):
    """Hybrid: concat for simple, graph for complex"""
    # Simple task - use concat
    concat_loss = run_concat(X_simple, y_simple, test_X_simple, test_y_simple)
    
    # Complex task - use graph features (simulating attention advantage)
    graph_loss = run_with_graph_features(X_complex, y_complex, test_X_complex, test_y_complex)
    
    return concat_loss, graph_loss


def run_experiment():
    print("\n=== H3.3: Hybrid Architecture ===", flush=True)
    
    results = {}
    
    # Simple tasks (8 steps)
    print("\n--- Simple tasks (8-step) ---", flush=True)
    train_X_simple, train_y_simple = generate_temporal_data(300, n_timesteps=8, seed=42)
    test_X_simple, test_y_simple = generate_temporal_data(150, n_timesteps=8, seed=789)
    
    baseline_simple = run_baseline(train_X_simple, train_y_simple, test_X_simple, test_y_simple)
    concat_simple = run_concat(train_X_simple, train_y_simple, test_X_simple, test_y_simple)
    
    print(f"Baseline: {baseline_simple:.4f}", flush=True)
    print(f"Concat: {concat_simple:.4f}", flush=True)
    results["simple_baseline"] = baseline_simple
    results["simple_concat"] = concat_simple
    
    # Complex tasks (16 steps)
    print("\n--- Complex tasks (16-step) ---", flush=True)
    train_X_complex, train_y_complex = generate_complex_data(300, n_timesteps=16, seed=42)
    test_X_complex, test_y_complex = generate_complex_data(150, n_timesteps=16, seed=789)
    
    baseline_complex = run_baseline(train_X_complex, train_y_complex, test_X_complex, test_y_complex)
    concat_complex = run_concat(train_X_complex, train_y_complex, test_X_complex, test_y_complex)
    graph_complex = run_with_graph_features(train_X_complex, train_y_complex, test_X_complex, test_y_complex)
    
    print(f"Baseline: {baseline_complex:.4f}", flush=True)
    print(f"Concat: {concat_complex:.4f}", flush=True)
    print(f"Graph: {graph_complex:.4f}", flush=True)
    results["complex_baseline"] = baseline_complex
    results["complex_concat"] = concat_complex
    results["complex_graph"] = graph_complex
    
    # 20-step tasks
    print("\n--- Very complex (20-step) ---", flush=True)
    train_X_20, train_y_20 = generate_complex_data(300, n_timesteps=20, seed=42)
    test_X_20, test_y_20 = generate_complex_data(150, n_timesteps=20, seed=789)
    
    baseline_20 = run_baseline(train_X_20, train_y_20, test_X_20, test_y_20)
    concat_20 = run_concat(train_X_20, train_y_20, test_X_20, test_y_20)
    graph_20 = run_with_graph_features(train_X_20, train_y_20, test_X_20, test_y_20)
    
    print(f"Baseline: {baseline_20:.4f}", flush=True)
    print(f"Concat: {concat_20:.4f}", flush=True)
    print(f"Graph: {graph_20:.4f}", flush=True)
    results["20_baseline"] = baseline_20
    results["20_concat"] = concat_20
    results["20_graph"] = graph_20
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    print("\n=== Summary ===", flush=True)
    print(f"Simple (8-step): Concat wins = {results['simple_concat'] < results['simple_baseline']}", flush=True)
    print(f"Complex (16-step): Graph wins = {results['complex_graph'] < results['complex_concat']}", flush=True)
    print(f"Very complex (20-step): Graph wins = {results['20_graph'] < results['20_concat']}", flush=True)