"""
H2.7: Graph + Regularization Combined
Hypothesis: Graph features + strong regularization is optimal
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


def run_baseline(X, y, test_X, test_y, hidden_size=4096, alpha=0.01):
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_size, hidden_size//2),
        activation='relu', solver='adam', alpha=alpha,
        max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.15
    )
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_graph_enhanced(X, y, test_X, test_y, hidden_size=4096, alpha=0.1):
    """Graph-enhanced: adds relative position features"""
    n_samples = X.shape[0]
    seq_len = X.shape[1] // 24
    state_dim = 24
    
    X_enh = []
    for i in range(n_samples):
        sample = X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        X_enh.append(np.concatenate([sample, diffs_padded.flatten()]))
    X_enh = np.array(X_enh)
    
    test_enh = []
    for i in range(test_X.shape[0]):
        sample = test_X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        test_enh.append(np.concatenate([sample, diffs_padded.flatten()]))
    test_enh = np.array(test_enh)
    
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_size+256, hidden_size//2),
        activation='relu', solver='adam', alpha=alpha,
        max_iter=600, random_state=42,
        early_stopping=True, validation_fraction=0.15
    )
    model.fit(X_enh, y)
    pred = model.predict(test_enh)
    return np.mean((pred - test_y) ** 2)


def run_experiment():
    print("\n=== H2.7: Graph + Regularization ===", flush=True)
    
    results = {}
    
    # Simple 8-step
    print("\n--- 8-step tasks ---", flush=True)
    train_X, train_y = generate_temporal_data(300, n_timesteps=8, seed=42)
    test_X, test_y = generate_temporal_data(150, n_timesteps=8, seed=789)
    
    baseline_8 = run_baseline(train_X, train_y, test_X, test_y, hidden_size=4096, alpha=0.01)
    graph_8 = run_graph_enhanced(train_X, train_y, test_X, test_y, hidden_size=4096, alpha=0.1)
    print(f"Baseline: {baseline_8:.4f}", flush=True)
    print(f"Graph+reg: {graph_8:.4f}", flush=True)
    results["8_base"] = baseline_8
    results["8_graph"] = graph_8
    
    # Complex 12-step
    print("\n--- 12-step tasks ---", flush=True)
    train_X12, train_y12 = generate_temporal_data(300, n_timesteps=12, seed=42)
    test_X12, test_y12 = generate_temporal_data(150, n_timesteps=12, seed=789)
    
    baseline_12 = run_baseline(train_X12, train_y12, test_X12, test_y12, hidden_size=4096, alpha=0.01)
    graph_12 = run_graph_enhanced(train_X12, train_y12, test_X12, test_y12, hidden_size=4096, alpha=0.1)
    print(f"Baseline: {baseline_12:.4f}", flush=True)
    print(f"Graph+reg: {graph_12:.4f}", flush=True)
    results["12_base"] = baseline_12
    results["12_graph"] = graph_12
    
    # Larger model
    print("\n--- 12-step with 8192 ---", flush=True)
    baseline_12k = run_baseline(train_X12, train_y12, test_X12, test_y12, hidden_size=8192, alpha=0.1)
    graph_12k = run_graph_enhanced(train_X12, train_y12, test_X12, test_y12, hidden_size=8192, alpha=0.1)
    print(f"Baseline (8192): {baseline_12k:.4f}", flush=True)
    print(f"Graph+reg (8192): {graph_12k:.4f}", flush=True)
    results["12_base_8k"] = baseline_12k
    results["12_graph_8k"] = graph_12k
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    print("\n=== Summary ===", flush=True)
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"{k}: {v:.4f}", flush=True)