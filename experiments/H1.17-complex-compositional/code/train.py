"""
H1.17: Test Graph + 4096 on complex compositional (8+ step) tasks
Hypothesis: Graph-enhanced unified architecture outperforms single-branch on complex tasks
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import sys


def generate_compositional_data(n_samples, n_objects=3, n_steps=8, seed=42):
    """Multi-object compositional tasks with multiple steps"""
    np.random.seed(seed)
    state_dim = 24  # Fixed state dim
    action_dim = 8
    X, y = [], []
    for _ in range(n_samples):
        states = []
        actions = []
        # Each object starts at random position
        state = np.random.randn(state_dim) * 0.5
        for t in range(n_steps):
            action = np.random.randn(action_dim) * 0.2
            action_expanded = np.tile(action, 3)[:state_dim]
            next_state = state + action_expanded * 0.12 + np.random.randn(state_dim) * 0.02
            states.append(state)
            actions.append(action)
            state = next_state
        X.append(np.concatenate([s for s in states] + actions))
        y.append(state)
    return np.array(X), np.array(y)


def generate_complex_fusion_data(n_samples, n_branches=2, n_steps=10, seed=42):
    """Test two-branch fusion vs single branch for complex tasks"""
    np.random.seed(seed)
    state_dim = 24
    action_dim = 8
    X, y = [], []
    for _ in range(n_samples):
        states = []
        actions = []
        state = np.random.randn(state_dim) * 0.5
        for t in range(n_steps):
            action = np.random.randn(action_dim) * 0.2
            # Two separate sources of information
            action_expanded = np.tile(action, 3)[:state_dim]
            nonlinearity = np.sin(state) * 0.1
            next_state = state + action_expanded * 0.15 + nonlinearity + np.random.randn(state_dim) * 0.02
            states.append(state)
            actions.append(action)
            state = next_state
        X.append(np.concatenate([s for s in states] + actions))
        y.append(state)
    return np.array(X), np.array(y)


def run_single_branch(X, y, test_X, test_y, hidden_size=4096):
    """Single large branch (like baseline)"""
    model = MLPRegressor(hidden_layer_sizes=(hidden_size, hidden_size//2), activation='relu',
                       solver='adam', max_iter=500, random_state=42,
                       early_stopping=True, validation_fraction=0.15, alpha=0.001)
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_graph_enhanced(X, y, test_X, test_y, hidden_size=4096):
    """Graph-enhanced (adds difference features)"""
    n_samples = X.shape[0]
    seq_len = X.shape[1] // 24
    state_dim = 24
    
    X_enhanced = []
    for i in range(n_samples):
        sample = X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        # Graph-like features: relative positions
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        # Add velocity magnitude
        velocity = np.linalg.norm(diffs_padded, axis=1, keepdims=True)
        velocity = np.tile(velocity, (1, state_dim))
        enhanced = np.concatenate([sample, diffs_padded.flatten(), velocity.flatten()])
        X_enhanced.append(enhanced)
    X_enhanced = np.array(X_enhanced)
    
    test_enhanced = []
    for i in range(test_X.shape[0]):
        sample = test_X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        velocity = np.linalg.norm(diffs_padded, axis=1, keepdims=True)
        velocity = np.tile(velocity, (1, state_dim))
        enhanced = np.concatenate([sample, diffs_padded.flatten(), velocity.flatten()])
        test_enhanced.append(enhanced)
    test_enhanced = np.array(test_enhanced)
    
    model = MLPRegressor(hidden_layer_sizes=(hidden_size+256, hidden_size//2), activation='relu',
                        solver='adam', max_iter=600, random_state=42,
                        early_stopping=True, validation_fraction=0.15, alpha=0.01)
    model.fit(X_enhanced, y)
    pred = model.predict(test_enhanced)
    return np.mean((pred - test_y) ** 2)


def run_experiment():
    print("\n=== H1.17: Graph + 4096 on Complex Compositional Tasks ===", flush=True)
    
    results = {}
    
    # 8-step compositional
    print("\n--- 8-step compositional ---", flush=True)
    train_X_8, train_y_8 = generate_compositional_data(300, n_objects=3, n_steps=8, seed=42)
    test_X_8, test_y_8 = generate_compositional_data(150, n_objects=3, n_steps=8, seed=789)
    
    single_8 = run_single_branch(train_X_8, train_y_8, test_X_8, test_y_8, hidden_size=4096)
    graph_8 = run_graph_enhanced(train_X_8, train_y_8, test_X_8, test_y_8, hidden_size=4096)
    
    print(f"Single (4096): {single_8:.4f}", flush=True)
    print(f"Graph+4096: {graph_8:.4f}", flush=True)
    results["8_single"] = single_8
    results["8_graph"] = graph_8
    
    # 12-step complex
    print("\n--- 12-step complex ---", flush=True)
    train_X_12, train_y_12 = generate_complex_fusion_data(300, n_steps=12, seed=42)
    test_X_12, test_y_12 = generate_complex_fusion_data(150, n_steps=12, seed=789)
    
    single_12 = run_single_branch(train_X_12, train_y_12, test_X_12, test_y_12, hidden_size=4096)
    graph_12 = run_graph_enhanced(train_X_12, train_y_12, test_X_12, test_y_12, hidden_size=4096)
    
    print(f"Single (4096): {single_12:.4f}", flush=True)
    print(f"Graph+4096: {graph_12:.4f}", flush=True)
    results["12_single"] = single_12
    results["12_graph"] = graph_12
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    print("\n=== Summary ===", flush=True)
    print(f"8-step: Single={results['8_single']:.4f}, Graph={results['8_graph']:.4f}, Graph wins: {results['8_graph'] < results['8_single']}", flush=True)
    print(f"12-step: Single={results['12_single']:.4f}, Graph={results['12_graph']:.4f}, Graph wins: {results['12_graph'] < results['12_single']}", flush=True)