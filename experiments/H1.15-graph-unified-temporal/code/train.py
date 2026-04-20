"""
H1.15: Test combined graph + unified architecture on temporal tasks
Hypothesis: Graph + unified architecture outperforms either alone
"""
import numpy as np
from sklearn.neural_network import MLPRegressor


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


def run_baseline(X, y, test_X, test_y):
    model = MLPRegressor(hidden_layer_sizes=(1024, 512), activation='relu',
                         solver='adam', max_iter=500, random_state=42,
                         early_stopping=True, validation_fraction=0.15)
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_unified(X, y, test_X, test_y, dim=2048):
    model = MLPRegressor(hidden_layer_sizes=(dim, dim // 2), activation='relu',
                         solver='adam', max_iter=500, random_state=42,
                         early_stopping=True, validation_fraction=0.15)
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_graph_unified(X, y, test_X, test_y, dim=2048):
    n_samples = X.shape[0]
    seq_len = 8
    state_dim = 24
    
    # Simulate graph-enhanced by adding temporal adjacency features
    X_enhanced = []
    for i in range(n_samples):
        sample = X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        # Add graph-like features: differences between consecutive states
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        # Concatenate original with graph features
        enhanced = np.concatenate([sample, diffs_padded.flatten()])
        X_enhanced.append(enhanced)
    X_enhanced = np.array(X_enhanced)
    
    model = MLPRegressor(hidden_layer_sizes=(dim, dim // 2), activation='relu',
                         solver='adam', max_iter=600, random_state=42,
                         early_stopping=True, validation_fraction=0.15)
    model.fit(X_enhanced, y)
    
    # Need to enhance test data too
    test_enhanced = []
    for i in range(test_X.shape[0]):
        sample = test_X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        enhanced = np.concatenate([sample, diffs_padded.flatten()])
        test_enhanced.append(enhanced)
    test_enhanced = np.array(test_enhanced)
    
    pred = model.predict(test_enhanced)
    return np.mean((pred - test_y) ** 2)


def run_experiment():
    print("\n=== H1.15: Graph + Unified on Temporal Tasks ===")
    
    train_X, train_y = generate_temporal_data(400, n_timesteps=8, seed=42)
    test_X, test_y = generate_temporal_data(200, n_timesteps=8, seed=789)
    
    print("\nTesting on 8-step temporal tasks:")
    
    baseline_loss = run_baseline(train_X, train_y, test_X, test_y)
    print(f"Baseline: {baseline_loss:.4f}")
    
    unified_loss = run_unified(train_X, train_y, test_X, test_y)
    print(f"Unified (2048): {unified_loss:.4f}")
    
    graph_unified_loss = run_graph_unified(train_X, train_y, test_X, test_y)
    print(f"Graph + Unified: {graph_unified_loss:.4f}")
    
    print("\nTesting on 12-step temporal tasks:")
    train_X12, train_y12 = generate_temporal_data(400, n_timesteps=12, seed=42)
    test_X12, test_y12 = generate_temporal_data(200, n_timesteps=12, seed=789)
    
    baseline_loss12 = run_baseline(train_X12, train_y12, test_X12, test_y12)
    print(f"Baseline: {baseline_loss12:.4f}")
    
    unified_loss12 = run_unified(train_X12, train_y12, test_X12, test_y12)
    print(f"Unified (2048): {unified_loss12:.4f}")
    
    graph_unified_loss12 = run_graph_unified(train_X12, train_y12, test_X12, test_y12)
    print(f"Graph + Unified: {graph_unified_loss12:.4f}")
    
    print("\n=== Results ===")
    print(f"8-step: Baseline={baseline_loss:.4f}, Unified={unified_loss:.4f}, Graph+U={graph_unified_loss:.4f}")
    print(f"12-step: Baseline={baseline_loss12:.4f}, Unified={unified_loss12:.4f}, Graph+U={graph_unified_loss12:.4f}")
    
    results = {
        "8step": {"baseline": baseline_loss, "unified": unified_loss, "graph_unified": graph_unified_loss},
        "12step": {"baseline": baseline_loss12, "unified": unified_loss12, "graph_unified": graph_unified_loss12}
    }
    return results


if __name__ == '__main__':
    results = run_experiment()