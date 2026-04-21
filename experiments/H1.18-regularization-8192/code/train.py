"""
H1.18: Test stronger regularization to enable 8192 without overfitting
Hypothesis: With proper regularization (dropout, weight decay), 8192 can beat 4096
Using smaller model to fit runtime constraints
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


def run_with_regularization(X, y, test_X, test_y, hidden_sizes, alpha=0.001, max_iter=500):
    """alpha is L2 regularization (weight decay)"""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,  # L2 regularization
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15
    )
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_experiment():
    print("\n=== H1.18: Regularization to Enable 8192 ===", flush=True)
    
    # Smaller data to run faster
    train_X, train_y = generate_temporal_data(300, n_timesteps=8, seed=42)
    test_X, test_y = generate_temporal_data(150, n_timesteps=8, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}", flush=True)
    sys.stdout.flush()
    
    results = {}
    
    # Test 4096 vs 8192 with different regularization
    print("\n--- 4096 baseline ---", flush=True)
    for alpha in [0.0001, 0.001, 0.01]:
        loss = run_with_regularization(train_X, train_y, test_X, test_y, (1024, 512), alpha=alpha)
        print(f"alpha={alpha}: MSE={loss:.4f}", flush=True)
        results[f"4096_a{alpha}"] = loss
    
    print("\n--- 8192 test ---", flush=True)
    for alpha in [0.001, 0.01, 0.1]:
        loss = run_with_regularization(train_X, train_y, test_X, test_y, (2048, 1024), alpha=alpha)
        print(f"alpha={alpha}: MSE={loss:.4f}", flush=True)
        results[f"8192_a{alpha}"] = loss
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    print("\n=== Summary ===", flush=True)
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"{k}: {v:.4f}", flush=True)