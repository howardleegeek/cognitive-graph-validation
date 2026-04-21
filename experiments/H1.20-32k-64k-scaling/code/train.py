"""
H1.20: Test 32k dimensions with optimal regularization - FAST VERSION
Hypothesis: With proper regularization, scaling continues beyond 16k
Based on H1.19: 16384 > 8192 > 4096 with α=0.1
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


def run_with_regularization(X, y, test_X, test_y, hidden_sizes, alpha=0.1, max_iter=200):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10
    )
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_experiment():
    print("\n=== H1.20: 32k Scaling (Fast) ===", flush=True)
    
    # Smaller data for faster runs
    train_X, train_y = generate_temporal_data(200, n_timesteps=8, seed=42)
    test_X, test_y = generate_temporal_data(100, n_timesteps=8, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}", flush=True)
    
    results = {}
    
    # Baseline: 4096, 8192, 16384
    print("\n--- Baselines ---", flush=True)
    for dims, name in [((1024, 512), "4096"), ((2048, 1024), "8192"), ((4096, 2048), "16384")]:
        loss = run_with_regularization(train_X, train_y, test_X, test_y, dims, alpha=0.1)
        print(f"{name}: MSE={loss:.4f}", flush=True)
        results[name] = loss
    
    # 32k with varying regularization
    print("\n--- 32768 with α ---", flush=True)
    for alpha in [0.1, 0.3, 0.5]:
        try:
            loss = run_with_regularization(train_X, train_y, test_X, test_y, (8192, 4096), alpha=alpha, max_iter=250)
            print(f"32768 α={alpha}: MSE={loss:.4f}", flush=True)
            results[f"32768_a{alpha}"] = loss
        except Exception as e:
            print(f"32768 α={alpha}: Failed - {e}", flush=True)
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    print("\n=== Summary (sorted by MSE) ===", flush=True)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for k, v in sorted_results:
        print(f"{k}: {v:.4f}", flush=True)
    
    best = sorted_results[0]
    print(f"\nBest: {best[0]} with MSE={best[1]:.4f}", flush=True)