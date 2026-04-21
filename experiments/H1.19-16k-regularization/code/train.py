"""
H1.19: Test regularization on 16k dimensions
Hypothesis: With α≥0.1, larger models can scale beyond 4096
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


def run_with_regularization(X, y, test_X, test_y, hidden_sizes, alpha=0.1, max_iter=400):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
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
    print("\n=== H1.19: Regularization Enables 16k ===", flush=True)
    
    # Smaller data to run faster
    train_X, train_y = generate_temporal_data(300, n_timesteps=8, seed=42)
    test_X, test_y = generate_temporal_data(150, n_timesteps=8, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}", flush=True)
    
    results = {}
    
    # 4096 baseline
    print("\n--- 4096 baseline ---", flush=True)
    loss_4096 = run_with_regularization(train_X, train_y, test_X, test_y, (1024, 512), alpha=0.01)
    print(f"4096: MSE={loss_4096:.4f}", flush=True)
    results["4096"] = loss_4096
    
    # 8192 with regularization
    print("\n--- 8192 with α ---", flush=True)
    for alpha in [0.1, 0.5, 1.0]:
        loss = run_with_regularization(train_X, train_y, test_X, test_y, (2048, 1024), alpha=alpha)
        print(f"8192 α={alpha}: MSE={loss:.4f}", flush=True)
        results[f"8192_{alpha}"] = loss
    
    # 16384 with stronger regularization
    print("\n--- 16384 with α ---", flush=True)
    for alpha in [0.1, 0.5, 1.0]:
        try:
            loss = run_with_regularization(train_X, train_y, test_X, test_y, (4096, 2048), alpha=alpha)
            print(f"16384 α={alpha}: MSE={loss:.4f}", flush=True)
            results[f"16384_{alpha}"] = loss
        except Exception as e:
            print(f"16384 α={alpha}: Failed", flush=True)
    
    # 32768
    print("\n--- 32768 with α ---", flush=True)
    for alpha in [0.5, 1.0, 2.0]:
        try:
            loss = run_with_regularization(train_X, train_y, test_X, test_y, (8192, 4096), alpha=alpha)
            print(f"32768 α={alpha}: MSE={loss:.4f}", flush=True)
            results[f"32768_{alpha}"] = loss
        except Exception as e:
            print(f"32768 α={alpha}: Failed", flush=True)
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    print("\n=== Summary ===", flush=True)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for k, v in sorted_results:
        print(f"{k}: {v:.4f}", flush=True)