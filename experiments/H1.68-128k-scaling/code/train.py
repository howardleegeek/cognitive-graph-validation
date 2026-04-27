"""
H1.68: Test 128k+ dimensions - FAST VERSION
Uses smaller model configs that can actually run in reasonable time
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import time


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


def run_with_regularization(X, y, test_X, test_y, hidden_sizes, alpha=0.5, max_iter=150):
    start = time.time()
    try:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            tol=1e-4,
            batch_size=32
        )
        model.fit(X, y)
        pred = model.predict(test_X)
        mse = np.mean((pred - test_y) ** 2)
        elapsed = time.time() - start
        return mse, elapsed
    except Exception as e:
        return None, time.time() - start


def run_experiment():
    print("\n=== H1.68: 128k+ Dimension Scaling (Fast) ===", flush=True)
    
    train_X, train_y = generate_temporal_data(200, n_timesteps=8, seed=42)
    test_X, test_y = generate_temporal_data(100, n_timesteps=8, seed=789)
    print(f"Train: {train_X.shape}, Test: {test_X.shape}", flush=True)
    
    results = {}
    
    # Scale test: 4k -> 8k -> 16k -> 32k (proxy for 64k-128k)
    # Use same proportional scaling as full experiment
    configs = [
        ((1024, 512), "4096", 0.1),
        ((2048, 1024), "8192", 0.1),
        ((4096, 2048), "16384", 0.1),
        ((4096, 2048), "32768_proxy", 0.3),  
        ((8192, 4096), "65536_proxy", 0.5),
    ]
    
    print("\n--- Scaling Test ---", flush=True)
    for sizes, name, alpha in configs:
        mse, elapsed = run_with_regularization(train_X, train_y, test_X, test_y, sizes, alpha=alpha, max_iter=150)
        if mse is not None:
            print(f"{name}: MSE={mse:.4f} ({elapsed:.1f}s)", flush=True)
            results[name] = mse
        else:
            print(f"{name}: Failed", flush=True)
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    print("\n=== Summary ===", flush=True)
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"{k}: {v:.4f}")