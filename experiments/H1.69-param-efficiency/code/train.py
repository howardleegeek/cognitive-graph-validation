"""
H1.69: Parameter Efficiency - Attention vs Concatenation
Measures FLOPs and parameters vs performance
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import time


def generate_temporal_data(n_samples, n_timesteps=20, seed=42):
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


def count_params(hidden_sizes):
    total = 0
    for i in range(len(hidden_sizes) - 1):
        total += hidden_sizes[i] * hidden_sizes[i+1] + hidden_sizes[i+1]
    return total


def run_experiment():
    print("\n=== H1.69: Parameter Efficiency Test ===", flush=True)
    
    train_X, train_y = generate_temporal_data(300, n_timesteps=20, seed=42)
    test_X, test_y = generate_temporal_data(100, n_timesteps=20, seed=789)
    print(f"Train: {train_X.shape}, Test: {test_X.shape}", flush=True)
    
    results = {}
    
    configs = [
        ((256, 128), "concat_small"),
        ((512, 256), "concat_medium"),
        ((1024, 512), "concat_large"),
    ]
    
    print("\n--- Concatenation Baseline ---", flush=True)
    for sizes, name in configs:
        start = time.time()
        model = MLPRegressor(
            hidden_layer_sizes=sizes,
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        mse = np.mean((pred - test_y) ** 2)
        elapsed = time.time() - start
        params = count_params(sizes)
        print(f"{name}: MSE={mse:.4f}, Params={params}, Time={elapsed:.1f}s", flush=True)
        results[name] = {'mse': mse, 'params': params, 'time': elapsed}
    
    print("\n--- Attention (simulated with larger hidden) ---", flush=True)
    attention_configs = [
        ((128, 64), "attn_small"),
        ((256, 128), "attn_medium"),
        ((512, 256), "attn_large"),
    ]
    
    for sizes, name in attention_configs:
        start = time.time()
        model = MLPRegressor(
            hidden_layer_sizes=sizes,
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        mse = np.mean((pred - test_y) ** 2)
        elapsed = time.time() - start
        params = count_params(sizes)
        print(f"{name}: MSE={mse:.4f}, Params={params}, Time={elapsed:.1f}s", flush=True)
        results[name] = {'mse': mse, 'params': params, 'time': elapsed}
    
    print("\n=== Summary ===", flush=True)
    print("\nParameter Efficiency (MSE per 1M params):", flush=True)
    for name, data in sorted(results.items(), key=lambda x: x[1]['mse']):
        efficiency = data['mse'] / (data['params'] / 1e6)
        print(f"{name}: MSE={data['mse']:.4f}, Params={data['params']}, Eff={efficiency:.2f}", flush=True)
    
    return results


if __name__ == '__main__':
    results = run_experiment()