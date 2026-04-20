"""
H1.14: Test 4096 dimensions
Push dimension scaling even further.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_robot_data(n_samples, seed=42):
    np.random.seed(seed)
    state_dim = 24
    action_dim = 8
    X = []
    y = []
    for _ in range(n_samples):
        state = np.random.randn(state_dim) * 0.5
        action = np.random.randn(action_dim) * 0.2
        action_expanded = np.tile(action, 3)[:state_dim]
        next_state = state + action_expanded * 0.15 + np.random.randn(state_dim) * 0.02
        X.append(np.concatenate([state, action]))
        y.append(next_state)
    return np.array(X), np.array(y)


def run_experiment():
    print("\n=== H1.14: Test 4096 Dimensions ===")
    train_X, train_y = generate_robot_data(600, seed=42)
    test_X, test_y = generate_robot_data(200, seed=789)
    
    results = {}
    for dim, name in [(512, "512"), (1024, "1024"), (2048, "2048"), (4096, "4096")]:
        model = MLPRegressor(
            hidden_layer_sizes=(dim, dim // 2),
            activation='relu',
            solver='adam',
            max_iter=600,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        loss = np.mean((pred - test_y) ** 2)
        results[name] = {'dim': dim, 'loss': float(loss)}
        print(f"{name}: MSE = {loss:.4f}")
    
    best = min(results.items(), key=lambda x: x[1]['loss'])
    print(f"\n=== Best: {best[0]} with MSE = {best[1]['loss']:.4f} ===")
    return results


if __name__ == '__main__':
    result = run_experiment()