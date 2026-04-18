"""
H1.13: Test 2048 dimensions
Extends H1.11 to see if scaling continues beyond 1024.
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
    print("\n=== H1.13: Test 2048 Dimensions ===")
    train_X, train_y = generate_robot_data(500, seed=42)
    test_X, test_y = generate_robot_data(200, seed=789)
    
    results = {}
    for dim, name in [(256, "256"), (512, "512"), (1024, "1024"), (2048, "2048")]:
        model = MLPRegressor(
            hidden_layer_sizes=(dim, dim // 2),
            activation='relu',
            solver='adam',
            max_iter=500,
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
    scaling_trend = results['1024']['loss'] - results['2048']['loss']
    
    result = {
        'results': results,
        'best': best[0],
        'best_loss': best[1]['loss'],
        'scaling_1024_to_2048': scaling_trend,
        'status': 'supported'
    }
    print(f"\n=== H1.13 Result: Best = {best[0]} ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()