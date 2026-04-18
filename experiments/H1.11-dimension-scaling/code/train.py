"""
H1.11: Dimension Scaling Test
Tests if 512 is optimal or if 256/1024 performs better.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_robot_data(n_samples, seed=42):
    """Generate standard robot data."""
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
    print("\n=== H1.11: Dimension Scaling Test ===")
    
    train_X, train_y = generate_robot_data(500, seed=42)
    test_X, test_y = generate_robot_data(200, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")
    
    results = {}
    
    configs = [
        (256, "256_total"),
        (512, "512_total"),
        (1024, "1024_total"),
    ]
    
    for total_dim, name in configs:
        model = MLPRegressor(
            hidden_layer_sizes=(total_dim, total_dim // 2),
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
        
        results[name] = {
            'total_dim': total_dim,
            'loss': float(loss)
        }
        print(f"{name}: MSE = {loss:.4f}")
    
    best = min(results.items(), key=lambda x: x[1]['loss'])
    print(f"\nBest: {best[0]} with MSE = {best[1]['loss']:.4f}")
    
    improvement_512 = (results['256_total']['loss'] - results['512_total']['loss']) / results['256_total']['loss'] * 100
    improvement_1024 = (results['1024_total']['loss'] - results['512_total']['loss']) / results['1024_total']['loss'] * 100
    
    result = {
        'results': results,
        'best_dim': best[0],
        'best_loss': best[1]['loss'],
        '512_vs_256': improvement_512,
        '512_vs_1024': improvement_1024,
        'status': 'supported' if results['512_total']['loss'] <= results['256_total']['loss'] else 'refuted'
    }
    
    print(f"\n=== H1.11 Result ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()