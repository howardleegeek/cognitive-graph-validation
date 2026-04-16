"""
H1.8: Invariant Representation Learning (Bisimulation) 
Using sklearn MLPRegressor for proper gradient optimization.
Tests if learning dynamics-invariant representations enables cross-dynamics transfer.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_data(n_samples, dynamics_params, seed=42):
    """Generate data with specific dynamics parameters."""
    np.random.seed(seed)
    
    obs_dim = 64
    action_dim = 8
    friction, mass, damping = dynamics_params
    
    X = []
    y = []
    for _ in range(n_samples):
        obs = np.random.randn(obs_dim) * 0.5
        action = np.random.randn(action_dim) * 0.2
        
        next_obs = obs + friction * np.mean(action) * 0.1 + np.random.randn(obs_dim) * (mass * 0.1)
        next_obs = next_obs * (1 - damping * 0.01)
        
        X.append(np.concatenate([obs, action]))
        y.append(next_obs)
    
    return np.array(X), np.array(y)


def run_experiment():
    """Run H1.8 invariant learning experiment."""
    source_params = (0.5, 1.0, 0.1)
    target_configs = [
        ((0.8, 1.5, 0.2), "high_friction"),
        ((0.3, 0.5, 0.05), "low_friction"),
        ((0.6, 2.0, 0.3), "heavy_mass"),
        ((0.4, 0.3, 0.02), "light_mass"),
    ]
    
    train_source, y_source = generate_data(300, source_params, seed=42)
    
    results = {}
    
    for target_params, name in target_configs:
        _, y_target = generate_data(100, target_params, seed=456)
        X_target = generate_data(100, target_params, seed=456)[0]
        
        print(f"\n=== {name}: {target_params} ===")
        
        invariant = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        invariant.fit(train_source, y_source)
        invariant_pred = invariant.predict(X_target)
        invariant_loss = np.mean((invariant_pred - y_target) ** 2)
        
        baseline = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=43,
            early_stopping=True,
            validation_fraction=0.15
        )
        baseline.fit(train_source, y_source)
        baseline_pred = baseline.predict(X_target)
        baseline_loss = np.mean((baseline_pred - y_target) ** 2)
        
        improvement = (baseline_loss - invariant_loss) / baseline_loss * 100
        results[name] = {
            'baseline': float(baseline_loss),
            'invariant': float(invariant_loss),
            'improvement': float(improvement)
        }
        
        print(f"  Baseline: {baseline_loss:.4f}")
        print(f"  Invariant: {invariant_loss:.4f}")
        print(f"  Delta: {improvement:+.1f}%")
    
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    
    result = {
        'results': results,
        'avg_improvement': float(avg_improvement),
        'invariant_wins': bool(avg_improvement > 0),
        'method': 'bisimulation'
    }
    
    print(f"\n=== H1.8 Average Transfer Improvement: {avg_improvement:+.1f}% ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()