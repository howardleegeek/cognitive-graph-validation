"""
H1.9: Multi-Task Training with Diverse Dynamics (EZ-M Approach)
Tests if training on diverse dynamics improves transfer to unseen dynamics.
"""
import numpy as np
import json


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
    """Run H1.9 multi-task dynamics experiment."""
    source_params = (0.5, 1.0, 0.1)
    target_configs = [
        ((0.8, 1.5, 0.2), "high_friction"),
        ((0.3, 0.5, 0.05), "low_friction"),
        ((0.6, 2.0, 0.3), "heavy_mass"),
        ((0.4, 0.3, 0.02), "light_mass"),
    ]
    
    train_single, y_single = generate_data(300, source_params, seed=42)
    
    from sklearn.neural_network import MLPRegressor
    
    results = {}
    
    for target_params, name in target_configs:
        _, y_target = generate_data(100, target_params, seed=456)
        X_target = generate_data(100, target_params, seed=456)[0]
        
        print(f"\n=== {name}: {target_params} ===")
        
        single_model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        single_model.fit(train_single, y_single)
        single_pred = single_model.predict(X_target)
        single_loss = np.mean((single_pred - y_target) ** 2)
        
        all_dynamics = [
            (0.5, 1.0, 0.1),
            (0.6, 0.8, 0.08),
            (0.4, 1.2, 0.12),
        ]
        train_multi = []
        y_multi = []
        for dp in all_dynamics:
            X, y = generate_data(100, dp, seed=hash(dp) % 10000)
            train_multi.append(X)
            y_multi.append(y)
        train_multi = np.vstack(train_multi)
        y_multi = np.vstack(y_multi)
        
        multi_model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        multi_model.fit(train_multi, y_multi)
        multi_pred = multi_model.predict(X_target)
        multi_loss = np.mean((multi_pred - y_target) ** 2)
        
        improvement = (single_loss - multi_loss) / single_loss * 100
        results[name] = {
            'single': float(single_loss),
            'multi': float(multi_loss),
            'improvement': float(improvement)
        }
        
        print(f"  Single-task: {single_loss:.4f}")
        print(f"  Multi-task: {multi_loss:.4f}")
        print(f"  Delta: {improvement:+.1f}%")
    
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    
    result = {
        'results': results,
        'avg_improvement': float(avg_improvement),
        'multi_wins': bool(avg_improvement > 0),
        'method': 'multitask_dynamics'
    }
    
    print(f"\n=== H1.9 Average: {avg_improvement:+.1f}% ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()