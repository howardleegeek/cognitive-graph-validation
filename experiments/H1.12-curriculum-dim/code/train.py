"""
H1.12: Curriculum + Larger Dimensions
Tests if curriculum learning works better with larger model.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_robot_data(n_samples, difficulty=1.0, seed=42):
    """Generate robot data with controllable difficulty."""
    np.random.seed(seed)
    
    state_dim = 24
    action_dim = 8
    
    X = []
    y = []
    
    for _ in range(n_samples):
        state = np.random.randn(state_dim) * 0.5 * difficulty
        action = np.random.randn(action_dim) * 0.2 * difficulty
        
        action_expanded = np.tile(action, 3)[:state_dim]
        noise_level = 0.02 * difficulty
        next_state = state + action_expanded * 0.15 + np.random.randn(state_dim) * noise_level
        
        X.append(np.concatenate([state, action]))
        y.append(next_state)
    
    return np.array(X), np.array(y)


def run_experiment():
    print("\n=== H1.12: Curriculum + Dimension Scaling ===")
    
    train_easy, y_easy = generate_robot_data(200, difficulty=0.5, seed=42)
    train_medium, y_medium = generate_robot_data(200, difficulty=1.0, seed=43)
    train_hard, y_hard = generate_robot_data(200, difficulty=1.5, seed=44)
    
    train_mixed = np.vstack([train_easy, train_medium, train_hard])
    y_mixed = np.vstack([y_easy, y_medium, y_hard])
    
    test_X, test_y = generate_robot_data(200, difficulty=1.5, seed=789)
    
    print(f"Curriculum train: {train_mixed.shape}, Test: {test_X.shape}")
    
    results = {}
    
    for dim, name in [(512, "512_curriculum"), (1024, "1024_curriculum")]:
        model = MLPRegressor(
            hidden_layer_sizes=(dim, dim // 2),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        model.fit(train_mixed, y_mixed)
        pred = model.predict(test_X)
        loss = np.mean((pred - test_y) ** 2)
        
        results[name] = {'dim': dim, 'loss': float(loss)}
        print(f"{name}: MSE = {loss:.4f}")
    
    baseline_512 = MLPRegressor(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    )
    baseline_512.fit(train_hard, y_hard)
    baseline_pred = baseline_512.predict(test_X)
    baseline_loss = np.mean((baseline_pred - test_y) ** 2)
    
    print(f"Baseline (512, hard only): MSE = {baseline_loss:.4f}")
    
    curriculum_improvement = (baseline_loss - results['512_curriculum']['loss']) / baseline_loss * 100
    dim_improvement = (results['512_curriculum']['loss'] - results['1024_curriculum']['loss']) / results['512_curriculum']['loss'] * 100
    
    result = {
        'curriculum_512': results['512_curriculum']['loss'],
        'curriculum_1024': results['1024_curriculum']['loss'],
        'baseline_512': baseline_loss,
        'curriculum_vs_baseline': curriculum_improvement,
        '1024_vs_512': dim_improvement,
        'status': 'supported' if curriculum_improvement > 0 else 'refuted'
    }
    
    print(f"\n=== H1.12 Result ===")
    print(f"Curriculum vs Baseline: {curriculum_improvement:+.1f}%")
    print(f"1024 vs 512: {dim_improvement:+.1f}%")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()