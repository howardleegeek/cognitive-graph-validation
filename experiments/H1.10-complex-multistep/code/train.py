"""
H1.10: Complex Multi-Step Tasks with Composition
Tests unified architecture on 7+ step tasks requiring compositional reasoning.
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor


def generate_complex_task_data(n_samples, n_steps=7, seed=42):
    """Generate complex multi-step tasks requiring compositional reasoning."""
    np.random.seed(seed)
    
    state_dim = 12
    action_dim = 6
    n_objects = 4
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Initial state: 4 objects with positions
        objects = np.random.randn(n_objects, state_dim) * 0.5
        
        # Sequence of actions (7 steps)
        actions = np.random.randn(n_steps, action_dim) * 0.2
        
        # Complex dynamics: each object responds differently to actions
        next_states = []
        current = objects.copy()
        
        for t in range(n_steps):
            next_obj = current.copy()
            
            # Object 0: follows first 2 action dims
            next_obj[0] += np.concatenate([actions[t, :2], np.zeros(state_dim - 2)]) * 0.2
            
            # Object 1: follows action dims 2-4
            next_obj[1] += np.concatenate([np.zeros(2), actions[t, 2:4], np.zeros(state_dim - 4)]) * 0.2
            
            # Object 2: follows all actions but with damping
            action_repeated = np.tile(actions[t], 2)[:state_dim]
            next_obj[2] += action_repeated * 0.15
            
            # Object 3: static (object permanence - doesn't move)
            next_obj[3] += 0
            
            next_states.append(next_obj.flatten())
            current = next_obj
        
        # Input: initial state + full action sequence
        input_state = objects.flatten()
        input_actions = actions.flatten()
        
        X.append(np.concatenate([input_state, input_actions]))
        y.append(np.array(next_states).flatten())
    
    return np.array(X), np.array(y)


def run_experiment():
    """Run H1.10 complex multi-step tasks."""
    print("\n=== H1.10: Complex Multi-Step Tasks (7+ steps) ===")
    
    train_X, train_y = generate_complex_task_data(500, n_steps=7, seed=42)
    test_X, test_y = generate_complex_task_data(200, n_steps=7, seed=789)
    
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"Predicting {train_y.shape[1]} values (4 objects x 7 steps x 12 dims)")
    
    # Baseline: simple concatenation
    baseline_model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    )
    baseline_model.fit(train_X, train_y)
    baseline_pred = baseline_model.predict(test_X)
    baseline_loss = np.mean((baseline_pred - test_y) ** 2)
    
    # Cognitive Graph style: separate physical + semantic branches
    # Physical: first 48 dims (4 objects x 12), Semantic: rest
    
    def split_physical_semantic(X):
        n_samples = X.shape[0]
        physical_dim = 48  # 4 objects x 12 dims
        semantic_dim = X.shape[1] - physical_dim
        
        physical = X[:, :physical_dim]
        semantic = X[:, physical_dim:]
        return physical, semantic
    
    train_phys, train_sem = split_physical_semantic(train_X)
    test_phys, test_sem = split_physical_semantic(test_X)
    
    # Two-branch model
    phys_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=43,
        early_stopping=True,
        validation_fraction=0.15
    )
    phys_model.fit(train_phys, train_y)
    phys_pred = phys_model.predict(test_phys)
    phys_loss = np.mean((phys_pred - test_y) ** 2)
    
    sem_model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=44,
        early_stopping=True,
        validation_fraction=0.15
    )
    sem_model.fit(train_sem, train_y)
    sem_pred = sem_model.predict(test_sem)
    sem_loss = np.mean((sem_pred - test_y) ** 2)
    
    # Fusion: combine both
    combined_X = np.hstack([
        phys_model.predict(train_X[:, :48]),
        sem_model.predict(train_X[:, 48:])
    ])
    combined_test_X = np.hstack([
        phys_model.predict(test_X[:, :48]),
        sem_model.predict(test_X[:, 48:])
    ])
    
    fusion_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=45,
        early_stopping=True,
        validation_fraction=0.15
    )
    fusion_model.fit(combined_X, train_y)
    fusion_pred = fusion_model.predict(combined_test_X)
    fusion_loss = np.mean((fusion_pred - test_y) ** 2)
    
    print(f"\n=== Results ===")
    print(f"Baseline (single branch): {baseline_loss:.4f}")
    print(f"Physical branch only: {phys_loss:.4f}")
    print(f"Semantic branch only: {sem_loss:.4f}")
    print(f"Fusion (two-branch): {fusion_loss:.4f}")
    
    improvement = (baseline_loss - fusion_loss) / baseline_loss * 100
    
    print(f"\nFusion vs Baseline: {improvement:+.1f}%")
    
    result = {
        'baseline_loss': float(baseline_loss),
        'physical_loss': float(phys_loss),
        'semantic_loss': float(sem_loss),
        'fusion_loss': float(fusion_loss),
        'improvement_percent': float(improvement),
        'status': 'supported' if improvement > 0 else 'refuted'
    }
    
    print(f"\n=== H1.10 Result ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()