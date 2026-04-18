"""
H2.2: Cross-Embodiment Transfer Test
Tests if cognitive graph architecture transfers across different robot embodiments.
Train on one embodiment (7-DOF), test on another (4-DOF).
"""
import numpy as np
import json
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def generate_robot_data(n_samples, n_joints, seed=42):
    """Generate robot data for different embodiments."""
    np.random.seed(seed)
    
    joint_dim = n_joints * 3  # position, velocity, torque
    state_dim = 16  # ee pose, object pose
    
    X = []
    y = []
    
    for _ in range(n_samples):
        joint_state = np.random.randn(joint_dim) * 0.5
        ee_state = np.random.randn(6) * 0.3
        object_state = np.random.randn(6) * 0.3
        action = np.random.randn(n_joints) * 0.2
        
        next_joint = joint_state[:joint_dim] + np.tile(action, 3) * 0.1 + np.random.randn(joint_dim) * 0.02
        next_ee = ee_state + np.sum(action[:min(6, n_joints)]) * 0.05 + np.random.randn(6) * 0.01
        next_object = object_state + np.random.randn(6) * 0.01
        
        state = np.concatenate([joint_state, ee_state, object_state])
        next_state = np.concatenate([next_joint, next_ee, next_object])
        
        X.append(np.concatenate([state, action]))
        y.append(next_state)
    
    return np.array(X), np.array(y)


def run_experiment():
    """Run H2.2 cross-embodiment experiment."""
    print("\n=== H2.2: Cross-Embodiment Transfer ===")
    
    # Train on 7-DOF arm (source embodiment)
    train_7dof, y_train_7dof = generate_robot_data(500, n_joints=7, seed=42)
    
    # Test on 4-DOF gripper (different embodiment)
    test_4dof_X, test_4dof_y = generate_robot_data(200, n_joints=4, seed=789)
    
    # Also test on same embodiment (7-DOF) for baseline comparison
    test_7dof_X, test_7dof_y = generate_robot_data(200, n_joints=7, seed=789)
    
    print(f"Training: 7-DOF arm, {train_7dof.shape[1]} features")
    print(f"Test 1: 4-DOF gripper (different embodiment)")
    print(f"Test 2: 7-DOF arm (same embodiment)")
    
    # Train cognitive graph style model
    cg_model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate='adaptive'
    )
    cg_model.fit(train_7dof, y_train_7dof)
    
    # Test on same embodiment
    pred_7dof = cg_model.predict(test_7dof_X)
    loss_same = np.mean((pred_7dof - test_7dof_y) ** 2)
    
    # Test on different embodiment
    pred_4dof = cg_model.predict(test_4dof_X)
    loss_diff = np.mean((pred_4dof - test_4dof_y) ** 2)
    
    print(f"\n=== Results ===")
    print(f"Same embodiment (7-DOF -> 7-DOF): MSE = {loss_same:.4f}")
    print(f"Different embodiment (7-DOF -> 4-DOF): MSE = {loss_diff:.4f}")
    print(f"Transfer gap: {((loss_diff - loss_same) / loss_same * 100):+.1f}%")
    
    # Now train a baseline that uses domain adaptation
    # Test with training on mixed embodiments
    train_4dof, y_train_4dof = generate_robot_data(500, n_joints=4, seed=43)
    train_mixed = np.vstack([train_7dof, train_4dof])
    y_train_mixed = np.vstack([y_train_7dof, y_train_4dof])
    
    mixed_model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate='adaptive'
    )
    mixed_model.fit(train_mixed, y_train_mixed)
    
    # Test on 4-DOF (unseen during training)
    pred_4dof_mixed = mixed_model.predict(test_4dof_X)
    loss_mixed = np.mean((pred_4dof_mixed - test_4dof_y) ** 2)
    
    print(f"\n=== Multi-Embodiment Training ===")
    print(f"Mixed training on 7-DOF + 4-DOF:")
    print(f"  Test on 4-DOF: MSE = {loss_mixed:.4f}")
    print(f"  Improvement over single-embodiment: {((loss_diff - loss_mixed) / loss_diff * 100):+.1f}%")
    
    result = {
        'same_embodiment_loss': float(loss_same),
        'different_embodiment_loss': float(loss_diff),
        'transfer_gap_percent': float((loss_diff - loss_same) / loss_same * 100),
        'mixed_training_loss': float(loss_mixed),
        'improvement_from_mixed': float((loss_diff - loss_mixed) / loss_diff * 100),
        'status': 'refuted' if loss_diff > loss_same * 1.1 else 'supported'
    }
    
    print(f"\n=== H2.2 Result ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_experiment()