"""
H1.144: Hybrid Architecture - Concatenation for Simple, Attention for Complex

Based on findings:
- H3: Concatenation wins on simple tasks
- H3.2: Graph attention helps on 16+ step tasks (+5.8%)
- H3.4: Attention marginally helps on very long sequences (24, 30 steps)

Hypothesis: Task-complexity-aware hybrid architecture outperforms uniform approaches
"""

import numpy as np
import json
import os
from datetime import datetime

np.random.seed(42)

def generate_complex_trajectory(n_steps, n_objects=3):
    """Generate complex multi-step manipulation trajectory"""
    t = np.linspace(0, 1, n_steps)
    
    n_joints = 7
    joint_trajectory = np.zeros((n_steps, n_joints))
    for j in range(n_joints):
        freq = 0.5 + j * 0.1
        phase = j * 0.3
        amplitude = 0.5 + np.random.random() * 0.5
        joint_trajectory[:, j] = amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    ee_positions = np.cumsum(np.random.randn(n_steps, 3) * 0.02, axis=0)
    ee_positions = ee_positions - ee_positions[0]
    
    object_positions = np.zeros((n_steps, n_objects, 3))
    for obj in range(n_objects):
        obj_start = np.random.randn(3) * 0.3
        obj_velocity = np.random.randn(3) * 0.01
        for step in range(n_steps):
            if step > 0:
                obj_velocity += np.random.randn(3) * 0.005
                obj_velocity *= 0.95
            object_positions[step, obj] = object_positions[step-1, obj] + obj_velocity
            if step < n_steps // 3:
                object_positions[step, obj] = obj_start + (step / (n_steps // 3)) * np.random.randn(3) * 0.1
    
    instruction_tokens = np.random.randint(0, 100, size=10)
    
    gripper = np.zeros(n_steps)
    open_idx = n_steps // 4
    close_idx = 3 * n_steps // 4
    gripper[open_idx:close_idx] = 1
    
    return {
        'joints': joint_trajectory,
        'ee_positions': ee_positions,
        'object_positions': object_positions,
        'instruction_tokens': instruction_tokens,
        'gripper': gripper
    }

def create_state_action_pairs(trajectories, n_steps):
    """Create state-action pairs"""
    states = []
    actions = []
    
    for traj in trajectories:
        for i in range(len(traj['joints']) - n_steps):
            state = np.concatenate([
                traj['joints'][i].flatten(),
                traj['ee_positions'][i].flatten(),
                traj['object_positions'][i].reshape(-1),
                traj['instruction_tokens'],
                traj['gripper'][i:i+3]
            ])
            action = traj['joints'][i+n_steps]
            states.append(state)
            actions.append(action)
    
    return np.array(states), np.array(actions)

class ConcatenationModel:
    """Baseline: concatenation fusion"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
    
    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
    
    def train(self, X, y, lr=0.001, epochs=100):
        for epoch in range(epochs):
            pred = self.forward(X)
            error = y - pred
            grad2 = error.T @ np.tanh(X @ self.W1 + self.b1) / len(X)
            self.W2 += lr * grad2.T
            self.b2 += lr * error.mean(axis=0)
            grad1 = (error @ self.W2.T) * (1 - np.tanh(X @ self.W1 + self.b1)**2)
            self.W1 += lr * (grad1.T @ X).T / len(X)
            self.b1 += lr * grad1.mean(axis=0)
    
    def predict(self, X):
        return self.forward(X)

class AttentionModel:
    """Attention model for complex tasks"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        self.hidden_dim = hidden_dim
        self.W_q = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_k = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_v = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_out = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b_out = np.zeros(output_dim)
    
    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        scores = Q @ K.T / np.sqrt(self.hidden_dim)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-8)
        context = attn_weights @ V
        return context @ self.W_out + self.b_out
    
    def train(self, X, y, lr=0.001, epochs=100):
        for epoch in range(epochs):
            pred = self.forward(X)
            error = y - pred
            grad = error.T @ np.ones((X.shape[0], self.hidden_dim)) / len(X)
            self.W_out += lr * grad.T
            self.b_out += lr * error.mean(axis=0)
    
    def predict(self, X):
        return self.forward(X)

class HybridModel:
    """Hybrid: uses concatenation for simple, attention for complex"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, threshold=20):
        self.concat_model = ConcatenationModel(input_dim, output_dim, hidden_dim)
        self.attention_model = AttentionModel(input_dim, output_dim, hidden_dim)
        self.threshold = threshold
    
    def predict(self, X, n_steps):
        """Use attention for complex tasks (n_steps > threshold)"""
        use_attention = n_steps > self.threshold
        
        if use_attention:
            return self.attention_model.predict(X)
        else:
            return self.concat_model.predict(X)
    
    def train(self, X, y, n_steps, lr=0.001, epochs=100):
        use_attention = n_steps > self.threshold
        
        if use_attention:
            self.attention_model.train(X, y, lr, epochs)
        else:
            self.concat_model.train(X, y, lr, epochs)

def run_experiment():
    """Run H1.144 experiment"""
    print("=" * 60)
    print("H1.144: Hybrid Architecture - Concatenation/Attention")
    print("=" * 60)
    
    results = {
        'experiment': 'H1.144',
        'hypothesis': 'Hybrid concatenation + attention based on task complexity',
        'timestamp': datetime.now().isoformat(),
        'task_lengths': [],
        'concat_mses': [],
        'attention_mses': [],
        'hybrid_mses': [],
        'improvements': []
    }
    
    # Test different task complexities with different thresholds
    test_configs = [
        (10, 'simple'),
        (15, 'medium'),
        (20, 'boundary'),
        (25, 'complex'),
        (30, 'very_complex'),
        (40, 'ultra_complex'),
        (50, 'extreme')
    ]
    
    for n_steps, complexity in test_configs:
        print(f"\n--- Testing {complexity} ({n_steps} steps) ---")
        
        # Generate data - reduced for speed
        np.random.seed(42 + n_steps)
        train_trajectories = [generate_complex_trajectory(n_steps + 20) for _ in range(50)]
        X_train, y_train = create_state_action_pairs(train_trajectories, n_steps)
        
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - X_mean) / X_std
        y_train_norm = (y_train - y_mean) / y_std
        
        np.random.seed(123 + n_steps)
        test_trajectories = [generate_complex_trajectory(n_steps + 20) for _ in range(50)]
        X_test, y_test = create_state_action_pairs(test_trajectories, n_steps)
        X_test_norm = (X_test - X_mean) / X_std
        
        # Train concatenation model
        concat = ConcatenationModel(X_train.shape[1], y_train.shape[1])
        concat.train(X_train_norm, y_train_norm, lr=0.01, epochs=20)
        concat_pred = concat.predict(X_test_norm)
        concat_mse = np.mean((concat_pred * y_std - y_test) ** 2)
        
        # Train attention model
        attn = AttentionModel(X_train.shape[1], y_train.shape[1])
        attn.train(X_train_norm, y_train_norm, lr=0.01, epochs=20)
        attn_pred = attn.predict(X_test_norm)
        attn_mse = np.mean((attn_pred * y_std - y_test) ** 2)
        
        # Train hybrid model (threshold=20)
        hybrid = HybridModel(X_train.shape[1], y_train.shape[1], threshold=20)
        hybrid.train(X_train_norm, y_train_norm, n_steps, lr=0.01, epochs=20)
        hybrid_pred = hybrid.predict(X_test_norm, n_steps)
        hybrid_mse = np.mean((hybrid_pred * y_std - y_test) ** 2)
        
        # Choose best for improvement calculation
        best_baseline = min(concat_mse, attn_mse)
        improvement = (best_baseline - hybrid_mse) / best_baseline * 100
        
        print(f"  Concat MSE: {concat_mse:.6f}")
        print(f"  Attention MSE: {attn_mse:.6f}")
        print(f"  Hybrid MSE: {hybrid_mse:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results['task_lengths'].append(n_steps)
        results['concat_mses'].append(float(concat_mse))
        results['attention_mses'].append(float(attn_mse))
        results['hybrid_mses'].append(float(hybrid_mse))
        results['improvements'].append(float(improvement))
    
    # Summary
    avg_improvement = np.mean(results['improvements'])
    print("\n" + "=" * 60)
    print(f"AVERAGE IMPROVEMENT: {avg_improvement:+.1f}%")
    print("=" * 60)
    
    results['average_improvement'] = float(avg_improvement)
    
    # Determine status
    if avg_improvement > 10:
        results['status'] = 'SUPPORTED'
        results['conclusion'] = f'Hybrid architecture achieves {avg_improvement:.1f}% improvement'
    elif avg_improvement > 0:
        results['status'] = 'PARTIAL'
        results['conclusion'] = f'Marginal {avg_improvement:.1f}% improvement'
    else:
        results['status'] = 'REFUTED'
        results['conclusion'] = f'Hybrid does not improve over baseline ({avg_improvement:.1f}%)'
    
    # Save results
    results_dir = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation'
    with open(f'{results_dir}/experiments/H1.144-hybrid-complexity-aware/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/H1.144-hybrid-complexity-aware/results.json")
    print(f"Status: {results['status']}")
    
    return results

if __name__ == '__main__':
    results = run_experiment()