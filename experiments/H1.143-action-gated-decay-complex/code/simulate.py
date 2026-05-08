"""
H1.143: Action-Gated Attention with Query-Key Decay on Complex Multi-Step Tasks

Combines best elements from successful experiments:
- H1.39: Action-gated attention (+30% over standard)
- H1.40: Query-key decay attention (+30% over standard)
- H1.41: +99% on real robot complex multi-step

Hypothesis: Combined action-gated + decay attention outperforms on complex multi-step tasks
"""

import numpy as np
import json
import os
from datetime import datetime

np.random.seed(42)

def generate_complex_trajectory(n_steps, n_objects=3):
    """Generate complex multi-step manipulation trajectory with multiple objects"""
    t = np.linspace(0, 1, n_steps)
    
    # Robot state: joint angles + end-effector pose
    n_joints = 7
    joint_trajectory = np.zeros((n_steps, n_joints))
    for j in range(n_joints):
        freq = 0.5 + j * 0.1
        phase = j * 0.3
        amplitude = 0.5 + np.random.random() * 0.5
        joint_trajectory[:, j] = amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # End-effector positions
    ee_positions = np.cumsum(np.random.randn(n_steps, 3) * 0.02, axis=0)
    ee_positions = ee_positions - ee_positions[0]
    
    # Object positions (multiple objects being manipulated)
    object_positions = np.zeros((n_steps, n_objects, 3))
    for obj in range(n_objects):
        obj_start = np.random.randn(3) * 0.3
        obj_velocity = np.random.randn(3) * 0.01
        for step in range(n_steps):
            if step > 0:
                obj_velocity += np.random.randn(3) * 0.005
                obj_velocity *= 0.95  # damping
            object_positions[step, obj] = object_positions[step-1, obj] + obj_velocity
            if step < n_steps // 3:
                object_positions[step, obj] = obj_start + (step / (n_steps // 3)) * np.random.randn(3) * 0.1
    
    # Language instructions (embedded as tokens)
    instruction_tokens = np.random.randint(0, 100, size=10)
    
    # Gripper state
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
    """Create state-action pairs with proper temporal structure"""
    states = []
    actions = []
    
    for traj in trajectories:
        for i in range(len(traj['joints']) - n_steps):
            # State: joints + ee + objects + instructions (compact representation)
            state = np.concatenate([
                traj['joints'][i].flatten(),  # current joints only
                traj['ee_positions'][i].flatten(),  # current ee
                traj['object_positions'][i].reshape(-1),  # current objects
                traj['instruction_tokens'],
                traj['gripper'][i:i+3]  # recent gripper
            ])
            
            # Action: next joint positions
            action = traj['joints'][i+n_steps]
            
            states.append(state)
            actions.append(action)
    
    return np.array(states), np.array(actions)

class BaselineModel:
    """Baseline: simple MLP with concatenation fusion"""
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
            # Simplified gradient - just update output layer
            grad2 = error.T @ np.tanh(X @ self.W1 + self.b1) / len(X)
            self.W2 += lr * grad2.T
            self.b2 += lr * error.mean(axis=0)
            # Update input layer slightly
            grad1 = (error @ self.W2.T) * (1 - np.tanh(X @ self.W1 + self.b1)**2)
            self.W1 += lr * (grad1.T @ X).T / len(X)
            self.b1 += lr * grad1.mean(axis=0)
    
    def predict(self, X):
        return self.forward(X)

class AttentionModel:
    """Simplified attention model with action-gating and decay"""
    def __init__(self, input_dim, output_dim, hidden_dim=128, decay=0.7):
        self.hidden_dim = hidden_dim
        self.decay = decay
        
        # Simple attention: query-key-value projections
        self.W_q = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_k = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_v = np.random.randn(input_dim, hidden_dim) * 0.01
        
        # Action gating
        self.W_gate = np.random.randn(hidden_dim, hidden_dim) * 0.01
        
        # Output projection
        self.W_out = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b_out = np.zeros(output_dim)
    
    def attention_with_decay(self, Q, K, V, action_context=None):
        """Simplified attention with decay"""
        # Apply decay to keys
        scale = np.sqrt(self.hidden_dim)
        scores = Q @ K.T / scale
        
        # Apply exponential decay
        n = scores.shape[1]
        decay_weights = np.power(self.decay, np.arange(n))
        scores = scores * decay_weights[np.newaxis, :]
        
        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-8)
        
        # Context
        context = attn_weights @ V
        
        # Action gating
        if action_context is not None:
            gate = 1 / (1 + np.exp(-(context @ self.W_gate * action_context[:, np.newaxis])))
            context = context * gate
        
        return context
    
    def forward(self, x, action_context=None):
        # Project to Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Attention
        attn_output = self.attention_with_decay(Q, K, V, action_context)
        
        # Output
        return attn_output @ self.W_out + self.b_out
    
    def train(self, X, y, lr=0.001, epochs=100):
        for epoch in range(epochs):
            # Use last few actions as context
            action_context = y[:, 0] if len(y) > 0 else np.zeros(len(y))
            pred = self.forward(X, action_context)
            error = y - pred
            
            # Simplified update
            grad = error.T @ np.ones((X.shape[0], self.hidden_dim)) / len(X)
            self.W_out += lr * grad.T
            self.b_out += lr * error.mean(axis=0)
    
    def predict(self, X):
        return self.forward(X)

def run_experiment():
    """Run H1.143 experiment"""
    print("=" * 60)
    print("H1.143: Action-Gated Attention with Decay on Complex Tasks")
    print("=" * 60)
    
    results = {
        'experiment': 'H1.143',
        'hypothesis': 'Action-gated + decay attention on complex multi-step tasks',
        'timestamp': datetime.now().isoformat(),
        'task_lengths': [],
        'baseline_mses': [],
        'attention_mses': [],
        'improvements': []
    }
    
    # Test different task complexities
    test_configs = [
        (15, 'medium'),
        (25, 'complex'),
        (35, 'very_complex'),
        (45, 'ultra_complex'),
        (60, 'extreme')
    ]
    
    for n_steps, complexity in test_configs:
        print(f"\n--- Testing {complexity} ({n_steps} steps) ---")
        
        # Generate training data
        np.random.seed(42 + n_steps)
        train_trajectories = [generate_complex_trajectory(n_steps + 20) for _ in range(200)]
        X_train, y_train = create_state_action_pairs(train_trajectories, n_steps)
        
        # Normalize
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - X_mean) / X_std
        y_train_norm = (y_train - y_mean) / y_std
        
        # Generate test data
        np.random.seed(123 + n_steps)
        test_trajectories = [generate_complex_trajectory(n_steps + 20) for _ in range(50)]
        X_test, y_test = create_state_action_pairs(test_trajectories, n_steps)
        X_test_norm = (X_test - X_mean) / X_std
        y_test_norm = (y_test - y_mean) / y_std
        
        # Train baseline
        baseline = BaselineModel(X_train.shape[1], y_train.shape[1])
        baseline.train(X_train_norm, y_train_norm, lr=0.01, epochs=50)
        baseline_pred = baseline.predict(X_test_norm)
        baseline_mse = np.mean((baseline_pred * y_std - y_test) ** 2)
        
        # Train attention model
        attention = AttentionModel(X_train.shape[1], y_train.shape[1], 
                                   hidden_dim=128, decay=0.7)
        attention.train(X_train_norm, y_train_norm, lr=0.01, epochs=50)
        attention_pred = attention.predict(X_test_norm)
        attention_mse = np.mean((attention_pred * y_std - y_test) ** 2)
        
        # Calculate improvement
        improvement = (baseline_mse - attention_mse) / baseline_mse * 100
        
        print(f"  Baseline MSE: {baseline_mse:.6f}")
        print(f"  Attention MSE: {attention_mse:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results['task_lengths'].append(n_steps)
        results['baseline_mses'].append(float(baseline_mse))
        results['attention_mses'].append(float(attention_mse))
        results['improvements'].append(float(improvement))
    
    # Summary
    avg_improvement = np.mean(results['improvements'])
    print("\n" + "=" * 60)
    print(f"AVERAGE IMPROVEMENT: {avg_improvement:+.1f}%")
    print("=" * 60)
    
    results['average_improvement'] = float(avg_improvement)
    
    # Determine status
    if avg_improvement > 30:
        results['status'] = 'SUPPORTED'
        results['conclusion'] = f'Action-gated + decay attention achieves {avg_improvement:.1f}% improvement'
    elif avg_improvement > 0:
        results['status'] = 'PARTIAL'
        results['conclusion'] = f'Marginal {avg_improvement:.1f}% improvement - attention helps on complex tasks'
    else:
        results['status'] = 'REFUTED'
        results['conclusion'] = f'Attention does not help on complex tasks ({avg_improvement:.1f}%)'
    
    # Save results
    results_dir = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation'
    with open(f'{results_dir}/experiments/H1.143-action-gated-decay-complex/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/H1.143-action-gated-decay-complex/results.json")
    print(f"Status: {results['status']}")
    
    return results

if __name__ == '__main__':
    results = run_experiment()