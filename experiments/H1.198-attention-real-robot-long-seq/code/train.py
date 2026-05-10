"""
H1.198: Attention on Real Robot Long Sequences (50-100 steps)

Based on findings:
- H1.180: Real robot vs synthetic gap - autocorrelation is key (+20% gap)
- H1.181: Autocorrelation injection unlocks attention on synthetic data
- H1.196: Attention fails on synthetic 20-40 step sequences
- H1.197: SSM+Attention fails on synthetic 30-60 step

Hypothesis: Attention with high autocorrelation (0.85+) will outperform concatenation on 50-100 step sequences
"""

import numpy as np
import json
from typing import Dict, List, Tuple

np.random.seed(42)

def generate_robot_temporal_data(n_samples: int, n_timesteps: int, autocorrelation: float = 0.85) -> Tuple:
    """Generate robot-like temporal data with autocorrelation."""
    state_dim = 16
    action_dim = 4
    
    all_states = []
    all_actions = []
    all_next_states = []
    
    for _ in range(n_samples):
        state = np.random.randn(state_dim) * 0.1
        state_sequence = []
        action_sequence = []
        next_state_sequence = []
        
        for t in range(n_timesteps):
            action = np.random.randn(action_dim) * 0.1
            
            # Robot-like dynamics with autocorrelation
            if t > 0:
                state = autocorrelation * state + (1 - autocorrelation) * np.random.randn(state_dim) * 0.1
            
            # State transition with action influence
            action_projected = np.zeros(state_dim)
            action_projected[:action_dim] = action
            next_state = state + 0.1 * action_projected + np.random.randn(state_dim) * 0.01
            
            state_sequence.append(state.copy())
            action_sequence.append(action.copy())
            next_state_sequence.append(next_state.copy())
            
            state = next_state
        
        all_states.append(np.array(state_sequence))
        all_actions.append(np.array(action_sequence))
        all_next_states.append(np.array(next_state_sequence))
    
    return np.array(all_states), np.array(all_actions), np.array(all_next_states)


def train_linear_model_multioutput(X, y, lr=0.01, epochs=100):
    """Train simple linear model with gradient descent for multi-output."""
    n_samples, n_features = X.shape
    n_outputs = y.shape[1]
    weights = np.random.randn(n_features, n_outputs) * 0.01
    bias = np.zeros(n_outputs)
    
    for _ in range(epochs):
        pred = X @ weights + bias
        loss = np.mean((pred - y) ** 2)
        grad = 2 * (pred - y) / n_outputs
        weights -= lr * (X.T @ grad) / n_samples
        bias -= lr * np.mean(grad, axis=0)
    
    return weights, bias


def evaluate_concatenation(train_states, train_actions, train_next, val_states, val_actions, val_next):
    """Evaluate concatenation baseline."""
    # Use last timestep for prediction
    train_X = np.concatenate([train_states[:, -1, :], train_actions[:, -1, :]], axis=-1)
    train_y = train_next[:, -1, :]
    
    val_X = np.concatenate([val_states[:, -1, :], val_actions[:, -1, :]], axis=-1)
    val_y = val_next[:, -1, :]
    
    # Train model
    weights, bias = train_linear_model_multioutput(train_X, train_y, lr=0.01, epochs=100)
    
    # Predict
    pred = val_X @ weights + bias
    mse = np.mean((pred - val_y) ** 2)
    
    return mse


def evaluate_attention(train_states, train_actions, train_next, val_states, val_actions, val_next):
    """Evaluate attention model using weighted average of history."""
    n_train = len(train_states)
    n_val = len(val_states)
    seq_len = train_states.shape[1]
    
    # Compute attention weights based on similarity
    all_mses = []
    
    for i in range(n_val):
        # Compute attention weights for validation sample
        val_sa = np.concatenate([val_states[i:i+1], val_actions[i:i+1]], axis=-1)  # (1, seq, 20)
        
        # Compute self-attention weights (use last dimension for similarity)
        # Compute similarity between each pair of timesteps
        scores = np.zeros((seq_len, seq_len))
        for t1 in range(seq_len):
            for t2 in range(seq_len):
                scores[t1, t2] = np.dot(val_sa[0, t1], val_sa[0, t2])
        
        # Softmax over rows
        weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
        
        # Weighted average of states (each row weights how much to attend to each timestep)
        attended_state = np.zeros(16)
        for t in range(seq_len):
            attended_state += weights[-1, t] * val_states[i, t]
        
        attended_state = attended_state.reshape(1, -1)
        
        # Predict next state
        X = np.concatenate([attended_state, val_actions[i:i+1, -1, :]], axis=-1)
        
        # Train on subset of training samples and predict
        preds = []
        for j in range(min(n_train, 50)):  # Use subset for speed
            train_sa = np.concatenate([train_states[j:j+1], train_actions[j:j+1]], axis=-1)
            
            # Compute similarity
            train_scores = np.zeros((seq_len, seq_len))
            for t1 in range(seq_len):
                for t2 in range(seq_len):
                    train_scores[t1, t2] = np.dot(train_sa[0, t1], train_sa[0, t2])
            
            train_weights = np.exp(train_scores) / np.exp(train_scores).sum(axis=1, keepdims=True)
            
            train_attended = np.zeros(16)
            for t in range(seq_len):
                train_attended += train_weights[-1, t] * train_states[j, t]
            
            train_X = np.concatenate([train_attended.reshape(1, -1), train_actions[j:j+1, -1, :]], axis=-1)
            
            w, b = train_linear_model_multioutput(train_X, train_next[j:j+1, -1, :], lr=0.01, epochs=50)
            pred = (X @ w + b)[0]
            preds.append(pred)
        
        pred = np.mean(preds, axis=0)
        mse = np.mean((pred - val_next[i, -1, :]) ** 2)
        all_mses.append(mse)
    
    return np.mean(all_mses)


def run_experiment():
    """Run H1.198 experiment."""
    results = []
    
    # Test different sequence lengths
    for n_steps in [50, 60, 70, 80, 90, 100]:
        print(f"\n=== Testing {n_steps}-step sequences ===")
        
        # Generate data with high autocorrelation (real robot characteristic)
        states, actions, next_states = generate_robot_temporal_data(
            n_samples=200, 
            n_timesteps=n_steps, 
            autocorrelation=0.85  # High autocorrelation like real robot data
        )
        
        # Split into train/val
        n_train = int(0.8 * len(states))
        train_states, val_states = states[:n_train], states[n_train:]
        train_actions, val_actions = actions[:n_train], actions[n_train:]
        train_next, val_next = next_states[:n_train], next_states[n_train:]
        
        # Evaluate concatenation baseline
        concat_mse = evaluate_concatenation(
            train_states, train_actions, train_next,
            val_states, val_actions, val_next
        )
        
        # Evaluate attention model
        attn_mse = evaluate_attention(
            train_states, train_actions, train_next,
            val_states, val_actions, val_next
        )
        
        delta = (attn_mse - concat_mse) / concat_mse * 100
        winner = "ATTENTION" if delta < 0 else "CONCATENATION"
        
        print(f"  Concat MSE: {concat_mse:.6f}")
        print(f"  Attention MSE: {attn_mse:.6f}")
        print(f"  Delta: {delta:.2f}%")
        print(f"  Winner: {winner}")
        
        results.append({
            "n_steps": n_steps,
            "concat_mse": float(concat_mse),
            "attention_mse": float(attn_mse),
            "delta": float(delta),
            "winner": winner
        })
    
    # Calculate average
    avg_delta = np.mean([r["delta"] for r in results])
    status = "SUPPORTED" if avg_delta < 0 else "REFUTED"
    
    print(f"\n=== Summary ===")
    print(f"Average delta: {avg_delta:.2f}%")
    print(f"Status: {status}")
    
    # Save results
    output = {
        "hypothesis": "H1.198",
        "title": "Attention on Real Robot Long Sequences (50-100 steps)",
        "results": results,
        "avg_delta": float(avg_delta),
        "status": status
    }
    
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output


if __name__ == "__main__":
    run_experiment()