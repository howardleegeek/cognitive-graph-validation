"""
H1.196: Attention on 20-40 Step Sequences (Next-Step Prediction)

Based on findings:
- H3.4: Attention marginally helps on 24, 30 step sequences (-0.4% avg)
- H3.6: Linear attention +100% on 40+ step sequences
- H1.193: SSM +97.6% on 50-step with next-step prediction
- H1.195: Baseline wins on final-step prediction

Hypothesis: Attention with next-step prediction will outperform concatenation on 20-40 step sequences
"""

import numpy as np
import json
from typing import Dict, List, Tuple

def generate_robot_temporal_data(n_samples: int, n_timesteps: int, autocorrelation: float = 0.85, train_split: float = 0.8) -> Tuple:
    """Generate robot-like temporal data with autocorrelation (real robot characteristic)."""
    np.random.seed(42)
    
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
            
            state_sequence.append(state)
            action_sequence.append(action)
            next_state_sequence.append(next_state)
            
            state = next_state
        
        all_states.append(np.array(state_sequence))
        all_actions.append(np.array(action_sequence))
        all_next_states.append(np.array(next_state_sequence))
    
    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    all_next_states = np.array(all_next_states)
    
    # Split into train and test
    n_train = int(n_samples * train_split)
    return (
        all_states[:n_train], all_actions[:n_train], all_next_states[:n_train],
        all_states[n_train:], all_actions[n_train:], all_next_states[n_train:]
    )

def train_and_evaluate_concatenation(states_train, actions_train, next_states_train, states_test, actions_test, next_states_test):
    """Train and evaluate concatenation model."""
    n_samples, n_timesteps, state_dim = states_train.shape
    _, _, action_dim = actions_train.shape
    
    # Concatenate all timesteps
    X_train = np.concatenate([states_train.reshape(n_samples, -1), actions_train.reshape(n_samples, -1)], axis=1)
    y_train = next_states_train.reshape(n_samples, -1)
    
    # Add some noise to prevent perfect fit
    X_train = X_train + np.random.randn(*X_train.shape) * 0.001
    
    # Simple linear regression with regularization
    X_with_bias = np.concatenate([X_train, np.ones((n_samples, 1))], axis=1)
    lambda_reg = 0.01
    XTX = X_with_bias.T @ X_with_bias + lambda_reg * np.eye(X_with_bias.shape[1])
    XTy = X_with_bias.T @ y_train
    weights = np.linalg.solve(XTX, XTy)
    
    # Evaluate on test set
    n_test = states_test.shape[0]
    X_test = np.concatenate([states_test.reshape(n_test, -1), actions_test.reshape(n_test, -1)], axis=1)
    X_test_with_bias = np.concatenate([X_test, np.ones((n_test, 1))], axis=1)
    predictions = X_test_with_bias @ weights
    mse = np.mean((predictions - next_states_test.reshape(n_test, -1)) ** 2)
    
    return mse

def train_and_evaluate_attention(states_train, actions_train, next_states_train, states_test, actions_test, next_states_test):
    """Train and evaluate attention model."""
    n_samples, n_timesteps, state_dim = states_train.shape
    _, _, action_dim = actions_train.shape
    
    # For attention, we use a learned attention mechanism
    # Train a simple model that uses attention-weighted previous states
    
    # First, compute attention weights for each training sample
    all_attention_features = []
    all_targets = []
    
    for i in range(n_samples):
        for t in range(n_timesteps):
            if t == 0:
                # First timestep: use initial state
                context = states_train[i, 0]
            else:
                # Compute attention over previous timesteps
                query = states_train[i, t]
                keys = states_train[i, :t]
                
                # Attention scores
                scores = np.sum(query * keys, axis=1)
                attention = np.exp(scores) / (np.exp(scores).sum() + 1e-8)
                
                # Weighted context
                context = np.sum(attention[:, None] * keys, axis=0)
            
            # Combine context with current action
            action_projected = np.zeros(state_dim)
            action_projected[:action_dim] = actions_train[i, t]
            
            features = np.concatenate([context, action_projected])
            all_attention_features.append(features)
            all_targets.append(next_states_train[i, t])
    
    all_attention_features = np.array(all_attention_features)
    all_targets = np.array(all_targets)
    
    # Add noise to prevent perfect fit
    all_attention_features = all_attention_features + np.random.randn(*all_attention_features.shape) * 0.001
    
    # Train linear model
    X_with_bias = np.concatenate([all_attention_features, np.ones((len(all_attention_features), 1))], axis=1)
    lambda_reg = 0.01
    XTX = X_with_bias.T @ X_with_bias + lambda_reg * np.eye(X_with_bias.shape[1])
    XTy = X_with_bias.T @ all_targets
    weights = np.linalg.solve(XTX, XTy)
    
    # Evaluate on test set
    n_test = states_test.shape[0]
    total_mse = 0
    count = 0
    
    for i in range(n_test):
        for t in range(n_timesteps):
            if t == 0:
                context = states_test[i, 0]
            else:
                query = states_test[i, t]
                keys = states_test[i, :t]
                scores = np.sum(query * keys, axis=1)
                attention = np.exp(scores) / (np.exp(scores).sum() + 1e-8)
                context = np.sum(attention[:, None] * keys, axis=0)
            
            action_projected = np.zeros(state_dim)
            action_projected[:action_dim] = actions_test[i, t]
            
            features = np.concatenate([context, action_projected])
            features_with_bias = np.concatenate([features, [1]])
            
            pred = features_with_bias @ weights
            mse = np.mean((pred - next_states_test[i, t]) ** 2)
            total_mse += mse
            count += 1
    
    return total_mse / count

def run_experiment():
    """Run H1.196 experiment."""
    results = {
        "hypothesis": "H1.196",
        "title": "Attention on 20-40 Step Sequences (Next-Step Prediction)",
        "results": []
    }
    
    n_samples = 200
    
    for n_steps in [20, 25, 30, 35, 40]:
        print(f"\nTesting {n_steps}-step sequences...")
        
        # Generate data with autocorrelation (real robot characteristic)
        states_train, actions_train, next_states_train, states_test, actions_test, next_states_test = \
            generate_robot_temporal_data(n_samples, n_steps, autocorrelation=0.85)
        
        # Train and evaluate models
        concat_mse = train_and_evaluate_concatenation(
            states_train, actions_train, next_states_train,
            states_test, actions_test, next_states_test
        )
        attention_mse = train_and_evaluate_attention(
            states_train, actions_train, next_states_train,
            states_test, actions_test, next_states_test
        )
        
        delta = ((concat_mse - attention_mse) / concat_mse) * 100
        
        result = {
            "n_steps": n_steps,
            "concat_mse": float(concat_mse),
            "attention_mse": float(attention_mse),
            "delta": float(delta),
            "winner": "ATTENTION" if delta > 0 else "CONCATENATION"
        }
        
        results["results"].append(result)
        
        print(f"  Concat MSE: {concat_mse:.6f}")
        print(f"  Attention MSE: {attention_mse:.6f}")
        print(f"  Delta: {delta:.2f}%")
    
    # Calculate average
    avg_delta = np.mean([r["delta"] for r in results["results"]])
    results["avg_delta"] = float(avg_delta)
    
    # Determine status
    if avg_delta > 5:
        results["status"] = "SUPPORTED"
    elif avg_delta < -5:
        results["status"] = "REFUTED"
    else:
        results["status"] = "INCONCLUSIVE"
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== H1.196 Results ===")
    print(f"Average Delta: {avg_delta:.2f}%")
    print(f"Status: {results['status']}")
    
    return results

if __name__ == "__main__":
    run_experiment()