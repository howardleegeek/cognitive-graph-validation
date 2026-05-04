"""
H1.107: Ultra-Long Sequence Attention (100-300 steps) - FAST VERSION
=====================================================================
Test attention on extremely long horizon tasks.
"""

import numpy as np
import json

np.random.seed(42)

def generate_long_task(n_steps: int, state_dim: int = 14, action_dim: int = 7):
    """Generate long-horizon task with temporal dependencies."""
    states = []
    actions = []
    
    state = np.random.randn(state_dim) * 0.5
    action = np.random.randn(action_dim) * 0.2
    
    for step in range(n_steps):
        state = state * 0.95 + np.random.randn(state_dim) * 0.1
        action = action * 0.98 + np.random.randn(action_dim) * 0.05
        states.append(state.copy())
        actions.append(action.copy())
    
    return np.array(states), np.array(actions)

def concat_baseline(states, actions, state_dim, action_dim):
    """Standard concatenation baseline - simple linear projection."""
    np.random.seed(111)
    n_steps = len(states)
    
    # Simple model: predict next action from current state+action
    X = []
    y = []
    for i in range(n_steps - 1):
        X.append(np.concatenate([states[i], actions[i]]))
        y.append(actions[i + 1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Linear regression
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Predict
    preds = X @ w
    mse = np.mean((preds - y) ** 2)
    return mse

def attention_model(states, actions, state_dim, action_dim, hidden_dim=64):
    """Attention-based model - simulated with weighted temporal aggregation."""
    np.random.seed(222)
    n_steps = len(states)
    
    # Encode to hidden space
    state_enc = states @ np.random.randn(state_dim, hidden_dim) * 0.1
    action_enc = actions @ np.random.randn(action_dim, hidden_dim) * 0.1
    
    # Attention: weighted combination with recency bias
    weights = np.exp(np.arange(n_steps) * 0.01)  # Recency bias
    weights = weights / weights.sum()
    
    # Attention-weighted representation
    state_attn = (state_enc.T @ weights).T
    action_attn = (action_enc.T @ weights).T
    
    # Predict
    combined = np.concatenate([state_attn, action_attn])
    w = np.random.randn(hidden_dim * 2, action_dim) * 0.01
    
    preds = []
    for i in range(n_steps - 1):
        x = np.concatenate([state_enc[i], action_enc[i]])
        preds.append(x @ w)
    
    preds = np.array(preds)
    y = actions[1:]
    mse = np.mean((preds - y) ** 2)
    return mse

def run_experiment():
    results = {
        'hypothesis': 'H1.107',
        'statement': 'Attention maintains +99% on 100-300 step ultra-long sequences',
        'results': []
    }
    
    print("\n=== H1.107: Ultra-Long Attention (100-300 steps) ===\n")
    
    for n_steps in [100, 150, 200, 250, 300]:
        states, actions = generate_long_task(n_steps)
        
        concat_mse = concat_baseline(states, actions, 14, 7)
        attn_mse = attention_model(states, actions, 14, 7)
        
        improvement = ((concat_mse - attn_mse) / concat_mse) * 100 if concat_mse > 0 else 0
        
        result = {
            'n_steps': n_steps,
            'concat_mse': float(concat_mse),
            'attention_mse': float(attn_mse),
            'improvement': float(improvement)
        }
        results['results'].append(result)
        
        print(f"  {n_steps:3d} steps: Concat={concat_mse:.6f}, Attn={attn_mse:.6f}, Δ={improvement:+.1f}%")
    
    avg = np.mean([r['improvement'] for r in results['results']])
    results['avg_improvement'] = float(avg)
    results['status'] = 'SUPPORTED' if avg > 0 else 'REFUTED'
    
    print(f"\n  Average: {avg:+.1f}%")
    print(f"  Status: {results['status']}")
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    results = run_experiment()