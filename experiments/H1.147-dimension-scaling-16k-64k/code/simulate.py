#!/usr/bin/env python3
"""
H1.147: Dimension Scaling 16k-64k with Attention

Previous H1.20 showed 32k optimal with α=0.3.
This tests larger dimensions (16k-64k) with attention mechanism.
"""

import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

def generate_task(num_steps, dim):
    """Generate task with specified dimension."""
    num_objects = 3
    
    states = []
    actions = []
    for t in range(num_steps):
        # State dimension matches model dimension
        state = np.random.randn(dim).tolist()
        action = np.random.randn(4).tolist()
        states.append(state)
        actions.append(action)
    
    return states, actions

def baseline_model(states, actions, dim):
    """Baseline: simple linear projection."""
    W = np.random.randn(dim, 4) * 0.01
    b = np.random.randn(4) * 0.01
    
    total_loss = 0.0
    for t in range(len(states)):
        state = np.array(states[t])
        pred = W.T @ state + b
        actual = np.array(actions[t])
        total_loss += np.sum((pred - actual) ** 2)
    
    return total_loss / len(actions)

def attention_model(states, actions, dim):
    """Attention model with dimension scaling."""
    num_steps = len(states)
    decay = 0.9
    
    # Attention weights
    weights = [decay ** (num_steps - 1 - t) for t in range(num_steps)]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    # Weighted state representation
    weighted_state = np.zeros(dim)
    for t in range(num_steps):
        weighted_state += weights[t] * np.array(states[t])
    
    # Project to action
    W = np.random.randn(dim, 4) * 0.01
    b = np.random.randn(4) * 0.01
    
    total_loss = 0.0
    for t in range(len(actions)):
        pred = W.T @ weighted_state + b
        actual = np.array(actions[t])
        total_loss += np.sum((pred - actual) ** 2)
    
    return total_loss / len(actions)

def run_experiment():
    """Run the experiment."""
    dimensions = [16384, 32768, 65536]
    num_steps = 20
    
    results = {
        'experiment': 'H1.147',
        'hypothesis': 'Dimension scaling 16k-64k with attention',
        'timestamp': np.datetime64('now').astype(str),
        'dimensions': dimensions,
        'baseline_mses': [],
        'attention_mses': [],
        'improvements': []
    }
    
    for dim in dimensions:
        baseline_losses = []
        attention_losses = []
        
        for trial in range(20):
            states, actions = generate_task(num_steps, dim)
            
            baseline_loss = baseline_model(states, actions, dim)
            attention_loss = attention_model(states, actions, dim)
            
            baseline_losses.append(baseline_loss)
            attention_losses.append(attention_loss)
        
        baseline_mse = np.mean(baseline_losses)
        attention_mse = np.mean(attention_losses)
        
        if baseline_mse > 0:
            improvement = ((baseline_mse - attention_mse) / baseline_mse) * 100
        else:
            improvement = 0
        
        results['baseline_mses'].append(baseline_mse)
        results['attention_mses'].append(attention_mse)
        results['improvements'].append(improvement)
        
        print(f"Dim: {dim}, Baseline: {baseline_mse:.4f}, Attention: {attention_mse:.4f}, Improvement: {improvement:.1f}%")
    
    avg_improvement = np.mean(results['improvements'])
    results['average_improvement'] = avg_improvement
    
    if avg_improvement > 5:
        status = 'SUPPORTED'
    elif avg_improvement > 0:
        status = 'PARTIAL'
    else:
        status = 'REFUTED'
    
    results['status'] = status
    results['conclusion'] = f'{status}: {avg_improvement:.1f}% average improvement'
    
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    print(f"Status: {status}")
    
    return results

if __name__ == '__main__':
    results = run_experiment()
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")