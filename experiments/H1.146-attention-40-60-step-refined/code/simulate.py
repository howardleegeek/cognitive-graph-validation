#!/usr/bin/env python3
"""
H1.146: Refined Attention on 40-60 Step Tasks

Previous H1.137 was inconclusive on 40-60 step sequences.
This experiment tests a simpler, more robust attention mechanism.
"""

import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

def generate_complex_task(num_steps):
    """Generate complex multi-step task with 40-60 steps."""
    num_objects = random.randint(3, 5)
    task_type = random.choice(['manipulation', 'assembly'])
    
    states = []
    actions = []
    for t in range(num_steps):
        state = {
            'positions': [np.random.randn(2).tolist() for _ in range(num_objects)],
            'velocities': [np.random.randn(2).tolist() for _ in range(num_objects)],
            'task_type': task_type,
            'step': t
        }
        action = np.random.randn(4).tolist()
        states.append(state)
        actions.append(action)
    
    return states, actions

def baseline_model(states, actions):
    """Baseline: simple average."""
    total_loss = 0.0
    for t in range(len(actions)):
        pred = sum(actions[t]) / len(actions[t])
        actual = pred + np.random.randn() * 0.01
        total_loss += (pred - actual) ** 2
    return total_loss / len(actions)

def simple_attention_model(states, actions):
    """Simple attention: weighted average with exponential decay."""
    num_steps = len(states)
    decay = 0.8  # Simpler decay
    
    # Compute attention weights
    weights = []
    for t in range(num_steps):
        w = decay ** (num_steps - 1 - t)
        weights.append(w)
    
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    # Compute prediction
    total_loss = 0.0
    for t in range(len(actions)):
        pred = sum(w * sum(actions[i]) / len(actions[i]) for i, w in enumerate(weights[:t+1]))
        actual = sum(actions[t]) / len(actions[t])
        total_loss += (pred - actual) ** 2
    
    return total_loss / len(actions)

def run_experiment():
    """Run the experiment."""
    task_lengths = [40, 45, 50, 55, 60]
    
    results = {
        'experiment': 'H1.146',
        'hypothesis': 'Simple attention on 40-60 step tasks',
        'timestamp': np.datetime64('now').astype(str),
        'task_lengths': task_lengths,
        'baseline_mses': [],
        'attention_mses': [],
        'improvements': []
    }
    
    for num_steps in task_lengths:
        baseline_losses = []
        attention_losses = []
        
        for trial in range(30):
            states, actions = generate_complex_task(num_steps)
            
            baseline_loss = baseline_model(states, actions)
            attention_loss = simple_attention_model(states, actions)
            
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
        
        print(f"Steps: {num_steps}, Baseline: {baseline_mse:.6f}, Attention: {attention_mse:.6f}, Improvement: {improvement:.1f}%")
    
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