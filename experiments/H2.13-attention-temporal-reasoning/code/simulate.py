#!/usr/bin/env python3
"""
H2.13: Attention on Temporal Reasoning Tasks

Based on H2.3-2.5 success (+56-75% on temporal reasoning), test if attention
can also improve temporal reasoning tasks.
"""

import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

def generate_temporal_task(num_steps, num_objects=3):
    """Generate temporal reasoning task with object permanence."""
    # Initial positions
    positions = [np.random.randn(2) for _ in range(num_objects)]
    
    states = []
    actions = []
    
    for t in range(num_steps):
        # Objects move with some continuity (temporal structure)
        for i in range(num_objects):
            positions[i] += np.random.randn(2) * 0.1
        
        state = {
            'positions': [p.tolist() for p in positions],
            'object_ids': list(range(num_objects)),
            'step': t
        }
        
        # Action predicts next position
        action = positions[0].tolist() + [0.5]  # gripper state
        
        states.append(state)
        actions.append(action)
    
    return states, actions

def baseline_model(states, actions):
    """Baseline: linear extrapolation."""
    total_loss = 0.0
    
    for t in range(1, len(actions)):
        # Simple linear prediction
        prev_pos = states[t-1]['positions'][0]
        curr_pos = states[t]['positions'][0]
        
        velocity = [curr_pos[i] - prev_pos[i] for i in range(2)]
        pred = [curr_pos[i] + velocity[i] * 0.5 for i in range(2)]
        
        actual = actions[t][:2]
        total_loss += sum((pred[i] - actual[i]) ** 2 for i in range(2))
    
    return total_loss / max(1, len(actions) - 1)

def attention_model(states, actions):
    """Attention model with temporal structure."""
    num_steps = len(states)
    decay = 0.9  # Higher decay for temporal continuity
    
    # Build temporal representations
    positions = [s['positions'][0] for s in states]
    
    total_loss = 0.0
    for t in range(1, len(actions)):
        # Attention over past positions
        weights = []
        for j in range(t):
            dist = sum((positions[t][i] - positions[j][i]) ** 2 for i in range(2)) ** 0.5
            w = decay ** (t - j) * np.exp(-dist)
            weights.append(w)
        
        total_w = sum(weights) + 1e-8
        weights = [w / total_w for w in weights]
        
        # Weighted prediction
        pred = [sum(w * positions[j][i] for j, w in enumerate(weights)) for i in range(2)]
        actual = actions[t][:2]
        
        total_loss += sum((pred[i] - actual[i]) ** 2 for i in range(2))
    
    return total_loss / max(1, len(actions) - 1)

def run_experiment():
    """Run the experiment."""
    task_lengths = [5, 10, 15, 20, 25]
    
    results = {
        'experiment': 'H2.13',
        'hypothesis': 'Attention on temporal reasoning tasks',
        'timestamp': np.datetime64('now').astype(str),
        'task_lengths': task_lengths,
        'baseline_mses': [],
        'attention_mses': [],
        'improvements': []
    }
    
    for num_steps in task_lengths:
        baseline_losses = []
        attention_losses = []
        
        for trial in range(50):
            states, actions = generate_temporal_task(num_steps)
            
            baseline_loss = baseline_model(states, actions)
            attention_loss = attention_model(states, actions)
            
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