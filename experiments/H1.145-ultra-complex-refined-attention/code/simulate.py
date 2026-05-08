#!/usr/bin/env python3
"""
H1.145: Ultra-Complex Refined Attention on 60-100 Step Multi-Step Tasks

Based on H1 success (+25.6%), test refined attention architecture on ultra-complex tasks.
Previous H1.139 showed mixed results (-38.6% avg) on 60-100 step tasks.
This experiment tests a refined architecture combining:
- Action-gated attention (from H1.39: +30% over standard)
- Query-key decay (from H1.40: +30% improvement)
- Sparse attention patterns (from H1.43: stride pattern best, -2% vs full)
"""

import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

def generate_ultra_complex_task(num_steps):
    """Generate ultra-complex multi-step task with 60-100 steps."""
    num_objects = random.randint(3, 6)
    task_type = random.choice(['manipulation', 'assembly', 'navigation'])
    
    states = []
    actions = []
    for t in range(num_steps):
        state = {
            'positions': [np.random.randn(2).tolist() for _ in range(num_objects)],
            'velocities': [np.random.randn(2).tolist() for _ in range(num_objects)],
            'task_type': task_type,
            'step': t,
            'total_steps': num_steps
        }
        action = np.random.randn(4).tolist()
        states.append(state)
        actions.append(action)
    
    return states, actions

def baseline_model(states, actions):
    """Baseline: concatenation approach."""
    total_loss = 0.0
    for i in range(len(states)):
        state_vec = []
        for obj in states[i]['positions']:
            state_vec.extend(obj)
        for obj in states[i]['velocities']:
            state_vec.extend(obj)
        
        concat = state_vec + actions[i]
        pred = sum(concat) / len(concat)
        actual = sum(actions[i]) / len(actions[i])
        total_loss += (pred - actual) ** 2
    
    return total_loss / len(states)

def refined_attention_model(states, actions):
    """Refined attention: action-gated + query-key decay + sparse stride pattern."""
    num_steps = len(states)
    num_objects = len(states[0]['positions'])
    
    # Build state representations
    state_repr = []
    for s in states:
        repr_vec = []
        for obj in s['positions']:
            repr_vec.extend(obj)
        for obj in s['velocities']:
            repr_vec.extend(obj)
        state_repr.append(repr_vec)
    
    # Action-gated attention with query-key decay
    decay_factor = 0.7  # From H1.40
    attention_weights = []
    
    for t in range(num_steps):
        # Query-key decay: recent timesteps weighted more
        query = state_repr[t]
        
        # Compute attention with decay
        weights = []
        for j in range(t + 1):
            key = state_repr[j]
            # Simple dot product with decay
            similarity = sum(q * k for q, k in zip(query, key[:len(query)]))
            # Apply decay: more recent = higher weight
            decay = decay_factor ** (t - j)
            weights.append(similarity * decay)
        
        # Normalize
        total = sum(abs(w) for w in weights) + 1e-8
        weights = [w / total for w in weights]
        attention_weights.append(weights)
    
    # Sparse attention: stride pattern (from H1.43)
    stride = 2
    predictions = []
    for t in range(num_steps):
        # Use stride pattern for efficiency
        indices = list(range(max(0, t - 5), t + 1, stride))
        if not indices:
            indices = [t]
        
        weighted_sum = 0.0
        for idx in indices:
            w = attention_weights[t][idx] if idx < len(attention_weights[t]) else 0
            weighted_sum += w * sum(state_repr[idx])
        
        # Action-gating: incorporate action information
        if t < len(actions):
            action_gate = np.tanh(sum(actions[t][:2]))
            weighted_sum = weighted_sum * (1 + action_gate * 0.1)
        
        predictions.append(weighted_sum)
    
    # Compute loss
    total_loss = 0.0
    for t in range(len(actions)):
        actual = sum(actions[t]) / len(actions[t])
        pred = predictions[t] if t < len(predictions) else predictions[-1]
        total_loss += (pred - actual) ** 2
    
    return total_loss / len(actions)

def run_experiment():
    """Run the experiment."""
    task_lengths = [60, 70, 80, 90, 100]
    
    results = {
        'experiment': 'H1.145',
        'hypothesis': 'Refined attention on ultra-complex (60-100 step) multi-step tasks',
        'timestamp': np.datetime64('now').astype(str),
        'task_lengths': task_lengths,
        'baseline_mses': [],
        'attention_mses': [],
        'improvements': []
    }
    
    for num_steps in task_lengths:
        # Generate multiple tasks
        baseline_losses = []
        attention_losses = []
        
        for trial in range(20):
            states, actions = generate_ultra_complex_task(num_steps)
            
            baseline_loss = baseline_model(states, actions)
            attention_loss = refined_attention_model(states, actions)
            
            baseline_losses.append(baseline_loss)
            attention_losses.append(attention_loss)
        
        baseline_mse = np.mean(baseline_losses)
        attention_mse = np.mean(attention_losses)
        
        # Calculate improvement
        if baseline_mse > 0:
            improvement = ((baseline_mse - attention_mse) / baseline_mse) * 100
        else:
            improvement = 0
        
        results['baseline_mses'].append(baseline_mse)
        results['attention_mses'].append(attention_mse)
        results['improvements'].append(improvement)
        
        print(f"Steps: {num_steps}, Baseline: {baseline_mse:.4f}, Attention: {attention_mse:.4f}, Improvement: {improvement:.1f}%")
    
    avg_improvement = np.mean(results['improvements'])
    results['average_improvement'] = avg_improvement
    
    # Determine status
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
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")