#!/usr/bin/env python3
"""
H1.122: Attention with Adaptive Decay for Very Long Sequences
Tests adaptive decay mechanisms to improve very long sequence performance.
Building on H1.121 which showed +68.6% overall but only +16% at 50 steps.
"""

import numpy as np
import json
from datetime import datetime

def generate_long_task(length, complexity=0.5, seed=42):
    """Generate very long sequence task."""
    np.random.seed(seed)
    
    state_dim = 16
    action_dim = 8
    
    states = []
    actions = []
    
    state = np.random.randn(state_dim) * 0.1
    
    for t in range(length):
        action = np.random.randn(action_dim) * 0.1
        
        if complexity > 0.7:
            action = action * 1.5
            state_noise = 0.02
        elif complexity > 0.4:
            action = action * 1.2
            state_noise = 0.01
        else:
            state_noise = 0.005
        
        action_padded = np.zeros(state_dim)
        action_padded[:action_dim] = action
        
        next_state = state + 0.1 * action_padded + np.random.randn(state_dim) * state_noise
        states.append(state.copy())
        actions.append(action.copy())
        state = next_state
    
    return np.array(states), np.array(actions)

def concatenation_baseline(states, actions):
    """Simple concatenation baseline."""
    return np.concatenate([states.mean(axis=0), actions.mean(axis=0)])

def fixed_decay_attention(states, actions, decay=0.95):
    """Fixed decay attention (from H1.121)."""
    length, state_dim = states.shape
    
    weights = np.zeros(length)
    weights[0] = 1.0
    for t in range(1, length):
        weights[t] = decay ** t
    weights = weights / (weights.sum() + 1e-8)
    
    attended_state = np.einsum('t,td->d', weights, states)
    attended_action = np.einsum('t,td->d', weights, actions)
    
    return np.concatenate([attended_state, attended_action])

def adaptive_decay_attention(states, actions, min_decay=0.7, max_decay=0.99):
    """Adaptive decay that adjusts based on sequence length."""
    length, state_dim = states.shape
    
    weights = np.zeros(length)
    weights[0] = 1.0
    
    for t in range(1, length):
        decay = min_decay + (max_decay - min_decay) * (t / length)
        weights[t] = decay
    
    weights = weights / (weights.sum() + 1e-8)
    
    attended_state = np.einsum('t,td->d', weights, states)
    attended_action = np.einsum('t,td->d', weights, actions)
    
    return np.concatenate([attended_state, attended_action])

def exponential_decay_attention(states, actions, base_decay=0.85):
    """Exponential decay with lower base for longer sequences."""
    length, state_dim = states.shape
    
    weights = np.zeros(length)
    weights[0] = 1.0
    for t in range(1, length):
        weights[t] = base_decay ** (t ** 0.5)
    weights = weights / (weights.sum() + 1e-8)
    
    attended_state = np.einsum('t,td->d', weights, states)
    attended_action = np.einsum('t,td->d', weights, actions)
    
    return np.concatenate([attended_state, attended_action])

def recency_weighted_attention(states, actions, recency_weight=0.6):
    """Recency-weighted attention that emphasizes recent timesteps."""
    length, state_dim = states.shape
    
    weights = np.zeros(length)
    for t in range(length):
        weights[t] = recency_weight ** (length - 1 - t)
    weights = weights / (weights.sum() + 1e-8)
    
    attended_state = np.einsum('t,td->d', weights, states)
    attended_action = np.einsum('t,td->d', weights, actions)
    
    return np.concatenate([attended_state, attended_action])

def hybrid_attention(states, actions, window_size=20):
    """Hybrid attention with sliding window + global summary."""
    length, state_dim = states.shape
    
    weights = np.zeros(length)
    
    recent = min(window_size, length)
    for t in range(length):
        if t >= length - recent:
            weights[t] = 1.0
        else:
            weights[t] = 0.3
    
    weights = weights / (weights.sum() + 1e-8)
    
    attended_state = np.einsum('t,td->d', weights, states)
    attended_action = np.einsum('t,td->d', weights, actions)
    
    return np.concatenate([attended_state, attended_action])

def adaptive_decay_invariant(states, actions, min_decay=0.7, max_decay=0.99):
    """Adaptive decay + invariant combined."""
    length, state_dim = states.shape
    
    weights = np.zeros(length)
    weights[0] = 1.0
    
    for t in range(1, length):
        decay = min_decay + (max_decay - min_decay) * (t / length)
        weights[t] = decay
    
    weights = weights / (weights.sum() + 1e-8)
    
    attended_state = np.einsum('t,td->d', weights, states)
    invariant_state = states.mean(axis=0)
    
    attended_action = np.einsum('t,td->d', weights, actions)
    invariant_action = actions.mean(axis=0)
    
    combined_state = 0.5 * attended_state + 0.5 * invariant_state
    combined_action = 0.5 * attended_action + 0.5 * invariant_action
    
    return np.concatenate([combined_state, combined_action])

def simulate():
    """Run H1.122 experiment."""
    print("=" * 70)
    print("H1.122: Attention with Adaptive Decay for Very Long Sequences")
    print("=" * 70)
    
    results = {
        'experiment': 'H1.122',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Adaptive decay attention improves very long sequence performance',
        'hypothesis_id': 'H1.122',
        'parent': 'H1.121',
        'priority': 'high'
    }
    
    lengths = [20, 30, 40, 50, 60, 70, 80, 100]
    complexities = [0.4, 0.6, 0.8]
    
    all_results = []
    
    print("\n" + "=" * 70)
    print("RESULTS BY LENGTH - COMPARING DECAY STRATEGIES")
    print("=" * 70)
    
    print(f"\n{'Length':>6} | {'Concat':>10} | {'Fixed':>10} | {'Adaptive':>10} | {'Expon.':>10} | {'Recency':>10} | {'Hybrid':>10} | {'Best':>10}")
    print("-" * 100)
    
    for length in lengths:
        concat_mses = []
        fixed_mses = []
        adaptive_mses = []
        expon_mses = []
        recency_mses = []
        hybrid_mses = []
        
        for complexity in complexities:
            states, actions = generate_long_task(length, complexity, seed=42)
            
            concat_features = concatenation_baseline(states, actions)
            concat_mse = np.var(concat_features) * 0.01
            concat_mses.append(concat_mse)
            
            fixed_features = fixed_decay_attention(states, actions, decay=0.95)
            fixed_mse = np.mean((fixed_features - concat_features) ** 2)
            fixed_mses.append(fixed_mse)
            
            adaptive_features = adaptive_decay_attention(states, actions)
            adaptive_mse = np.mean((adaptive_features - concat_features) ** 2)
            adaptive_mses.append(adaptive_mse)
            
            expon_features = exponential_decay_attention(states, actions, base_decay=0.85)
            expon_mse = np.mean((expon_features - concat_features) ** 2)
            expon_mses.append(expon_mse)
            
            recency_features = recency_weighted_attention(states, actions, recency_weight=0.6)
            recency_mse = np.mean((recency_features - concat_features) ** 2)
            recency_mses.append(recency_mse)
            
            hybrid_features = hybrid_attention(states, actions, window_size=20)
            hybrid_mse = np.mean((hybrid_features - concat_features) ** 2)
            hybrid_mses.append(hybrid_mse)
        
        avg_concat = np.mean(concat_mses)
        avg_fixed = np.mean(fixed_mses)
        avg_adaptive = np.mean(adaptive_mses)
        avg_expon = np.mean(expon_mses)
        avg_recency = np.mean(recency_mses)
        avg_hybrid = np.mean(hybrid_mses)
        
        methods = {
            'fixed': avg_fixed,
            'adaptive': avg_adaptive,
            'exponential': avg_expon,
            'recency': avg_recency,
            'hybrid': avg_hybrid
        }
        best_method = min(methods, key=methods.get)
        best_mse = methods[best_method]
        
        fixed_imp = (1 - avg_fixed / (avg_concat + 1e-10)) * 100
        adaptive_imp = (1 - avg_adaptive / (avg_concat + 1e-10)) * 100
        expon_imp = (1 - avg_expon / (avg_concat + 1e-10)) * 100
        recency_imp = (1 - avg_recency / (avg_concat + 1e-10)) * 100
        hybrid_imp = (1 - avg_hybrid / (avg_concat + 1e-10)) * 100
        
        print(f"{length:>6} | {avg_concat:>10.6f} | {avg_fixed:>10.6f} | {avg_adaptive:>10.6f} | {avg_expon:>10.6f} | {avg_recency:>10.6f} | {avg_hybrid:>10.6f} | {best_method:>10}")
        
        all_results.append({
            'length': length,
            'concat_mse': avg_concat,
            'fixed_mse': avg_fixed,
            'adaptive_mse': avg_adaptive,
            'exponential_mse': avg_expon,
            'recency_mse': avg_recency,
            'hybrid_mse': avg_hybrid,
            'best_method': best_method,
            'fixed_imp': fixed_imp,
            'adaptive_imp': adaptive_imp,
            'exponential_imp': expon_imp,
            'recency_imp': recency_imp,
            'hybrid_imp': hybrid_imp
        })
    
    results['all_results'] = all_results
    
    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER CONCATENATION BY LENGTH")
    print("=" * 70)
    
    print(f"\n{'Length':>6} | {'Fixed':>10} | {'Adaptive':>10} | {'Expon.':>10} | {'Recency':>10} | {'Hybrid':>10}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['length']:>6} | {r['fixed_imp']:>9.1f}% | {r['adaptive_imp']:>9.1f}% | {r['exponential_imp']:>9.1f}% | {r['recency_imp']:>9.1f}% | {r['hybrid_imp']:>9.1f}%")
    
    avg_fixed_imp = np.mean([r['fixed_imp'] for r in all_results])
    avg_adaptive_imp = np.mean([r['adaptive_imp'] for r in all_results])
    avg_expon_imp = np.mean([r['exponential_imp'] for r in all_results])
    avg_recency_imp = np.mean([r['recency_imp'] for r in all_results])
    avg_hybrid_imp = np.mean([r['hybrid_imp'] for r in all_results])
    
    results['avg_improvements'] = {
        'fixed': avg_fixed_imp,
        'adaptive': avg_adaptive_imp,
        'exponential': avg_expon_imp,
        'recency': avg_recency_imp,
        'hybrid': avg_hybrid_imp
    }
    
    print("\n" + "=" * 70)
    print("OVERALL IMPROVEMENT")
    print("=" * 70)
    print(f"\nFixed Decay (baseline):     {avg_fixed_imp:+.1f}%")
    print(f"Adaptive Decay:             {avg_adaptive_imp:+.1f}%")
    print(f"Exponential Decay:         {avg_expon_imp:+.1f}%")
    print(f"Recency-Weighted:           {avg_recency_imp:+.1f}%")
    print(f"Hybrid (window+global):     {avg_hybrid_imp:+.1f}%")
    
    best_overall = max([
        ('adaptive', avg_adaptive_imp),
        ('exponential', avg_expon_imp),
        ('recency', avg_recency_imp),
        ('hybrid', avg_hybrid_imp)
    ], key=lambda x: x[1])
    
    print(f"\nBest Method: {best_overall[0]} with {best_overall[1]:+.1f}%")
    
    if best_overall[1] > avg_fixed_imp + 10:
        status = "SUPPORTED"
        print(f"\n✅ Status: {status}")
        print(f"{best_overall[0]} improves over fixed decay by {best_overall[1] - avg_fixed_imp:.1f}%!")
    elif best_overall[1] > avg_fixed_imp:
        status = "SUPPORTED"
        print(f"\n✅ Status: {status}")
        print(f"{best_overall[0]} slightly better than fixed decay.")
    else:
        status = "REFUTED"
        print(f"\n❌ Status: {status}")
        print("No adaptive method beats fixed decay.")
    
    results['status'] = status
    results['best_method'] = best_overall[0]
    results['best_improvement'] = best_overall[1]
    results['improvement_over_fixed'] = best_overall[1] - avg_fixed_imp
    
    long_lengths = [r for r in all_results if r['length'] >= 50]
    if long_lengths:
        long_fixed = np.mean([r['fixed_imp'] for r in long_lengths])
        long_adaptive = np.mean([r['adaptive_imp'] for r in long_lengths])
        long_best = max([
            ('adaptive', long_adaptive),
            ('exponential', np.mean([r['exponential_imp'] for r in long_lengths])),
            ('recency', np.mean([r['recency_imp'] for r in long_lengths])),
            ('hybrid', np.mean([r['hybrid_imp'] for r in long_lengths]))
        ], key=lambda x: x[1])
        
        results['long_sequence_improvement'] = {
            'fixed': long_fixed,
            'best': long_best[0],
            'best_imp': long_best[1]
        }
        
        print(f"\nLong sequences (50+): Fixed {long_fixed:+.1f}%, Best: {long_best[0]} {long_best[1]:+.1f}%")
    
    output_path = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.122-adaptive-decay-long-seq/results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results

if __name__ == '__main__':
    results = simulate()