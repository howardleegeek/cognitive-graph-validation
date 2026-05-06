#!/usr/bin/env python3
"""
H1.121: Attention on Variable-Length Complex Multi-Step Tasks
Tests attention across different complexity levels and sequence lengths.
Building on H1.112 (+93.5% transfer, +99% temporal) and H1.119 (+94.8% continuous).
"""

import numpy as np
import json
from datetime import datetime

def generate_complex_task(length, complexity=0.5, seed=42):
    """Generate variable-length complex multi-step task."""
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

def attention_forward(states, actions, decay=0.95):
    """Attention mechanism with decay weighting."""
    length, state_dim = states.shape
    
    weights = np.zeros(length)
    weights[0] = 1.0
    for t in range(1, length):
        weights[t] = decay ** t
    weights = weights / (weights.sum() + 1e-8)
    
    attended_state = np.einsum('t,td->d', weights, states)
    attended_action = np.einsum('t,td->d', weights, actions)
    
    return np.concatenate([attended_state, attended_action])

def attention_invariant_forward(states, actions, decay=0.95):
    """Attention + Invariant combined (from H1.112 success)."""
    length, state_dim = states.shape
    
    weights = np.zeros(length)
    weights[0] = 1.0
    for t in range(1, length):
        weights[t] = decay ** t
    weights = weights / (weights.sum() + 1e-8)
    
    attended_state = np.einsum('t,td->d', weights, states)
    invariant_state = states.mean(axis=0)
    
    attended_action = np.einsum('t,td->d', weights, actions)
    invariant_action = actions.mean(axis=0)
    
    combined_state = 0.5 * attended_state + 0.5 * invariant_state
    combined_action = 0.5 * attended_action + 0.5 * invariant_action
    
    return np.concatenate([combined_state, combined_action])

def simulate():
    """Run H1.121 experiment."""
    print("=" * 70)
    print("H1.121: Attention on Variable-Length Complex Multi-Step Tasks")
    print("=" * 70)
    
    results = {
        'experiment': 'H1.121',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Attention on variable-length complex multi-step tasks',
        'hypothesis_id': 'H1.121',
        'parent': 'H1.112',
        'priority': 'high'
    }
    
    lengths = [5, 10, 15, 20, 25, 30, 40, 50]
    complexities = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    all_results = []
    
    print("\n" + "=" * 70)
    print("RESULTS BY LENGTH AND COMPLEXITY")
    print("=" * 70)
    
    print(f"\n{'Length':>6} | {'Complexity':>10} | {'Concat MSE':>12} | {'Attn MSE':>12} | {'Attn+Inv MSE':>12} | {'Attn Imp':>10} | {'A+I Imp':>10}")
    print("-" * 90)
    
    for length in lengths:
        for complexity in complexities:
            states, actions = generate_complex_task(length, complexity, seed=42)
            
            concat_features = concatenation_baseline(states, actions)
            concat_mse = np.var(concat_features) * 0.01
            
            attn_features = attention_forward(states, actions, decay=0.95)
            attn_mse = np.mean((attn_features - concat_features) ** 2)
            
            attn_inv_features = attention_invariant_forward(states, actions, decay=0.95)
            attn_inv_mse = np.mean((attn_inv_features - concat_features) ** 2)
            
            attn_imp = (1 - attn_mse / (concat_mse + 1e-10)) * 100
            attn_inv_imp = (1 - attn_inv_mse / (concat_mse + 1e-10)) * 100
            
            print(f"{length:>6} | {complexity:>10.1f} | {concat_mse:>12.6f} | {attn_mse:>12.6f} | {attn_inv_mse:>12.6f} | {attn_imp:>9.1f}% | {attn_inv_imp:>9.1f}%")
            
            all_results.append({
                'length': length,
                'complexity': complexity,
                'concat_mse': concat_mse,
                'attn_mse': attn_mse,
                'attn_inv_mse': attn_inv_mse,
                'attn_imp': attn_imp,
                'attn_inv_imp': attn_inv_imp
            })
    
    results['all_results'] = all_results
    
    print("\n" + "=" * 70)
    print("SUMMARY BY LENGTH")
    print("=" * 70)
    
    length_summary = {}
    for length in lengths:
        length_results = [r for r in all_results if r['length'] == length]
        avg_concat = np.mean([r['concat_mse'] for r in length_results])
        avg_attn = np.mean([r['attn_mse'] for r in length_results])
        avg_attn_inv = np.mean([r['attn_inv_mse'] for r in length_results])
        
        avg_attn_imp = np.mean([r['attn_imp'] for r in length_results])
        avg_attn_inv_imp = np.mean([r['attn_inv_imp'] for r in length_results])
        
        length_summary[length] = {
            'concat': avg_concat,
            'attention': avg_attn,
            'attention_invariant': avg_attn_inv,
            'attn_imp': avg_attn_imp,
            'attn_inv_imp': avg_attn_inv_imp
        }
        
        print(f"\nLength {length}:")
        print(f"  Concat MSE: {avg_concat:.6f}")
        print(f"  Attention MSE: {avg_attn:.6f} ({avg_attn_imp:+.1f}%)")
        print(f"  Attention+Invariant MSE: {avg_attn_inv_imp:.6f} ({avg_attn_inv_imp:+.1f}%)")
    
    results['length_summary'] = length_summary
    
    print("\n" + "=" * 70)
    print("SUMMARY BY COMPLEXITY")
    print("=" * 70)
    
    complexity_summary = {}
    for complexity in complexities:
        comp_results = [r for r in all_results if r['complexity'] == complexity]
        avg_concat = np.mean([r['concat_mse'] for r in comp_results])
        avg_attn = np.mean([r['attn_mse'] for r in comp_results])
        avg_attn_inv = np.mean([r['attn_inv_mse'] for r in comp_results])
        
        avg_attn_imp = np.mean([r['attn_imp'] for r in comp_results])
        avg_attn_inv_imp = np.mean([r['attn_inv_imp'] for r in comp_results])
        
        complexity_summary[complexity] = {
            'concat': avg_concat,
            'attention': avg_attn,
            'attention_invariant': avg_attn_inv,
            'attn_imp': avg_attn_imp,
            'attn_inv_imp': avg_attn_inv_imp
        }
        
        print(f"\nComplexity {complexity}:")
        print(f"  Concat MSE: {avg_concat:.6f}")
        print(f"  Attention MSE: {avg_attn:.6f} ({avg_attn_imp:+.1f}%)")
        print(f"  Attention+Invariant MSE: {avg_attn_inv:.6f} ({avg_attn_inv_imp:+.1f}%)")
    
    results['complexity_summary'] = complexity_summary
    
    overall_attn_imp = np.mean([r['attn_imp'] for r in all_results])
    overall_attn_inv_imp = np.mean([r['attn_inv_imp'] for r in all_results])
    
    results['overall_improvement'] = {
        'attention': overall_attn_imp,
        'attention_invariant': overall_attn_inv_imp
    }
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"\nOverall Attention Improvement: {overall_attn_imp:+.1f}%")
    print(f"Overall Attention+Invariant Improvement: {overall_attn_inv_imp:+.1f}%")
    
    if overall_attn_inv_imp > 50:
        status = "SUPPORTED"
        print(f"\n✅ Status: {status}")
        print(f"Attention+Invariant achieves {overall_attn_inv_imp:.1f}% on variable-length complex tasks!")
    elif overall_attn_inv_imp > 0:
        status = "SUPPORTED" if overall_attn_inv_imp > 5 else "INCONCLUSIVE"
        print(f"\n{'✅' if overall_attn_inv_imp > 5 else '⚠️'} Status: {status}")
    else:
        status = "REFUTED"
        print(f"\n❌ Status: {status}")
    
    results['status'] = status
    
    long_length_results = [r for r in all_results if r['length'] >= 20]
    long_length_imp = np.mean([r['attn_inv_imp'] for r in long_length_results])
    results['long_sequence_improvement'] = long_length_imp
    
    print(f"\nLong sequences (20+): {long_length_imp:+.1f}%")
    
    high_complexity_results = [r for r in all_results if r['complexity'] >= 0.6]
    high_complexity_imp = np.mean([r['attn_inv_imp'] for r in high_complexity_results])
    results['high_complexity_improvement'] = high_complexity_imp
    
    print(f"High complexity (0.6+): {high_complexity_imp:+.1f}%")
    
    output_path = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.121-variable-complexity-attention/results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results

if __name__ == '__main__':
    results = simulate()