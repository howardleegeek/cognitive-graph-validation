#!/usr/bin/env python3
"""
H1.180: Real Robot vs Synthetic Data Gap Analysis
Building on:
- H1.171: +18.6% on actual real robot data
- H1.179: -13.4% on synthetic real-robot simulation

Hypothesis: The gap is due to noise characteristics and temporal correlations
that differ between real and synthetic data.
"""

import json
import numpy as np
import random
from datetime import datetime

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def generate_synthetic_data(num_samples=100, seq_len=100, noise_level=0.01):
    """Generate synthetic data with configurable noise."""
    states = []
    for i in range(num_samples):
        seq = []
        for t in range(seq_len):
            # Smooth continuous trajectory
            state = {
                'ee_pos': [0.4 + 0.1 * np.sin(t / 10), 0.2 + 0.1 * t / 100, 0.1 + 0.02 * np.cos(t / 5)],
                'ee_vel': [0.01 * np.cos(t / 10), 0.001, -0.002 * np.sin(t / 5)],
                'gripper': 0.5 + 0.4 * np.sin(t / 25) if 0.3 < t / seq_len < 0.7 else 0.1,
            }
            # Add Gaussian noise
            for k in state:
                if isinstance(state[k], list):
                    state[k] = [v + np.random.randn() * noise_level for v in state[k]]
            seq.append(state)
        states.append(seq)
    return states

def generate_real_robot_like_data(num_samples=100, seq_len=100, autocorr=0.7):
    """Generate data with real robot characteristics: temporal autocorrelation and 1/f noise."""
    states = []
    for i in range(num_samples):
        seq = []
        prev_state = None
        for t in range(seq_len):
            if prev_state is None:
                # Initialize
                base = {
                    'ee_pos': [0.4, 0.2, 0.1],
                    'ee_vel': [0.0, 0.0, 0.0],
                    'gripper': 0.1,
                }
            else:
                # Autocorrelated movement
                base = {
                    'ee_pos': [prev_state['ee_pos'][j] + 0.001 * np.random.randn() + (1-autocorr) * 0.01 * np.sin(t / 10 + j) 
                              for j in range(3)],
                    'ee_vel': [(prev_state['ee_pos'][j] - prev_state.get('prev_pos', prev_state['ee_pos'])[j]) + 0.0001 * np.random.randn()
                              for j in range(3)],
                    'gripper': prev_state['gripper'] + 0.01 * np.random.randn() if np.random.rand() < 0.1 else prev_state['gripper'],
                }
            
            # 1/f noise characteristic
            for k in base:
                if isinstance(base[k], list):
                    base[k] = [v + np.random.randn() * 0.005 for v in base[k]]
            
            base['prev_pos'] = base['ee_pos'].copy()
            seq.append(base)
            prev_state = base
        states.append(seq)
    return states

def simple_attention(seq, embed_dim=7):
    """Simple attention mechanism."""
    embeddings = []
    for s in seq:
        emb = np.array(s['ee_pos'] + [s['gripper']])
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    query = embeddings[-1]
    embed_dim = len(query)
    scores = np.dot(query, embeddings.T) / (np.sqrt(embed_dim) + 1e-8)
    decay = np.exp(-0.01 * np.arange(len(scores)))
    scores = scores * decay[::-1]
    weights = scores / (scores.sum() + 1e-8)
    return np.dot(weights, embeddings)

def concat_baseline(seq, embed_dim=7):
    """Concatenation baseline."""
    embeddings = []
    for s in seq:
        emb = np.array(s['ee_pos'] + [s['gripper']])
        embeddings.append(emb)
    return np.mean(embeddings, axis=0)

def run_experiment():
    set_seed(42)
    
    print("=" * 70)
    print("H1.180: Real vs Synthetic Data Gap Analysis")
    print("=" * 70)
    
    results = {
        'hypothesis': 'H1.180: Real vs Synthetic Gap Analysis',
        'timestamp': datetime.now().isoformat(),
        'parent_hypotheses': ['H1.171', 'H1.179'],
        'data_types': [],
        'noise_levels': [],
        'autocorr_levels': [],
        'concat_mses': [],
        'attn_mses': [],
    }
    
    # Test configurations
    configs = [
        ('low_noise_synthetic', 0.001, 0.0),
        ('mid_noise_synthetic', 0.01, 0.0),
        ('high_noise_synthetic', 0.1, 0.0),
        ('low_autocorr_real', 0.005, 0.3),
        ('mid_autocorr_real', 0.005, 0.7),
        ('high_autocorr_real', 0.005, 0.95),
    ]
    
    for name, noise, autocorr in configs:
        print(f"\n{name} (noise={noise}, autocorr={autocorr}):")
        
        # Generate data
        if autocorr == 0.0:
            data = generate_synthetic_data(50, 100, noise)
        else:
            data = generate_real_robot_like_data(50, 100, autocorr)
        
        # Test on multiple samples
        concat_errors = []
        attn_errors = []
        
        for seq in data[:20]:
            # Generate targets
            actions = [np.array(s['ee_pos'] + [s['gripper']]) for s in seq]
            
            # Concat baseline
            concat_emb = concat_baseline(seq)
            concat_pred = np.tanh(concat_emb)[:len(actions[-1])]
            concat_mse = np.mean((concat_pred - actions[-1]) ** 2)
            concat_errors.append(concat_mse)
            
            # Attention
            attn_emb = simple_attention(seq)
            attn_pred = np.tanh(attn_emb)[:len(actions[-1])]
            attn_mse = np.mean((attn_pred - actions[-1]) ** 2)
            attn_errors.append(attn_mse)
        
        avg_concat = np.mean(concat_errors)
        avg_attn = np.mean(attn_errors)
        improvement = (avg_concat - avg_attn) / (avg_concat + 1e-8) * 100
        
        print(f"  Concat MSE: {avg_concat:.6f}")
        print(f"  Attention MSE: {avg_attn:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results['data_types'].append(name)
        results['noise_levels'].append(noise)
        results['autocorr_levels'].append(autocorr)
        results['concat_mses'].append(avg_concat)
        results['attn_mses'].append(avg_attn)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Group by type
    synthetic = [i for i, n in enumerate(results['data_types']) if 'synthetic' in n]
    real_like = [i for i, n in enumerate(results['data_types']) if 'real' in n]
    
    avg_synth_concat = np.mean([results['concat_mses'][i] for i in synthetic])
    avg_synth_attn = np.mean([results['attn_mses'][i] for i in synthetic])
    avg_synth_imp = (avg_synth_concat - avg_synth_attn) / (avg_synth_concat + 1e-8) * 100
    
    avg_real_concat = np.mean([results['concat_mses'][i] for i in real_like])
    avg_real_attn = np.mean([results['attn_mses'][i] for i in real_like])
    avg_real_imp = (avg_real_concat - avg_real_attn) / (avg_real_concat + 1e-8) * 100
    
    print(f"\nSynthetic Data:")
    print(f"  Concat MSE: {avg_synth_concat:.6f}")
    print(f"  Attention MSE: {avg_synth_attn:.6f}")
    print(f"  Improvement: {avg_synth_imp:+.1f}%")
    
    print(f"\nReal-Robot-Like Data:")
    print(f"  Concat MSE: {avg_real_concat:.6f}")
    print(f"  Attention MSE: {avg_real_attn:.6f}")
    print(f"  Improvement: {avg_real_imp:+.1f}%")
    
    # Determine if there's a gap
    gap = avg_real_imp - avg_synth_imp
    print(f"\nGap (Real - Synth): {gap:+.1f}%")
    
    status = "SUPPORTED" if abs(gap) > 10 else "INCONCLUSIVE"
    note = f"Gap of {gap:+.1f}% between synthetic ({avg_synth_imp:+.1f}%) and real ({avg_real_imp:+.1f}%)"
    
    print(f"\nStatus: {status}")
    print(f"Note: {note}")
    
    results['summary'] = {
        'avg_synthetic_concat_mse': avg_synth_concat,
        'avg_synthetic_attn_mse': avg_synth_attn,
        'avg_synthetic_improvement': avg_synth_imp,
        'avg_reallike_concat_mse': avg_real_concat,
        'avg_reallike_attn_mse': avg_real_attn,
        'avg_reallike_improvement': avg_real_imp,
        'gap': gap,
        'status': status,
        'note': note
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
    return results

if __name__ == '__main__':
    results = run_experiment()