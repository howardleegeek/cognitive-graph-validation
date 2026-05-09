#!/usr/bin/env python3
"""
H3.85: Investigating Attention Failure on Multi-Object Tasks
Building on:
- H3.83: Attention (-47.0%) fails on multi-object with interactions
- H3.84: Attention (+25.2%) succeeds on multi-object in different setup

Hypothesis: The difference is due to HOW attention is applied, not WHAT is attended.
Key insight: H3.84 uses actual weighted attention, H3.83 uses multi-scale windowing.
"""

import json
import numpy as np
import random
from datetime import datetime

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def generate_multi_object_sequence(num_objects=3, seq_len=50, complexity='high'):
    """Generate multi-object interaction sequences with varying complexity."""
    states = []
    for i in range(seq_len):
        t = i / seq_len
        state = {
            'timestep': i,
            'physical': np.random.randn(256),
            'semantic': np.random.randn(256),
            'num_objects': num_objects,
        }
        
        # Object states
        for obj in range(num_objects):
            state[f'obj_{obj}_pos'] = [0.3 * np.sin(t + obj), 0.2 * t, 0.1]
            state[f'obj_{obj}_vel'] = [0.1 * np.cos(t + obj), 0.1, 0]
            state[f'obj_{obj}_contact'] = 1.0 if (t > 0.3 + obj * 0.1 and t < 0.6 + obj * 0.1) else 0.0
        
        # Task complexity affects interaction patterns
        if complexity == 'high':
            # Complex: objects interact with each other
            state['interaction'] = 1.0 if num_objects > 1 else 0.0
        else:
            # Simple: objects are independent
            state['interaction'] = 0.0
        
        states.append(state)
    return states

def concat_baseline(states, embed_dim=512):
    """Concatenation baseline - H3.83's winning method."""
    embeddings = []
    for s in states:
        obj_feats = []
        for obj in range(s['num_objects']):
            obj_feats.extend(s[f'obj_{obj}_pos'] + s[f'obj_{obj}_vel'] + [s[f'obj_{obj}_contact']])
        
        emb = np.concatenate([
            s['physical'], s['semantic'],
            np.array(obj_feats)
        ])
        embeddings.append(emb[:embed_dim] if len(emb) >= embed_dim else np.pad(emb, (0, embed_dim - len(emb))))
    return np.mean(embeddings, axis=0)

def multi_scale_attention(states, windows=[3, 5, 7], embed_dim=512):
    """H3.83's multi-scale attention - FAILS (-47.0%)."""
    seq_len = len(states)
    all_outputs = []
    
    for w in windows:
        if w > seq_len:
            continue
        window_states = states[-w:]
        embeddings = []
        for s in window_states:
            obj_feats = []
            for obj in range(s['num_objects']):
                obj_feats.extend(s[f'obj_{obj}_pos'] + s[f'obj_{obj}_vel'] + [s[f'obj_{obj}_contact']])
            
            emb = np.concatenate([
                s['physical'], s['semantic'][:len(obj_feats)],
                np.array(obj_feats)
            ])
            embeddings.append(emb[:embed_dim] if len(emb) >= embed_dim else np.pad(emb, (0, embed_dim - len(emb))))
        
        embeddings = np.array(embeddings)
        query = embeddings[-1]
        # Simple dot-product without proper weighting
        scores = np.dot(query, embeddings.T) / (np.sqrt(embed_dim) + 1e-8)
        weights = scores / (scores.sum() + 1e-8)
        output = np.dot(weights, embeddings)
        all_outputs.append(output)
    
    return np.mean(all_outputs, axis=0)

def weighted_attention(states, embed_dim=512):
    """H3.84's weighted attention - SUCCEEDS (+25.2%)."""
    seq_len = len(states)
    embeddings = []
    
    for s in states:
        obj_feats = []
        for obj in range(s['num_objects']):
            obj_feats.extend(s[f'obj_{obj}_pos'] + s[f'obj_{obj}_vel'] + [s[f'obj_{obj}_contact']])
        
        emb = np.concatenate([
            s['physical'], s['semantic'],
            np.array(obj_feats)
        ])
        embeddings.append(emb[:embed_dim] if len(emb) >= embed_dim else np.pad(emb, (0, embed_dim - len(emb))))
    
    embeddings = np.array(embeddings)
    query = embeddings[-1]
    
    # Proper weighted attention with decay
    scores = np.dot(query, embeddings.T) / (np.sqrt(embed_dim) + 1e-8)
    decay = np.exp(-0.01 * np.arange(seq_len))
    scores = scores * decay[::-1]
    weights = scores / (scores.sum() + 1e-8)
    return np.dot(weights, embeddings)

def object_conditioned_attention(states, embed_dim=512):
    """NEW: Condition attention on object identity."""
    seq_len = len(states)
    num_objects = states[0]['num_objects']
    
    # Create object-aware embeddings
    all_attns = []
    for obj in range(num_objects):
        obj_embeddings = []
        for s in states:
            obj_feats = s[f'obj_{obj}_pos'] + s[f'obj_{obj}_vel'] + [s[f'obj_{obj}_contact']]
            emb = np.concatenate([
                s['physical'][:len(obj_feats)], 
                np.array(obj_feats)
            ])
            obj_embeddings.append(emb[:embed_dim] if len(emb) >= embed_dim else np.pad(emb, (0, embed_dim - len(emb))))
        
        obj_embeddings = np.array(obj_embeddings)
        query = obj_embeddings[-1]
        
        scores = np.dot(query, obj_embeddings.T) / (np.sqrt(embed_dim) + 1e-8)
        decay = np.exp(-0.01 * np.arange(seq_len))
        scores = scores * decay[::-1]
        weights = scores / (scores.sum() + 1e-8)
        all_attns.append(np.dot(weights, obj_embeddings))
    
    return np.mean(all_attns, axis=0)

def run_experiment():
    set_seed(42)
    
    print("=" * 70)
    print("H3.85: Investigating Attention Failure on Multi-Object Tasks")
    print("=" * 70)
    
    results = {
        'hypothesis': 'H3.85: Attention vs Multi-Scale on Multi-Object',
        'timestamp': datetime.now().isoformat(),
        'parent_hypotheses': ['H3.83', 'H3.84'],
        'methodology': 'Compare multi-scale attention vs weighted attention on multi-object tasks',
        'task_complexities': ['low', 'high'],
        'num_objects': [2, 3, 4],
        'sequence_lengths': [30, 50, 70],
        'concat_mses': [],
        'multi_scale_mses': [],
        'weighted_attn_mses': [],
        'obj_cond_mses': [],
    }
    
    import itertools
    configs = list(itertools.product(results['task_complexities'], results['num_objects'], results['sequence_lengths']))
    
    for complexity, num_obj, seq_len in configs:
        print(f"\n{complexity} complexity, {num_obj} objects, {seq_len} steps:")
        
        states = generate_multi_object_sequence(num_obj, seq_len, complexity)
        actions = [np.random.randn(7) for _ in states]
        
        # Baseline
        concat_emb = concat_baseline(states)
        concat_pred = np.tanh(concat_emb[:7])
        concat_mse = np.mean([np.mean((concat_pred - a) ** 2) for a in actions[-10:]])
        
        # Multi-scale attention (H3.83 style)
        ms_emb = multi_scale_attention(states)
        ms_pred = np.tanh(ms_emb[:7])
        ms_mse = np.mean([np.mean((ms_pred - a) ** 2) for a in actions[-10:]])
        
        # Weighted attention (H3.84 style)
        wa_emb = weighted_attention(states)
        wa_pred = np.tanh(wa_emb[:7])
        wa_mse = np.mean([np.mean((wa_pred - a) ** 2) for a in actions[-10:]])
        
        # Object-conditioned attention
        oc_emb = object_conditioned_attention(states)
        oc_pred = np.tanh(oc_emb[:7])
        oc_mse = np.mean([np.mean((oc_pred - a) ** 2) for a in actions[-10:]])
        
        # Calculate improvements
        ms_imp = (concat_mse - ms_mse) / (concat_mse + 1e-8) * 100
        wa_imp = (concat_mse - wa_mse) / (concat_mse + 1e-8) * 100
        oc_imp = (concat_mse - oc_mse) / (concat_mse + 1e-8) * 100
        
        print(f"  Concat: {concat_mse:.4f}")
        print(f"  Multi-Scale (H3.83): {ms_mse:.4f} ({ms_imp:+.1f}%)")
        print(f"  Weighted Attn (H3.84): {wa_mse:.4f} ({wa_imp:+.1f}%)")
        print(f"  Obj-Cond Attn: {oc_mse:.4f} ({oc_imp:+.1f}%)")
        
        results['concat_mses'].append(concat_mse)
        results['multi_scale_mses'].append(ms_mse)
        results['weighted_attn_mses'].append(wa_mse)
        results['obj_cond_mses'].append(oc_mse)
    
    # Summary
    avg_concat = np.mean(results['concat_mses'])
    avg_ms = np.mean(results['multi_scale_mses'])
    avg_wa = np.mean(results['weighted_attn_mses'])
    avg_oc = np.mean(results['obj_cond_mses'])
    
    ms_imp = (avg_concat - avg_ms) / (avg_concat + 1e-8) * 100
    wa_imp = (avg_concat - avg_wa) / (avg_concat + 1e-8) * 100
    oc_imp = (avg_concat - avg_oc) / (avg_concat + 1e-8) * 100
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Concatenation: {avg_concat:.4f}")
    print(f"Multi-Scale (H3.83): {avg_ms:.4f} ({ms_imp:+.1f}%)")
    print(f"Weighted Attn (H3.84): {avg_wa:.4f} ({wa_imp:+.1f}%)")
    print(f"Object-Conditioned: {avg_oc:.4f} ({oc_imp:+.1f}%)")
    
    # Determine best method
    best_method = "Multi-Scale" if ms_imp > wa_imp and ms_imp > oc_imp else \
                  "Weighted" if wa_imp > oc_imp else "Object-Conditioned"
    best_imp = max(ms_imp, wa_imp, oc_imp)
    
    print(f"\nBest method: {best_method} ({best_imp:+.1f}%)")
    
    status = "SUPPORTED" if wa_imp > ms_imp else "REFUTED"
    note = f"Weighted attention ({wa_imp:+.1f}%) {'>' if wa_imp > ms_imp else '<'} Multi-Scale ({ms_imp:+.1f}%)"
    
    print(f"\nStatus: {status}")
    print(f"Note: {note}")
    
    results['summary'] = {
        'avg_concat_mse': avg_concat,
        'avg_multi_scale_mse': avg_ms,
        'avg_weighted_attn_mse': avg_wa,
        'avg_obj_cond_mse': avg_oc,
        'multi_scale_improvement': ms_imp,
        'weighted_attn_improvement': wa_imp,
        'obj_cond_improvement': oc_imp,
        'best_method': best_method,
        'status': status,
        'note': note
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
    return results

if __name__ == '__main__':
    results = run_experiment()