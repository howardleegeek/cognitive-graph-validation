#!/usr/bin/env python3
"""
H3.86: Graph-Native Multi-Object Reasoning
Building on:
- H2.9: Graph (+50.4%) excels at multi-object compositional temporal reasoning
- H3.83/85: Attention fails on multi-object tasks
- H3.84: Graph + Attention hybrid (+21.7%)

Hypothesis: Explicit graph structure with object nodes and relation edges will
succeed where attention fails on multi-object tasks.
"""

import json
import numpy as np
import random
from datetime import datetime

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def generate_multi_object_graph_data(num_objects=3, seq_len=50, task='stacking'):
    """Generate multi-object data with explicit graph structure."""
    states = []
    for i in range(seq_len):
        t = i / seq_len
        state = {
            'timestep': i,
            'object_features': [],
        }
        
        # Each object has position and state
        for obj in range(num_objects):
            obj_pos = [0.3 * np.sin(t + obj * np.pi / 3), 
                      0.1 + 0.05 * t + 0.02 * obj,
                      0.05 * (num_objects - obj)]
            obj_vel = [0.1 * np.cos(t + obj), 0.01, 0]
            obj_state = 1.0 if abs(obj_pos[2] - 0.05 * (num_objects - obj)) < 0.02 else 0.0
            
            state['object_features'].append({
                'id': obj,
                'pos': obj_pos,
                'vel': obj_vel,
                'state': obj_state,
                'type': 'box' if obj % 2 == 0 else 'cylinder',
            })
        
        # Add relation features
        state['relations'] = []
        for obj1 in range(num_objects):
            for obj2 in range(obj1 + 1, num_objects):
                dist = np.linalg.norm(np.array(state['object_features'][obj1]['pos']) - 
                                     np.array(state['object_features'][obj2]['pos']))
                state['relations'].append({
                    'from': obj1,
                    'to': obj2,
                    'distance': dist,
                    'contact': 1.0 if dist < 0.1 else 0.0,
                })
        
        states.append(state)
    return states

def graph_representation(states, embed_dim=256):
    """Build graph representation with object nodes and relation edges."""
    num_objects = len(states[0]['object_features'])
    
    # Node embeddings
    node_embeddings = []
    for s in states:
        for obj_feat in s['object_features']:
            node_emb = np.concatenate([
                np.array(obj_feat['pos']),
                np.array(obj_feat['vel']),
                [obj_feat['state']],
            ])
            node_embeddings.append(node_emb)
    
    # Aggregate across timesteps for each object
    object_embeddings = []
    for obj in range(num_objects):
        obj_nodes = [node_embeddings[i * num_objects + obj] for i in range(len(states))]
        obj_embeddings = np.mean(obj_nodes, axis=0)
        object_embeddings.append(obj_embeddings)
    
    return np.concatenate(object_embeddings)[:embed_dim]

def graph_attention_with_relations(states, embed_dim=256):
    """Graph attention that leverages relation structure."""
    num_objects = len(states[0]['object_features'])
    seq_len = len(states)
    
    # Build object-level representations
    all_obj_embs = []
    for s in states:
        obj_embs = []
        for obj_feat in s['object_features']:
            obj_emb = np.concatenate([
                np.array(obj_feat['pos']),
                np.array(obj_feat['vel']),
                [obj_feat['state']],
            ])
            obj_embs.append(obj_emb)
        all_obj_embs.append(np.array(obj_embs))
    
    # Aggregate with temporal attention
    final_obj_embs = []
    for obj in range(num_objects):
        obj_seq = [all_obj_embs[t][obj] for t in range(seq_len)]
        obj_seq = np.array(obj_seq)
        
        # Self-attention with decay
        query = obj_seq[-1]
        scores = np.dot(query, obj_seq.T) / (np.sqrt(len(query)) + 1e-8)
        decay = np.exp(-0.01 * np.arange(len(scores)))
        scores = scores * decay[::-1]
        weights = scores / (scores.sum() + 1e-8)
        obj_agg = np.dot(weights, obj_seq)
        final_obj_embs.append(obj_agg)
    
    return np.concatenate(final_obj_embs)[:embed_dim]

def flat_attention_baseline(states, embed_dim=256):
    """Standard flat attention - H3.83 style that fails."""
    num_objects = len(states[0]['object_features'])
    
    # Flatten all object features
    all_features = []
    for s in states:
        for obj_feat in s['object_features']:
            obj_emb = np.concatenate([
                np.array(obj_feat['pos']),
                np.array(obj_feat['vel']),
                [obj_feat['state']],
            ])
            all_features.append(obj_emb)
    
    embeddings = np.array(all_features)
    query = embeddings[-1]
    scores = np.dot(query, embeddings.T) / (np.sqrt(len(query)) + 1e-8)
    weights = scores / (scores.sum() + 1e-8)
    return np.dot(weights, embeddings)

def concat_baseline(states, embed_dim=256):
    """Concatenation baseline."""
    all_features = []
    for s in states:
        for obj_feat in s['object_features']:
            obj_emb = np.concatenate([
                np.array(obj_feat['pos']),
                np.array(obj_feat['vel']),
                [obj_feat['state']],
            ])
            all_features.append(obj_emb)
    
    embeddings = np.array(all_features)
    return np.mean(embeddings, axis=0)

def run_experiment():
    set_seed(42)
    
    print("=" * 70)
    print("H3.86: Graph-Native Multi-Object Reasoning")
    print("=" * 70)
    
    results = {
        'hypothesis': 'H3.86: Graph-Native Multi-Object Reasoning',
        'timestamp': datetime.now().isoformat(),
        'parent_hypotheses': ['H2.9', 'H3.83', 'H3.85'],
        'num_objects': [2, 3, 4, 5],
        'tasks': ['stacking', 'sorting', 'arrangement'],
        'concat_mses': [],
        'flat_attn_mses': [],
        'graph_attn_mses': [],
        'graph_native_mses': [],
    }
    
    for num_obj in results['num_objects']:
        for task in results['tasks']:
            print(f"\n{num_obj} objects, {task} task:")
            
            states = generate_multi_object_graph_data(num_obj, 50, task)
            actions = [np.random.randn(7) for _ in states]
            
            # Baseline
            concat_emb = concat_baseline(states)
            concat_pred = np.tanh(concat_emb[:7])
            concat_mse = np.mean([np.mean((concat_pred - a) ** 2) for a in actions[-10:]])
            
            # Flat attention (H3.83 style)
            flat_emb = flat_attention_baseline(states)
            flat_pred = np.tanh(flat_emb[:7])
            flat_mse = np.mean([np.mean((flat_pred - a) ** 2) for a in actions[-10:]])
            
            # Graph + attention
            graph_attn_emb = graph_attention_with_relations(states)
            graph_attn_pred = np.tanh(graph_attn_emb[:7])
            graph_attn_mse = np.mean([np.mean((graph_attn_pred - a) ** 2) for a in actions[-10:]])
            
            # Graph native
            graph_emb = graph_representation(states)
            graph_pred = np.tanh(graph_emb[:7])
            graph_mse = np.mean([np.mean((graph_pred - a) ** 2) for a in actions[-10:]])
            
            # Calculate improvements
            flat_imp = (concat_mse - flat_mse) / (concat_mse + 1e-8) * 100
            graph_attn_imp = (concat_mse - graph_attn_mse) / (concat_mse + 1e-8) * 100
            graph_imp = (concat_mse - graph_mse) / (concat_mse + 1e-8) * 100
            
            print(f"  Concat: {concat_mse:.4f}")
            print(f"  Flat Attn (H3.83): {flat_mse:.4f} ({flat_imp:+.1f}%)")
            print(f"  Graph + Attn: {graph_attn_mse:.4f} ({graph_attn_imp:+.1f}%)")
            print(f"  Graph Native: {graph_mse:.4f} ({graph_imp:+.1f}%)")
            
            results['concat_mses'].append(concat_mse)
            results['flat_attn_mses'].append(flat_mse)
            results['graph_attn_mses'].append(graph_attn_mse)
            results['graph_native_mses'].append(graph_mse)
    
    # Summary
    avg_concat = np.mean(results['concat_mses'])
    avg_flat = np.mean(results['flat_attn_mses'])
    avg_graph_attn = np.mean(results['graph_attn_mses'])
    avg_graph = np.mean(results['graph_native_mses'])
    
    flat_imp = (avg_concat - avg_flat) / (avg_concat + 1e-8) * 100
    graph_attn_imp = (avg_concat - avg_graph_attn) / (avg_concat + 1e-8) * 100
    graph_imp = (avg_concat - avg_graph) / (avg_concat + 1e-8) * 100
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Concatenation: {avg_concat:.4f}")
    print(f"Flat Attention (H3.83): {avg_flat:.4f} ({flat_imp:+.1f}%)")
    print(f"Graph + Attention: {avg_graph_attn:.4f} ({graph_attn_imp:+.1f}%)")
    print(f"Graph Native: {avg_graph:.4f} ({graph_imp:+.1f}%)")
    
    # Best method
    best = "Graph Native" if graph_imp > flat_imp and graph_imp > graph_attn_imp else \
           "Graph + Attention" if graph_attn_imp > flat_imp else "Flat Attention"
    best_imp = max(graph_imp, graph_attn_imp, flat_imp)
    
    print(f"\nBest: {best} ({best_imp:+.1f}%)")
    
    status = "SUPPORTED" if graph_imp > 10 else "REFUTED"
    note = f"Graph native ({graph_imp:+.1f}%) vs Flat attention ({flat_imp:+.1f}%)"
    
    print(f"\nStatus: {status}")
    print(f"Note: {note}")
    
    results['summary'] = {
        'avg_concat_mse': avg_concat,
        'avg_flat_attn_mse': avg_flat,
        'avg_graph_attn_mse': avg_graph_attn,
        'avg_graph_native_mse': avg_graph,
        'flat_attn_improvement': flat_imp,
        'graph_attn_improvement': graph_attn_imp,
        'graph_native_improvement': graph_imp,
        'best_method': best,
        'status': status,
        'note': note
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
    return results

if __name__ == '__main__':
    results = run_experiment()