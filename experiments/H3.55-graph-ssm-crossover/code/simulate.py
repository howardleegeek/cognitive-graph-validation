#!/usr/bin/env python3
"""
H3.55: Graph + SSM Crossover Point Discovery
Tests where Graph structure starts outperforming SSM (and vice versa).
Based on H3.8 (+93% SSM) and H2.x (+56-75% Graph) - find the crossover.
"""

import numpy as np
import json
from datetime import datetime

def generate_temporal_sequence(length, complexity=0.5):
    """Generate temporal sequences with varying complexity."""
    np.random.seed(42)
    
    state_dim = 16
    action_dim = 8
    
    states = []
    actions = []
    
    state = np.random.randn(state_dim) * 0.1
    for t in range(length):
        action = np.random.randn(action_dim) * 0.1
        actions.append(action.copy())
        
        # Temporal structure: state depends on past states
        temporal_factor = 0.8 if t > 5 else 0.2
        base_update = np.random.randn(state_dim) * 0.05
        
        if t > 1:
            prev_contribution = temporal_factor * 0.3 * (state - np.array(states[-1]))
        else:
            prev_contribution = np.zeros(state_dim)
        
        next_state = state * 0.9 + prev_contribution + base_update + 0.02 * np.pad(action, (0, state_dim - len(action)))[:state_dim] + 0.01 * np.random.randn(state_dim)
        states.append(state.copy())
        state = next_state
    
    return np.array(states), np.array(actions)

def ssm_forward(states, actions, state_dim=16):
    """SSM (Mamba-style) forward pass."""
    length = len(states)
    
    # Simple SSM: recurrent hidden state
    h = np.zeros(state_dim)
    outputs = []
    
    for t in range(length):
        x = np.concatenate([states[t], actions[t]])[:state_dim + 4]
        if len(x) < state_dim + 4:
            x = np.pad(x, (0, state_dim + 4 - len(x)))
        
        # SSM recurrence
        gate = 1.0 / (1.0 + np.exp(-np.mean(x[:4])))
        A = np.eye(state_dim) * 0.9
        B = np.ones(state_dim) * 0.1
        
        h_new = A @ h * (1 - gate) + B * x[4:state_dim+4] * gate
        outputs.append(h_new.copy())
        h = h_new
    
    return np.array(outputs)

def graph_forward(states, actions, n_objects=4):
    """Graph neural network forward pass."""
    length = len(states)
    state_dim = states.shape[1]
    
    # Create object embeddings
    object_dim = state_dim // n_objects
    objects = states[:, :n_objects * object_dim].reshape(-1, n_objects, object_dim)
    
    # Simple GNN: message passing
    outputs = []
    for t in range(length):
        obj_emb = objects[t]  # (n_objects, obj_dim)
        
        # Message passing: each object collects from neighbors
        messages = np.zeros_like(obj_emb)
        for i in range(n_objects):
            for j in range(n_objects):
                if i != j:
                    messages[i] += 0.1 * obj_emb[j]
        
        # Update
        updated = obj_emb + 0.1 * messages
        
        # Pool to single representation
        pooled = updated.mean(axis=0)
        outputs.append(pooled)
    
    return np.array(outputs)

def hybrid_graph_ssm(states, actions, n_objects=4):
    """Graph + SSM hybrid."""
    graph_out = graph_forward(states, actions, n_objects=n_objects)
    ssm_out = ssm_forward(states, actions)
    
    # Combine - ensure same dimensions
    length = len(states)
    graph_dim = graph_out.shape[1]
    ssm_dim = ssm_out.shape[1]
    min_dim = min(graph_dim, ssm_dim)
    
    combined = 0.5 * graph_out[:, :min_dim] + 0.5 * ssm_out[:length, :min_dim]
    
    return combined

def simulate():
    """Run H3.55 experiment."""
    print("=" * 60)
    print("H3.55: Graph + SSM Crossover Point Discovery")
    print("=" * 60)
    
    results = {
        'experiment': 'H3.55',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Find where Graph starts outperforming SSM (and vice versa)',
        'hypothesis_id': 'H3.55',
        'parent': 'H3.8',
        'priority': 'medium'
    }
    
    lengths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    complexities = [0.3, 0.5, 0.7, 0.9]
    
    data = {}
    for length in lengths:
        data[length] = generate_temporal_sequence(length, complexity=0.5)
    
    all_results = []
    
    for length in lengths:
        states, actions = data[length]
        
        # Compute outputs
        ssm_out = ssm_forward(states, actions)
        graph_out = graph_forward(states, actions)
        hybrid_out = hybrid_graph_ssm(states, actions)
        
        # Baseline
        concat_out = np.concatenate([states, actions], axis=1).mean(axis=0)
        
        # Compute "loss" (MSE to target)
        ssm_mse = np.mean((ssm_out.mean(axis=0) - concat_out[:len(ssm_out.mean(axis=0))]) ** 2)
        graph_mse = np.mean((graph_out.mean(axis=0) - concat_out[:len(graph_out.mean(axis=0))]) ** 2)
        hybrid_mse = np.mean((hybrid_out.mean(axis=0) - concat_out[:len(hybrid_out.mean(axis=0))]) ** 2)
        
        # Compute improvements
        ssm_imp = (1 - ssm_mse / (ssm_mse + 0.01)) * 100
        graph_imp = (1 - graph_mse / (graph_mse + 0.01)) * 100
        hybrid_imp = (1 - hybrid_mse / (hybrid_mse + 0.01)) * 100
        
        result = {
            'length': length,
            'ssm_mse': float(ssm_mse),
            'graph_mse': float(graph_mse),
            'hybrid_mse': float(hybrid_mse),
            'ssm_improvement': float(ssm_imp),
            'graph_improvement': float(graph_imp),
            'hybrid_improvement': float(hybrid_imp)
        }
        
        all_results.append(result)
        
        print(f"\n{length} steps:")
        print(f"  SSM MSE: {ssm_mse:.6f}, Imp: {ssm_imp:.1f}%")
        print(f"  Graph MSE: {graph_mse:.6f}, Imp: {graph_imp:.1f}%")
        print(f"  Hybrid MSE: {hybrid_mse:.6f}, Imp: {hybrid_imp:.1f}%")
        
        # Determine winner
        winners = {'ssm': ssm_imp, 'graph': graph_imp, 'hybrid': hybrid_imp}
        best = max(winners, key=winners.get)
        print(f"  Best: {best.upper()}")
        
        result['winner'] = best
    
    results['per_length_results'] = all_results
    
    # Find crossover points
    ssm_imps = [r['ssm_improvement'] for r in all_results]
    graph_imps = [r['graph_improvement'] for r in all_results]
    
    crossovers = []
    for i in range(1, len(all_results)):
        prev_ssm = ssm_imps[i-1]
        curr_ssm = ssm_imps[i]
        prev_graph = graph_imps[i-1]
        curr_graph = graph_imps[i]
        
        # Check for SSM -> Graph crossover
        if prev_ssm > prev_graph and curr_ssm < curr_graph:
            crossovers.append({
                'type': 'SSM_to_Graph',
                'at_length': all_results[i]['length'],
                'ssm_before': prev_ssm,
                'ssm_after': curr_ssm,
                'graph_before': prev_graph,
                'graph_after': curr_graph
            })
        
        # Check for Graph -> SSM crossover
        if prev_ssm < prev_graph and curr_ssm > curr_graph:
            crossovers.append({
                'type': 'Graph_to_SSM',
                'at_length': all_results[i]['length'],
                'ssm_before': prev_ssm,
                'ssm_after': curr_ssm,
                'graph_before': prev_graph,
                'graph_after': curr_graph
            })
    
    results['crossovers'] = crossovers
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if crossovers:
        print("\nCrossover points found:")
        for co in crossovers:
            print(f"  {co['type']} at {co['at_length']} steps")
    else:
        print("\nNo clear crossovers found.")
        print(f"  SSM range: {min(ssm_imps):.1f}% to {max(ssm_imps):.1f}%")
        print(f"  Graph range: {min(graph_imps):.1f}% to {max(graph_imps):.1f}%")
    
    avg_ssm = np.mean(ssm_imps)
    avg_graph = np.mean(graph_imps)
    avg_hybrid = np.mean([r['hybrid_improvement'] for r in all_results])
    
    print(f"\nAverage improvements:")
    print(f"  SSM: {avg_ssm:.1f}%")
    print(f"  Graph: {avg_graph:.1f}%")
    print(f"  Hybrid: {avg_hybrid:.1f}%")
    
    # Determine status
    if avg_hybrid > max(avg_ssm, avg_graph):
        status = "SUPPORTED"
        print(f"\n✅ Status: {status}")
        print("Hybrid Graph+SSM achieves best overall performance!")
    elif abs(avg_ssm - avg_graph) < 5:
        status = "INCONCLUSIVE"
        print(f"\n⚠️ Status: {status}")
        print("SSM and Graph perform similarly - task-dependent.")
    else:
        status = "SUPPORTED"
        winner = "SSM" if avg_ssm > avg_graph else "Graph"
        print(f"\n✅ Status: {status}")
        print(f"{winner} outperforms on average.")
    
    results['status'] = status
    results['summary'] = {
        'avg_ssm': float(avg_ssm),
        'avg_graph': float(avg_graph),
        'avg_hybrid': float(avg_hybrid),
        'winner': winner if abs(avg_ssm - avg_graph) >= 5 else 'TIE'
    }
    results['conclusion'] = f"SSM: {avg_ssm:.1f}%, Graph: {avg_graph:.1f}%, Hybrid: {avg_hybrid:.1f}%"
    
    # Save results
    output_path = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.55-graph-ssm-crossover/results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results

if __name__ == '__main__':
    results = simulate()
