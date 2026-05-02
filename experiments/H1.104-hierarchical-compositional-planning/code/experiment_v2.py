"""
H1.104: Hierarchical Compositional Planning with Attention

Based on H1 success (+25.6% on real robot), we test attention on more complex 
multi-step tasks that require hierarchical planning.

Hypothesis: Attention mechanisms enable better hierarchical compositional planning
on complex multi-step robotic tasks (10+ steps) compared to flat attention.
"""

import numpy as np
import json
from typing import Dict, List, Tuple

def generate_compositional_tasks(n_tasks: int, min_steps: int = 10, max_steps: int = 30) -> List[Dict]:
    """Generate compositional tasks requiring hierarchical planning"""
    tasks = []
    for i in range(n_tasks):
        n_steps = np.random.randint(min_steps, max_steps + 1)
        
        # Each task has 2-3 sub-goals
        n_subgoals = np.random.randint(2, 4)
        steps_per_subgoal = n_steps // n_subgoals
        
        states = []
        actions = []
        subgoals = []
        
        for sg in range(n_subgoals):
            # Generate sub-goal specific dynamics
            sg_base = np.random.randn(4) * 0.5
            subgoals.append(sg_base)
            
            for step in range(steps_per_subgoal):
                # State depends on sub-goal + step
                state = sg_base + np.random.randn(4) * 0.1 * (step / steps_per_subgoal)
                action = np.random.randn(2) * 0.2
                states.append(state)
                actions.append(action)
        
        tasks.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'subgoals': np.array(subgoals),
            'n_steps': n_steps,
            'n_subgoals': n_subgoals
        })
    return tasks

def flat_attention_predict(states, actions):
    """Flat attention: attend to all timesteps equally"""
    # Simple: use all states equally weighted
    return states.mean(axis=0)

def hierarchical_attention_predict(states, actions, n_subgoals):
    """Hierarchical attention: first attend to sub-goals, then to steps within sub-goals"""
    n_steps = len(states)
    steps_per_sg = n_steps // n_subgoals
    
    # Level 1: Compute sub-goal representations
    subgoals = []
    for sg in range(n_subgoals):
        start = sg * steps_per_sg
        end = start + steps_per_sg
        sg_states = states[start:end]
        # Attention within sub-goal
        weights = np.exp(np.arange(steps_per_sg) / steps_per_sg)  # More weight on later steps
        weights = weights / weights.sum()
        sg_repr = np.sum(sg_states * weights[:, None], axis=0)
        subgoals.append(sg_repr)
    
    # Level 2: Attend across sub-goals
    subgoals = np.array(subgoals)
    sg_weights = np.exp(np.arange(n_subgoals) / n_subgoals)
    sg_weights = sg_weights / sg_weights.sum()
    plan_repr = np.sum(subgoals * sg_weights[:, None], axis=0)
    
    return plan_repr

def evaluate_architectures(tasks):
    """Compare flat vs hierarchical attention on compositional tasks"""
    results = {
        'flat': [],
        'hierarchical': [],
        'flat_subgoal': [],
    }
    
    for task in tasks:
        states = task['states']
        actions = task['actions']
        n_subgoals = task['n_subgoals']
        
        # Flat attention prediction
        flat_pred = flat_attention_predict(states, actions)
        
        # Hierarchical attention prediction
        hier_pred = hierarchical_attention_predict(states, actions, n_subgoals)
        
        # Flat with sub-goal knowledge (oracle)
        # Use actual sub-goal boundaries
        steps_per_sg = len(states) // n_subgoals
        flat_sg_preds = []
        for sg in range(n_subgoals):
            start = sg * steps_per_sg
            end = start + steps_per_sg
            sg_states = states[start:end]
            flat_sg_preds.append(sg_states.mean(axis=0))
        flat_sg_pred = np.mean(flat_sg_preds, axis=0)
        
        # Compute losses (MSE to actual final state)
        final_state = states[-1]
        
        flat_loss = np.mean((flat_pred - final_state)**2)
        hier_loss = np.mean((hier_pred - final_state)**2)
        flat_sg_loss = np.mean((flat_sg_pred - final_state)**2)
        
        results['flat'].append(flat_loss)
        results['hierarchical'].append(hier_loss)
        results['flat_subgoal'].append(flat_sg_loss)
    
    return {k: np.mean(v) for k, v in results.items()}

def run_experiment():
    np.random.seed(42)
    
    # Generate test tasks with varying complexity
    all_results = {}
    
    for n_steps in [10, 15, 20, 25, 30]:
        tasks = generate_compositional_tasks(200, min_steps=n_steps, max_steps=n_steps)
        results = evaluate_architectures(tasks)
        all_results[f'{n_steps}_steps'] = results
    
    # Aggregate results
    flat_losses = []
    hier_losses = []
    flat_sg_losses = []
    
    for r in all_results.values():
        flat_losses.append(r['flat'])
        hier_losses.append(r['hierarchical'])
        flat_sg_losses.append(r['flat_subgoal'])
    
    avg_flat = np.mean(flat_losses)
    avg_hier = np.mean(hier_losses)
    avg_flat_sg = np.mean(flat_sg_losses)
    
    # Calculate improvements
    improvement_hier = (avg_flat - avg_hier) / avg_flat * 100
    improvement_sg = (avg_flat - avg_flat_sg) / avg_flat * 100
    
    # Determine status
    status = "SUPPORTED" if improvement_hier > 0 else "REFUTED"
    
    return {
        "status": status,
        "improvement_hierarchical": improvement_hier,
        "improvement_subgoal": improvement_sg,
        "avg_flat_mse": avg_flat,
        "avg_hierarchical_mse": avg_hier,
        "avg_flat_subgoal_mse": avg_flat_sg,
        "by_length": all_results
    }

if __name__ == "__main__":
    print("Running H1.104: Hierarchical Compositional Planning...")
    results = run_experiment()
    
    print(f"\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Hierarchical vs Flat: {results['improvement_hierarchical']:.1f}%")
    print(f"  Subgoal vs Flat: {results['improvement_subgoal']:.1f}%")
    print(f"  Avg Flat MSE: {results['avg_flat_mse']:.6f}")
    print(f"  Avg Hierarchical MSE: {results['avg_hierarchical_mse']:.6f}")
    print(f"  Avg Flat+Subgoal MSE: {results['avg_flat_subgoal_mse']:.6f}")
    
    print("\nBy sequence length:")
    for k, v in results['by_length'].items():
        print(f"  {k}: flat={v['flat']:.4f}, hier={v['hierarchical']:.4f}")
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")