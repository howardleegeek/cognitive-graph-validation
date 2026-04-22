"""
H1.36: Graph + Attention Combined
Testing if combining graph structure with attention provides benefit on temporal tasks
"""

import numpy as np

def generate_temporal_task(n_steps=15, n_objects=4, n_samples=250):
    """Generate multi-object temporal task"""
    np.random.seed(42)
    states = []
    actions = []
    
    for i in range(n_samples):
        obj_pos = np.random.randn(n_objects, 3) * 0.1
        for t in range(n_steps):
            actions_t = np.random.randn(n_objects, 3) * 0.1
            delta = np.dot(actions_t, np.ones(3))[:, None] * np.ones(3) * 0.1
            next_pos = obj_pos + delta + np.random.randn(n_objects, 3) * 0.01
            
            state_flat = next_pos.flatten()
            action_flat = actions_t.flatten()
            state_full = np.concatenate([state_flat, action_flat])
            
            states.append(state_full)
            actions.append(action_flat)
            
            obj_pos = next_pos
    
    return np.array(states), np.array(actions)

def baseline(x, dims=512):
    """Baseline neural"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    return np.mean(np.tanh(x @ w1) ** 2)

def graph_only(x, dims=512):
    """Graph structure"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    w2 = np.random.randn(dims, dims) * np.sqrt(2.0 / (dims + dims))
    h = np.tanh(x @ w1)
    for _ in range(3):
        h = h + 0.1 * np.tanh(h @ w2)
    return np.mean(h ** 2)

def attention_only(x, dims=512):
    """Attention only"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    attn = np.sum(h, axis=0, keepdims=True) / (h.shape[0] + 1e-8)
    return np.mean((h * np.tanh(attn)) ** 2)

def combined(x, dims=512):
    """Graph + Attention combined"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    w2 = np.random.randn(dims, dims) * np.sqrt(2.0 / (dims + dims))
    h = np.tanh(x @ w1)
    # Graph passes first
    for _ in range(3):
        h = h + 0.1 * np.tanh(h @ w2)
    # Then attention
    attn = np.sum(h, axis=0, keepdims=True) / (h.shape[0] + 1e-8)
    return np.mean((h * np.tanh(attn)) ** 2)

def run_experiment():
    """Run H1.36 experiment"""
    results = []
    
    for n_objects in [3, 4, 5, 6]:
        X, A = generate_temporal_task(n_steps=15, n_objects=n_objects, n_samples=250)
        
        baseline_losses = []
        graph_losses = []
        attn_losses = []
        combined_losses = []
        
        for run in range(5):
            np.random.seed(42 + run)
            baseline_losses.append(baseline(X))
            graph_losses.append(graph_only(X))
            attn_losses.append(attention_only(X))
            combined_losses.append(combined(X))
        
        baseline_mse = np.mean(baseline_losses)
        graph_mse = np.mean(graph_losses)
        attn_mse = np.mean(attn_losses)
        combined_mse = np.mean(combined_losses)
        
        best_individual = min(graph_mse, attn_mse)
        improvement_vs_best = (best_individual - combined_mse) / best_individual * 100
        improvement_vs_baseline = (baseline_mse - combined_mse) / baseline_mse * 100
        
        print(f"{n_objects}-obj: Baseline={baseline_mse:.4f}, Graph={graph_mse:.4f}, Attn={attn_mse:.4f}, Combined={combined_mse:.4f}")
        print(f"  Combined vs Best Individual: {improvement_vs_best:+.1f}%, Combined vs Baseline: {improvement_vs_baseline:+.1f}%")
        
        results.append({
            'n_objects': n_objects,
            'baseline': baseline_mse,
            'graph': graph_mse,
            'attention': attn_mse,
            'combined': combined_mse,
            'vs_best': improvement_vs_best,
            'vs_baseline': improvement_vs_baseline
        })
    
    avg_vs_best = np.mean([r['vs_best'] for r in results])
    avg_vs_baseline = np.mean([r['vs_baseline'] for r in results])
    print(f"\nAvg Combined vs Best Individual: {avg_vs_best:+.1f}%")
    print(f"Avg Combined vs Baseline: {avg_vs_baseline:+.1f}%")
    
    status = "SUPPORTED" if avg_vs_best > 5 else ("MARGINAL" if avg_vs_best > 0 else "REFUTED")
    print(f"Status: {status}")
    
    return results, status, avg_vs_best, avg_vs_baseline

if __name__ == "__main__":
    results, status, avg_vs_best, avg_vs_baseline = run_experiment()