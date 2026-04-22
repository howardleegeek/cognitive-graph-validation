"""
H1.37: Hierarchical Attention
Testing if hierarchical attention improves over flat attention
"""

import numpy as np

def generate_hierarchical_task(n_levels=3, n_samples=300):
    """Generate hierarchical task with multiple abstraction levels"""
    np.random.seed(42)
    states = []
    
    for i in range(n_samples):
        # Level 0: raw features
        level0 = np.random.randn(16) * 0.1
        # Level 1: mid-level abstraction
        level1 = np.tanh(level0 @ np.random.randn(16, 8))
        # Level 2: high-level abstraction
        level2 = np.tanh(level1 @ np.random.randn(8, 4))
        
        state = np.concatenate([level0, level1, level2])
        states.append(state)
    
    return np.array(states)

def flat_attention(x, dims=512):
    """Flat attention"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    attn = np.sum(h, axis=0, keepdims=True) / (h.shape[0] + 1e-8)
    return np.mean((h * np.tanh(attn)) ** 2)

def hierarchical_attention(x, dims=512):
    """Hierarchical attention with level-specific processing"""
    w0 = np.random.randn(16, dims) * np.sqrt(2.0 / (16 + dims))
    w1 = np.random.randn(8, dims) * np.sqrt(2.0 / (8 + dims))
    w2 = np.random.randn(4, dims) * np.sqrt(2.0 / (4 + dims))
    
    level0 = x[:, :16]
    level1 = x[:, 16:24]
    level2 = x[:, 24:28]
    
    h0 = np.tanh(level0 @ w0)
    h1 = np.tanh(level1 @ w1)
    h2 = np.tanh(level2 @ w2)
    
    # Cross-level attention
    attn1 = np.sum(h1, axis=0, keepdims=True) / (h1.shape[0] + 1e-8)
    h1_attended = h1 * np.tanh(attn1)
    
    attn2 = np.sum(h2, axis=0, keepdims=True) / (h2.shape[0] + 1e-8)
    h2_attended = h2 * np.tanh(attn2)
    
    combined = np.concatenate([h0, h1_attended, h2_attended], axis=1)
    return np.mean(combined ** 2)

def run_experiment():
    """Run H1.37 experiment"""
    results = []
    
    for _ in range(5):
        X = generate_hierarchical_task(n_levels=3, n_samples=300)
        
        flat_losses = []
        hier_losses = []
        
        for run in range(5):
            np.random.seed(42 + run)
            flat_losses.append(flat_attention(X))
            hier_losses.append(hierarchical_attention(X))
        
        flat_mse = np.mean(flat_losses)
        hier_mse = np.mean(hier_losses)
        improvement = (flat_mse - hier_mse) / flat_mse * 100
        
        print(f"Flat={flat_mse:.4f}, Hierarchical={hier_mse:.4f}, Δ={improvement:+.1f}%")
        results.append({'flat': flat_mse, 'hier': hier_mse, 'improvement': improvement})
    
    avg = np.mean([r['improvement'] for r in results])
    print(f"\nAverage: {avg:+.1f}%")
    
    status = "SUPPORTED" if avg > 5 else "REFUTED"
    print(f"Status: {status}")
    
    return results, status, avg

if __name__ == "__main__":
    results, status, avg = run_experiment()