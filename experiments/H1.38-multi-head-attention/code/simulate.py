"""
H1.38: Multi-Head Attention vs Single-Head
Testing if multiple attention heads provide benefit
"""

import numpy as np

def generate_long_task(n_steps=35, n_samples=300):
    """Generate long sequence task"""
    np.random.seed(42)
    states = []
    for i in range(n_samples):
        s = np.random.randn(12) * 0.1
        for t in range(n_steps):
            a = np.random.randn(7) * 0.1
            ns = s + np.dot(np.random.randn(12, 7), a) * 0.5 + np.random.randn(12) * 0.01
            s = ns
            states.append(s.copy())
    return np.array(states)

def single_head(x, dims=512):
    """Single attention head"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    attn = np.sum(h, axis=0, keepdims=True) / (h.shape[0] + 1e-8)
    return np.mean((h * np.tanh(attn)) ** 2)

def multi_head(x, dims=512, heads=4):
    """Multi-head attention"""
    w1 = np.random.randn(x.shape[1], dims * heads) * np.sqrt(2.0 / (x.shape[1] + dims * heads))
    h = np.tanh(x @ w1)
    # Split into heads
    h_heads = h.reshape(h.shape[0], heads, dims)
    # Each head computes attention
    results = []
    for i in range(heads):
        hi = h_heads[:, i, :]
        attn = np.sum(hi, axis=0, keepdims=True) / (hi.shape[0] + 1e-8)
        results.append(hi * np.tanh(attn))
    # Concatenate heads
    combined = np.concatenate(results, axis=1)
    return np.mean(combined ** 2)

def run_experiment():
    X = generate_long_task(n_steps=35, n_samples=300)
    
    single_losses = []
    multi_losses = []
    
    for run in range(5):
        np.random.seed(42 + run)
        single_losses.append(single_head(X))
        multi_losses.append(multi_head(X))
    
    single_mse = np.mean(single_losses)
    multi_mse = np.mean(multi_losses)
    improvement = (single_mse - multi_mse) / single_mse * 100
    
    print(f"Single={single_mse:.4f}, Multi={multi_mse:.4f}, Δ={improvement:+.1f}%")
    status = "SUPPORTED" if improvement > 5 else "REFUTED"
    print(f"Status: {status}")
    return {'single': single_mse, 'multi': multi_mse, 'improvement': improvement}, status, improvement

if __name__ == "__main__":
    results, status, improvement = run_experiment()