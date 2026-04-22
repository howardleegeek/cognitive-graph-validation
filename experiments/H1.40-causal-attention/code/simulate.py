"""
H1.40: Causal Attention
Testing causal/masked attention for sequential tasks
"""

import numpy as np

def generate_sequential(n_steps=30, n_samples=300):
    np.random.seed(42)
    states = []
    for i in range(n_samples):
        s = np.random.randn(12) * 0.1
        for t in range(n_steps):
            a = np.random.randn(7) * 0.1
            ns = s + np.dot(np.random.randn(12, 7), a) * 0.5
            s = ns
            states.append(s.copy())
    return np.array(states)

def standard_attention(x, dims=512):
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    attn = np.sum(h, axis=0, keepdims=True) / (h.shape[0] + 1e-8)
    return np.mean((h * np.tanh(attn)) ** 2)

def causal_attention(x, dims=512):
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    # Simple causal: prioritize recent timesteps
    n = h.shape[0]
    weights = np.linspace(0.1, 1.0, n)[:, None]
    h_weighted = h * weights
    attn = np.sum(h_weighted, axis=0, keepdims=True) / (n + 1e-8)
    return np.mean((h * np.tanh(attn)) ** 2)

def run_experiment():
    X = generate_sequential(n_steps=30, n_samples=300)
    
    standard_losses = []
    causal_losses = []
    
    for run in range(5):
        np.random.seed(42 + run)
        standard_losses.append(standard_attention(X))
        causal_losses.append(causal_attention(X))
    
    standard_mse = np.mean(standard_losses)
    causal_mse = np.mean(causal_losses)
    improvement = (standard_mse - causal_mse) / standard_mse * 100
    
    print(f"Standard={standard_mse:.4f}, Causal={causal_mse:.4f}, Δ={improvement:+.1f}%")
    status = "SUPPORTED" if improvement > 5 else "REFUTED"
    print(f"Status: {status}")
    return {'standard': standard_mse, 'causal': causal_mse, 'improvement': improvement}, status, improvement

if __name__ == "__main__":
    results, status, improvement = run_experiment()