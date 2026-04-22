"""
H1.35: Dimension Scaling with Attention
Testing if attention benefits from larger dimensions
"""

import numpy as np

def generate_task(n_steps=30, n_samples=250, noise=0.01):
    """Generate long sequence task"""
    np.random.seed(42)
    states = []
    actions = []
    
    for i in range(n_samples):
        s = np.random.randn(12) * 0.1
        for t in range(n_steps):
            a = np.random.randn(7) * 0.1
            ns = s + np.dot(np.random.randn(12, 7), a) * 0.5 + np.random.randn(12) * noise
            s = ns
            states.append(s.copy())
            actions.append(a.copy())
    
    return np.array(states), np.array(actions)

def concat_forward(x, dims):
    """Concatenation baseline"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    return np.mean(h ** 2)

def attention_forward(x, dims):
    """Attention variant"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    attn = np.sum(h, axis=0, keepdims=True) / (h.shape[0] + 1e-8)
    h_weighted = h * np.tanh(attn)
    return np.mean(h_weighted ** 2)

def run_experiment():
    """Run H1.35 experiment"""
    results = []
    
    for dims in [512, 1024, 2048, 4096, 8192]:
        X, A = generate_task(n_steps=30, n_samples=250)
        X_full = np.concatenate([X, A], axis=1)
        
        concat_losses = []
        attn_losses = []
        
        for run in range(5):
            np.random.seed(42 + run)
            concat_losses.append(concat_forward(X_full, dims))
            attn_losses.append(attention_forward(X_full, dims))
        
        concat_mse = np.mean(concat_losses)
        attn_mse = np.mean(attn_losses)
        improvement = (concat_mse - attn_mse) / concat_mse * 100
        
        print(f"{dims:5d} dims: Concat={concat_mse:.4f}, Attn={attn_mse:.4f}, Δ={improvement:+.1f}%")
        results.append({
            'dims': dims,
            'concat': concat_mse,
            'attention': attn_mse,
            'improvement': improvement
        })
    
    avg = np.mean([r['improvement'] for r in results])
    print(f"\nAverage: {avg:+.1f}%")
    
    status = "SUPPORTED" if avg > 5 else "REFUTED"
    print(f"Status: {status}")
    
    return results, status, avg

if __name__ == "__main__":
    results, status, avg = run_experiment()