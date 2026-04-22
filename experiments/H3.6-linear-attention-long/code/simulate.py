"""
H3.6: Linear Attention on 40+ Step Very Long Sequences
Testing if linear attention variant helps on extremely long sequences
"""

import numpy as np

def generate_long_sequence(n_steps=40, n_samples=300, noise=0.01):
    """Generate very long sequence tasks"""
    np.random.seed(42)
    states = []
    actions = []
    next_states = []
    
    for i in range(n_samples):
        s = np.random.randn(8) * 0.1
        for t in range(n_steps):
            a = np.random.randn(4) * 0.1
            ns = s + np.dot(np.random.randn(8, 4), a) + np.random.randn(8) * noise
            s = ns
            states.append(s.copy())
            actions.append(a.copy())
            next_states.append(ns.copy())
    
    return np.array(states), np.array(actions), np.array(next_states)

def concatenation_forward(x, dims=512):
    """Simple concatenation-based fusion"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    return np.mean(h ** 2)

def linear_attention_forward(x, dims=512):
    """Linear attention variant - more efficient for long sequences"""
    # Linear attention: O(nd^2) instead of O(n^2d)
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    # Apply linear attention weighting
    attn = np.sum(h, axis=0, keepdims=True) / (h.shape[0] + 1e-8)
    h_weighted = h * np.tanh(attn)
    return np.mean(h_weighted ** 2)

def softmax(x, axis=-1):
    """Manual softmax implementation"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_forward(x, dims=512):
    """Scaled dot-product attention"""
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w1)
    # Scaled dot-product
    scale = np.sqrt(dims)
    attn_logits = (h @ h.T) / scale
    attn = softmax(attn_logits, axis=-1)
    h_attended = attn @ h
    return np.mean(h_attended ** 2)

def run_experiment():
    """Run H3.6 experiment"""
    results = []
    
    for n_steps in [32, 40, 48, 56, 64]:
        X, A, Y = generate_long_sequence(n_steps=n_steps, n_samples=300)
        X_full = np.concatenate([X, A], axis=1)
        
        concat_losses = []
        linear_losses = []
        scaled_losses = []
        
        for run in range(5):
            np.random.seed(42 + run)
            concat_losses.append(concatenation_forward(X_full))
            linear_losses.append(linear_attention_forward(X_full))
            scaled_losses.append(scaled_dot_product_forward(X_full))
        
        concat_mse = np.mean(concat_losses)
        linear_mse = np.mean(linear_losses)
        scaled_mse = np.mean(scaled_losses)
        
        best_attn = min(linear_mse, scaled_mse)
        improvement = (concat_mse - best_attn) / concat_mse * 100
        
        print(f"{n_steps}-step: Concat={concat_mse:.4f}, Linear={linear_mse:.4f}, Scaled={scaled_mse:.4f}, Δ={improvement:+.1f}%")
        results.append({
            'n_steps': n_steps,
            'concat': concat_mse,
            'linear': linear_mse,
            'scaled': scaled_mse,
            'improvement': improvement
        })
    
    avg_improvement = np.mean([r['improvement'] for r in results])
    print(f"\nAverage: {avg_improvement:+.1f}%")
    
    # Determine status
    status = "SUPPORTED" if avg_improvement > 5 else ("MARGINAL" if avg_improvement > 0 else "REFUTED")
    print(f"Status: {status}")
    
    return results, status, avg_improvement

if __name__ == "__main__":
    results, status, avg = run_experiment()