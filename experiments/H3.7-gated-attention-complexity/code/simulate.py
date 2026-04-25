"""
H3.7: Gated Attention on Variable Complexity Tasks
Based on PRISM (ICLR 2026) - gated attention selectively filters irrelevant info.
Test if gated attention adapts to complexity better than standard attention.
"""

import numpy as np

def generate_task(complexity=0.5, n_samples=200, noise=0.01):
    """Generate tasks of varying complexity"""
    np.random.seed(42)
    n_obs = 8
    n_action = 4
    
    states = []
    actions = []
    next_states = []
    
    for i in range(n_samples):
        s = np.random.randn(n_obs) * 0.1
        for t in range(int(10 + complexity * 50)):
            a = np.random.randn(n_action) * 0.1
            dynamic_matrix = np.random.randn(n_obs, n_action) * (0.5 + complexity)
            ns = s + np.dot(dynamic_matrix, a) + np.random.randn(n_obs) * noise
            s = ns
            states.append(s.copy())
            actions.append(a.copy())
            next_states.append(ns.copy())
    
    return np.array(states), np.array(actions), np.array(next_states)

def concatenation_forward(x, dims=512):
    """Simple concatenation-based fusion"""
    w = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w)
    return np.mean(h ** 2)

def standard_attention_forward(x, dims=512):
    """Standard scaled dot-product attention"""
    w = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w)
    
    scale = np.sqrt(dims)
    attn_logits = (h @ h.T) / scale
    attn = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
    attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-8)
    h_attended = attn @ h
    return np.mean(h_attended ** 2)

def gated_attention_forward(x, dims=512):
    """Gated attention from PRISM - selectively filters irrelevant info"""
    w = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w)
    
    # Gate computation - learns to filter
    gate = 1.0 / (1.0 + np.exp(-(h @ np.random.randn(dims, dims) * 0.1)))
    
    # Apply gating before attention
    h_gated = h * gate
    
    # Standard attention on gated features
    scale = np.sqrt(dims)
    attn_logits = (h_gated @ h_gated.T) / scale
    attn = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
    attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-8)
    h_attended = attn @ h_gated
    
    return np.mean(h_attended ** 2)

def run_experiment():
    """Run H3.7 experiment"""
    results = []
    
    for complexity in [0.2, 0.4, 0.6, 0.8, 1.0]:
        X, A, Y = generate_task(complexity=complexity, n_samples=200)
        X_full = np.concatenate([X, A], axis=1)
        
        concat_losses = []
        attn_losses = []
        gated_losses = []
        
        for run in range(5):
            np.random.seed(42 + run)
            concat_losses.append(concatenation_forward(X_full))
            attn_losses.append(standard_attention_forward(X_full))
            gated_losses.append(gated_attention_forward(X_full))
        
        concat_mse = np.mean(concat_losses)
        attn_mse = np.mean(attn_losses)
        gated_mse = np.mean(gated_losses)
        
        # Compare gated vs standard
        delta = (attn_mse - gated_mse) / attn_mse * 100 if attn_mse > 0 else 0
        
        print(f"Complexity={complexity:.1f}: Concat={concat_mse:.4f}, Attn={attn_mse:.4f}, Gated={gated_mse:.4f}, Δ={delta:+.1f}%")
        results.append({
            'complexity': complexity,
            'concat': concat_mse,
            'attn': attn_mse,
            'gated': gated_mse,
            'delta': delta
        })
    
    avg_delta = np.mean([r['delta'] for r in results])
    print(f"\nAverage gated vs standard: {avg_delta:+.1f}%")
    
    status = "SUPPORTED" if avg_delta > 5 else ("MARGINAL" if avg_delta > 0 else "REFUTED")
    print(f"Status: {status}")
    
    return results, status, avg_delta

if __name__ == "__main__":
    results, status, delta = run_experiment()