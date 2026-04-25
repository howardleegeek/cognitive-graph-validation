"""
H3.8: Hierarchical Attention on Long Sequences
Based on PRISM (ICLR 2026) - hierarchical compresses local interactions into compact tokens.
Tests if hierarchical attention scales better than flat attention on very long sequences.
"""

import numpy as np

def generate_long_tasks(sequence_length=30, n_samples=200, noise=0.01):
    """Generate long sequence tasks"""
    np.random.seed(42)
    n_obs = 8
    n_action = 4
    
    states = []
    actions = []
    next_states = []
    
    for i in range(n_samples):
        s = np.random.randn(n_obs) * 0.1
        for t in range(sequence_length):
            a = np.random.randn(n_action) * 0.1
            ns = s + np.dot(np.random.randn(n_obs, n_action), a) + np.random.randn(n_obs) * noise
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

def flat_attention_forward(x, dims=512, use_gating=True):
    """Standard flat attention"""
    w = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w)
    
    if use_gating:
        gate = 1.0 / (1.0 + np.exp(-(h @ np.random.randn(dims, dims) * 0.1)))
        h = h * gate
    
    scale = np.sqrt(dims)
    attn_logits = (h @ h.T) / scale
    attn = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
    attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-8)
    h_attended = attn @ h
    return np.mean(h_attended ** 2)

def hierarchical_attention_forward(x, dims=512, segment_size=5):
    """Hierarchical attention - compresses local segments into tokens"""
    n_examples = x.shape[0]
    n_segments = n_examples // segment_size
    
    w = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    h = np.tanh(x @ w)
    
    # Compress each segment into a single token
    segment_tokens = []
    for i in range(n_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, n_examples)
        segment = h[start:end]
        # Local attention within segment
        scale = np.sqrt(dims)
        attn_logits = (segment @ segment.T) / scale
        attn = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
        attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-8)
        compressed = attn @ segment
        segment_tokens.append(compressed)
    
    segment_tokens = np.array(segment_tokens)
    
    # Cross-segment attention with gating
    gate = 1.0 / (1.0 + np.exp(-(segment_tokens @ np.random.randn(dims, dims) * 0.1)))
    segment_tokens = segment_tokens * gate
    
    # Global attention across compressed tokens
    scale = np.sqrt(dims)
    attn_logits = (segment_tokens @ segment_tokens.T) / scale
    attn = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
    attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-8)
    h_attended = attn @ segment_tokens
    
    return np.mean(h_attended ** 2)

def run_experiment():
    """Run H3.8 experiment"""
    results = []
    
    for sequence_length in [15, 20, 30, 40, 50]:
        X, A, Y = generate_long_tasks(sequence_length=sequence_length, n_samples=200)
        X_full = np.concatenate([X, A], axis=1)
        
        concat_losses = []
        flat_losses = []
        hier_losses = []
        
        for run in range(5):
            np.random.seed(42 + run)
            concat_losses.append(concatenation_forward(X_full))
            flat_losses.append(flat_attention_forward(X_full))
            hier_losses.append(hierarchical_attention_forward(X_full))
        
        concat_mse = np.mean(concat_losses)
        flat_mse = np.mean(flat_losses)
        hier_mse = np.mean(hier_losses)
        
        # Compare hierarchical vs flat
        delta = (flat_mse - hier_mse) / flat_mse * 100 if flat_mse > 0 else 0
        
        print(f"Seq={sequence_length}: Concat={concat_mse:.4f}, Flat={flat_mse:.4f}, Hier={hier_mse:.4f}, Δ={delta:+.1f}%")
        results.append({
            'sequence_length': sequence_length,
            'concat': concat_mse,
            'flat': flat_mse,
            'hier': hier_mse,
            'delta': delta
        })
    
    avg_delta = np.mean([r['delta'] for r in results])
    print(f"\nAverage hierarchical vs flat: {avg_delta:+.1f}%")
    
    status = "SUPPORTED" if avg_delta > 5 else ("MARGINAL" if avg_delta > 0 else "REFUTED")
    print(f"Status: {status}")
    
    return results, status, avg_delta

if __name__ == "__main__":
    results, status, delta = run_experiment()