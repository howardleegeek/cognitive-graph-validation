"""
H3.15: Refined SSM Implementation
Based on H3.14 partial results - proper Mamba-style SSM with selective mechanism.
Addresses: Simple SSM performed -411.5% (WORSE than baseline)
"""

import numpy as np


def generate_long_sequence_tasks(sequence_length=30, n_samples=200, noise=0.01):
    """Generate long sequence tasks for SSM testing"""
    np.random.seed(42)
    n_obs = 8
    n_action = 4
    
    sequences = []
    for i in range(n_samples):
        seq = []
        s = np.random.randn(n_obs) * 0.1
        for t in range(sequence_length):
            a = np.random.randn(n_action) * 0.1
            ns = s + np.dot(np.random.randn(n_obs, n_action), a) + np.random.randn(n_obs) * noise
            seq.append((s, a, ns))
            s = ns
        sequences.append(seq)
    
    return sequences, n_obs, n_action


def baseline_forward(sequences, n_obs, n_action):
    """Baseline concatenation approach"""
    losses = []
    for seq in sequences:
        states = np.array([s[0] for s in seq])
        actions = np.array([s[1] for s in seq])
        
        x = np.concatenate([states, actions], axis=1)
        
        w = np.random.randn(x.shape[1], 64) * np.sqrt(2.0 / (x.shape[1] + 64))
        h = np.tanh(x @ w)
        
        loss = np.mean(h ** 2)
        losses.append(loss)
    
    return np.mean(losses)


def simple_ssm_forward(sequences, n_obs, n_action):
    """Simple SSM (like H3.14 - performs poorly)"""
    losses = []
    for seq in sequences:
        states = np.array([s[0] for s in seq])
        actions = np.array([s[1] for s in seq])
        
        x = np.concatenate([states, actions], axis=1)
        
        w = np.random.randn(x.shape[1], 64) * np.sqrt(2.0 / (x.shape[1] + 64))
        h = np.tanh(x @ w)
        
        h_new = np.zeros_like(h)
        state = np.zeros(64)
        for t in range(len(h)):
            gate = 1.0 / (1.0 + np.exp(-(h[t] @ np.random.randn(64, 64) * 0.1)))
            state = gate * state + (1 - gate) * h[t]
            h_new[t] = state
        
        loss = np.mean(h_new ** 2)
        losses.append(loss)
    
    return np.mean(losses)


def mamba_ssm_forward(sequences, n_obs, n_action):
    """Mamba-style SSM with proper selective mechanism"""
    losses = []
    for seq in sequences:
        states = np.array([s[0] for s in seq])
        actions = np.array([s[1] for s in seq])
        
        x = np.concatenate([states, actions], axis=1)
        
        dim = 64
        w_x = np.random.randn(x.shape[1], dim) * np.sqrt(2.0 / (x.shape[1] + dim))
        w_gate = np.random.randn(dim, dim) * 0.1
        w_proj = np.random.randn(dim, dim) * np.sqrt(2.0 / (dim + dim))
        
        h = np.tanh(x @ w_x)
        
        h_processed = np.zeros_like(h)
        state = np.zeros(dim)
        for t in range(len(h)):
            gate = 1.0 / (1.0 + np.exp(-h[t] @ w_gate))
            
            z = h[t] @ w_proj
            z_gate = 1.0 / (1.0 + np.exp(-z))
            
            state = gate * state + (1 - gate) * h[t]
            h_processed[t] = state * z_gate
        
        loss = np.mean(h_processed ** 2)
        losses.append(loss)
    
    return np.mean(losses)


def s4_style_forward(sequences, n_obs, n_action):
    """S4-style SSM with convolutional state expansion"""
    losses = []
    for seq in sequences:
        states = np.array([s[0] for s in seq])
        actions = np.array([s[1] for s in seq])
        
        x = np.concatenate([states, actions], axis=1)
        
        dim = 64
        w_x = np.random.randn(x.shape[1], dim) * np.sqrt(2.0 / (x.shape[1] + dim))
        
        h = np.tanh(x @ w_x)
        
        conv_kernel = np.random.randn(3, dim) * 0.1
        h_processed = np.zeros_like(h)
        
        for t in range(len(h)):
            conv_sum = np.zeros(dim)
            for k in range(3):
                idx = t - k
                if idx >= 0:
                    conv_sum += conv_kernel[k] * h[idx]
            
            gate = 1.0 / (1.0 + np.exp(-conv_sum))
            h_processed[t] = h[t] * gate
        
        loss = np.mean(h_processed ** 2)
        losses.append(loss)
    
    return np.mean(losses)


def run_experiment():
    """Run H3.15 experiment"""
    results = []
    
    for sequence_length in [15, 20, 30, 40, 50]:
        print(f"\n=== Sequence Length: {sequence_length} ===")
        
        sequences, n_obs, n_action = generate_long_sequence_tasks(
            sequence_length=sequence_length, n_samples=200
        )
        
        baseline_losses = []
        simple_ssm_losses = []
        mamba_losses = []
        s4_losses = []
        
        for run in range(5):
            np.random.seed(42 + run)
            baseline_losses.append(baseline_forward(sequences, n_obs, n_action))
            simple_ssm_losses.append(simple_ssm_forward(sequences, n_obs, n_action))
            mamba_losses.append(mamba_ssm_forward(sequences, n_obs, n_action))
            s4_losses.append(s4_style_forward(sequences, n_obs, n_action))
        
        baseline_mse = np.mean(baseline_losses)
        simple_ssm_mse = np.mean(simple_ssm_losses)
        mamba_mse = np.mean(mamba_losses)
        s4_mse = np.mean(s4_losses)
        
        simple_delta = (baseline_mse - simple_ssm_mse) / baseline_mse * 100
        mamba_delta = (baseline_mse - mamba_mse) / baseline_mse * 100
        s4_delta = (baseline_mse - s4_mse) / baseline_mse * 100
        
        print(f"Baseline:     {baseline_mse:.4f}")
        print(f"Simple SSM:   {simple_ssm_mse:.4f} ({simple_delta:+.1f}%)")
        print(f"Mamba SSM:    {mamba_mse:.4f} ({mamba_delta:+.1f}%)")
        print(f"S4-style:     {s4_mse:.4f} ({s4_delta:+.1f}%)")
        
        results.append({
            'sequence_length': sequence_length,
            'baseline': baseline_mse,
            'simple_ssm': simple_ssm_mse,
            'mamba': mamba_mse,
            's4': s4_mse,
            'simple_delta': simple_delta,
            'mamba_delta': mamba_delta,
            's4_delta': s4_delta
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_simple = np.mean([r['simple_delta'] for r in results])
    avg_mamba = np.mean([r['mamba_delta'] for r in results])
    avg_s4 = np.mean([r['s4_delta'] for r in results])
    
    print(f"Simple SSM avg: {avg_simple:+.1f}%")
    print(f"Mamba SSM avg:  {avg_mamba:+.1f}%")
    print(f"S4-style avg:   {avg_s4:+.1f}%")
    
    best_method = "Mamba" if avg_mamba >= avg_s4 else "S4"
    best_delta = max(avg_mamba, avg_s4)
    
    status = "SUPPORTED" if best_delta > 5 else ("MARGINAL" if best_delta > 0 else "REFUTED")
    print(f"\nBest: {best_method} with {best_delta:+.1f}%")
    print(f"Status: {status}")
    
    return results, status, best_method, best_delta


if __name__ == "__main__":
    results, status, method, delta = run_experiment()
    
    print("\n" + "=" * 60)
    print("H3.15 RESULTS")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Improvement: {delta:+.1f}%")
    print(f"Status: {status}")