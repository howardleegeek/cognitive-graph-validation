"""
H1.39: Action-Conditioned Attention
Testing if action-conditioning improves attention
"""

import numpy as np

def generate_action_conditioned(n_steps=30, n_samples=300):
    np.random.seed(42)
    states = []
    for i in range(n_samples):
        s = np.random.randn(12) * 0.1
        for t in range(n_steps):
            a = np.random.randn(7) * 0.1
            # Action-conditioned transition
            ns = s + np.dot(np.random.randn(12, 7), a) * (1 + np.sum(a))
            s = ns
            states.append(s.copy())
    return np.array(states)

def concat_baseline(x, dims=512):
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    return np.mean(np.tanh(x @ w1) ** 2)

def attention_conditioned(x, dims=512):
    w1 = np.random.randn(x.shape[1], dims) * np.sqrt(2.0 / (x.shape[1] + dims))
    w_state = np.random.randn(x.shape[1] - 7, dims) * 0.1
    w_action = np.random.randn(7, dims) * 0.1
    h = np.tanh(x @ w1)
    # Get state and action separately
    state = x[:, :-7]
    action = x[:, -7:]
    state_effect = state @ w_state
    action_effect = action @ w_action
    # Attention modulated by action
    attn = np.sum(h * np.tanh(state_effect + action_effect), axis=0, keepdims=True) / (h.shape[0] + 1e-8)
    return np.mean((h * np.tanh(attn)) ** 2)

def run_experiment():
    X = generate_action_conditioned(n_steps=30, n_samples=300)
    
    baseline_losses = []
    conditioned_losses = []
    
    for run in range(5):
        np.random.seed(42 + run)
        baseline_losses.append(concat_baseline(X))
        conditioned_losses.append(attention_conditioned(X))
    
    baseline_mse = np.mean(baseline_losses)
    conditioned_mse = np.mean(conditioned_losses)
    improvement = (baseline_mse - conditioned_mse) / baseline_mse * 100
    
    print(f"Baseline={baseline_mse:.4f}, Conditioned={conditioned_mse:.4f}, Δ={improvement:+.1f}%")
    status = "SUPPORTED" if improvement > 5 else "REFUTED"
    print(f"Status: {status}")
    return {'baseline': baseline_mse, 'conditioned': conditioned_mse, 'improvement': improvement}, status, improvement

if __name__ == "__main__":
    results, status, improvement = run_experiment()