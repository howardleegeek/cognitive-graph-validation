"""
H1.124: Phase-Aware Attention - Simple Simulation

This experiment uses numpy to demonstrate phase-aware attention benefits.
"""
import numpy as np


def simulate_phase_attention():
    """Simulate attention with phase conditioning."""
    np.random.seed(42)
    
    print("=== H1.124: Phase-Aware Attention ===")
    print("Testing: Does phase conditioning improve temporal attention?")
    
    # Generate manipulation tasks with 4 phases
    seq_len = 20
    n_samples = 100
    
    # Create data with phase structure
    # Phase 0 (reach): robot moves toward object
    # Phase 1 (grasp): gripper closes
    # Phase 2 (move): transport object
    # Phase 3 (release): let go
    
    # Task representation: temporal features
    features = []
    phases = []
    targets = []
    
    for i in range(n_samples):
        traj = []
        ph = []
        tgt = []
        
        for step in range(seq_len):
            phase = step // 5  # 4 phases
            # Add phase-specific signal
            signal = np.sin(phase * np.pi / 2) * np.ones(10)
            traj.append(signal + np.random.randn(10) * 0.1)
            ph.append(phase)
            tgt.append(np.sin(phase * np.pi / 2))
        
        features.append(traj)
        phases.append(ph)
        targets.append(tgt)
    
    features = np.array(features)  # (100, 20, 10)
    targets = np.array(targets)  # (100, 20)
    
    # Train simple attention model
    print("\n--- Training Phase-Aware Attention ---")
    
    # Standard attention: weighted average
    def standard_attention(x):
        # Simple attention: recent timesteps weighted more
        weights = np.exp(np.arange(x.shape[1])[::-1] * 0.1)
        weights = weights / weights.sum()
        return np.einsum('btd,t->bd', x, weights)
    
    # Phase-aware: phase modulates weights
    def phase_attention(x, phases):
        # Weight by phase
        phase_weights = np.zeros(x.shape[1])
        for i, p in enumerate(phases):
            p_mod = 0.5 + p * 0.2  # Later phases weighted more
            phase_weights[i] = p_mod
        
        weights = phase_weights * np.exp(np.arange(x.shape[1])[::-1] * 0.1)
        weights = weights / weights.sum()
        return np.einsum('btd,t->bd', x, weights)
    
    # Simple concatenation baseline
    def concat_baseline(x):
        return np.mean(x, axis=1)
    
    # Evaluate
    pred_std = standard_attention(features)
    pred_phase = phase_attention(features, phases[0])
    pred_concat = concat_baseline(features)
    
    # True target: phase value
    true_phase = np.mean(targets, axis=1, keepdims=True)
    
    # MSE
    mse_std = np.mean((pred_std[:, 0] - true_phase[:, 0]) ** 2)
    mse_phase = np.mean((pred_phase[:, 0] - true_phase[:, 0]) ** 2)
    mse_concat = np.mean((pred_concat[:, 0] - true_phase[:, 0]) ** 2)
    
    print("\n=== Results ===")
    print(f"Standard Attention MSE: {mse_std:.4f}")
    print(f"Phase-Aware MSE: {mse_phase:.4f}")
    print(f"Concatenation MSE: {mse_concat:.4f}")
    
    # Improvement
    phase_vs_concat = (1 - mse_phase / mse_concat) * 100
    phase_vs_std = (1 - mse_phase / mse_std) * 100
    
    print(f"\nPhase vs Concatenation: {phase_vs_concat:+.1f}%")
    print(f"Phase vs Standard: {phase_vs_std:+.1f}%")
    
    # Summary
    print("\n=== Summary ===")
    if phase_vs_concat > 5:
        print("Status: SUPPORTED - Phase conditioning helps attention")
        return "supported", phase_vs_concat
    elif phase_vs_concat > -5:
        print("Status: INCONCLUSIVE - Marginal difference")
        return "inconclusive", phase_vs_concat
    else:
        print("Status: REFUTED - Phase conditioning doesn't help")
        return "refuted", phase_vs_concat


if __name__ == '__main__':
    status, improvement = simulate_phase_attention()
    
    print(f"\n=== H1.124 Final ===")
    print(f"Status: {status.upper()}")
    print(f"Improvement: {improvement:+.1f}%")