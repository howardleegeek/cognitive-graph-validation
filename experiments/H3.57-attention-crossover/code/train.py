"""
H3.57: Attention Crossover Test - Simple
"""
import numpy as np

np.random.seed(42)
print("=== H3.57: Attention Crossover ===")

results = []
for seq_len in [10, 15, 20, 25, 30, 40, 50]:
    # Simple trajectory
    pos = np.linspace(0, 1, seq_len)
    noise = np.linspace(1.0, 0.1, seq_len) * 0.1
    states = pos + np.random.randn(seq_len) * noise
    
    # Attention: recent weighted more (matching seq_len)
    w = np.exp(np.arange(seq_len) * 0.1)
    w = w / w.sum()
    attn = np.dot(w, states)  # full weighted average
    
    # Concat: simple average
    concat = np.mean(states)
    
    # Target
    true = pos[-1]
    
    err_a = (attn - true) ** 2
    err_c = (concat - true) ** 2
    imp = (1 - err_a / err_c) * 100
    
    results.append((seq_len, err_a, err_c, imp))
    print(f"Len {seq_len:2d}: Attn={err_a:.4f}, Concat={err_c:.4f}, Δ={imp:+.1f}%")

avg_short = np.mean([r[3] for r in results if r[0] <= 20])
avg_long = np.mean([r[3] for r in results if r[0] >= 30])
print(f"\nShort (10-20): {avg_short:+.1f}%")
print(f"Long (30-50): {avg_long:+.1f}%")
print(f"Status: {'SUPPORTED' if avg_long > 5 else 'INCONCLUSIVE' if avg_long > -5 else 'REFUTED'}")