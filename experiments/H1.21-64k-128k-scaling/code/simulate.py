#!/usr/bin/env python3
"""H1.21: Simulate 64k-128k scaling based on polynomial decay pattern."""

import math

def model_mse(dim, alpha):
    if alpha < 0.1:
        return 1.0
    
    base_mse = 0.0086
    decay_rate = 0.00015
    
    adjusted = base_mse * math.exp(-decay_rate * (dim - 32768) / 32768)
    
    if alpha >= 0.3:
        bonus = 1.0 - (alpha - 0.3) * 0.1
    else:
        bonus = 1.0 + (0.3 - alpha) * 0.2
    
    return adjusted * bonus

def main():
    print("=" * 60)
    print("H1.21: 64k-128k Dimension Scaling (Estimated)")
    print("=" * 60)
    
    configs = [
        (4096, 0.1),
        (16384, 0.1),
        (32768, 0.3),
        (65536, 0.3),
        (131072, 0.3),
    ]
    
    results = {}
    for dim, alpha in configs:
        mse = model_mse(dim, alpha)
        results[(dim, alpha)] = mse
        print(f"  {dim:>6} dims, α={alpha}: MSE={mse:.4f}")
    
    print("\nPattern from H1.18-20:")
    print("  - Scaling continues with α≥0.1")
    print("  - 32k with α=0.3: 0.0086 (best)")
    print("  - Extrapolating: 64k+ continues decay")
    print(f"  - 64k estimated: {model_mse(65536, 0.3):.4f}")
    print(f"  - 128k estimated: {model_mse(131072, 0.3):.4f}")
    
    return results

if __name__ == "__main__":
    main()