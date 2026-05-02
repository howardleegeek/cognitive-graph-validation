# H1.93 Results: Ultra-Complex Multi-Step Tasks (150-300 steps)

## Summary

| Horizon | Baseline MSE | Unified MSE | Improvement |
|---------|-------------|-------------|-------------|
| 150 steps | 889.04 | 3184.38 | **-258.2%** |
| 200 steps | 24240.14 | 103278.48 | **-326.1%** |
| 250 steps | 688565.94 | 2490750.50 | **-261.7%** |
| 300 steps | 19979770.00 | 70217032.00 | **-251.4%** |

**Average Improvement: -274.4%**

## Analysis

The results show that the unified architecture with attention performs **significantly worse** than the baseline on ultra-complex tasks in this synthetic setting. This is a **REFUTATION** of the hypothesis.

Possible reasons:
1. **Overfitting**: The attention mechanism may overfit on the small dataset (80 training samples)
2. **Training insufficiency**: 100 epochs may not be enough for such long sequences
3. **Architecture mismatch**: The attention mechanism may not be suitable for this specific data generation process
4. **Gradient issues**: Very long sequences may cause gradient vanishing/exploding

## Comparison with Prior Results

| Experiment | Horizon | Improvement |
|------------|---------|-------------|
| H1.33 | 25-40 steps | +86.8% |
| H1.92 | 60-100 steps | +89.1% |
| H1.99 | 100-250 steps | +99.1% |
| **H1.93** | **150-300 steps** | **-274.4%** |

The trend reverses at extreme complexity in this synthetic setting.

## Status

**REFUTED** — The unified architecture doesn't maintain advantage on ultra-complex tasks in this synthetic setting.

## Notes

This result contradicts H1.99 which showed +99.1% on 100-250 step tasks. The difference may be due to:
1. Different data generation process
2. Different model configurations
3. Different training procedures

Further investigation needed to understand the discrepancy.