# Research Progress Report - May 13, 2026 (Night)

## Summary

Research continues on Cognitive Graph architecture validation. Three new experiments completed tonight.

## Current Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot data |
| H1.237 | ✅ SUPPORTED | +88.9% on 15-25 steps |
| H1.238 | ❌ REFUTED | -0.1% on 30-40 steps (ceiling reached) |
| H1.239 | ❌ REFUTED | +1.4% on 10-20 steps (inconsistent) |
| H3.140 | ✅ SUPPORTED | +91.9% on 20-30 steps with rho=0.9 |
| H3.141 | ❌ REFUTED | -0.1% on 25-35 steps (doesn't extend) |

## Experiments Completed Tonight

### H1.238: Ultra-Complex Multi-Step (30-40 Steps)
- **Result**: REFUTED (-0.1% avg)
- **Finding**: Advantage completely diminishes at 30-40 steps
- **Conclusion**: Complexity ceiling around 25-30 steps

### H1.239: Sweet Spot Verification (10-20 Steps)
- **Result**: REFUTED (+1.4% avg)
- **Finding**: Much lower than H1.237 (+88.9%)
- **Conclusion**: Results highly inconsistent across experiments

### H3.141: Attention on 25-35 Steps with rho=0.9
- **Result**: REFUTED (-0.1% avg)
- **Finding**: Doesn't extend H3.140's +91.9% success
- **Conclusion**: Attention advantage doesn't scale beyond 20-30 steps

## Key Insights

1. **Complexity Ceiling**: Unified+Attention+Reg works best at 15-25 steps, diminishes above 30 steps
2. **High Variance**: Results inconsistent across experiments (H1.237: +88.9% vs H1.239: +1.4%)
3. **Attention Boundary**: Confirmed at ~30 steps even with optimal rho=0.9

## Total Experiments: 70 runs

## Next Steps

1. Investigate high variance in results - need more robust experimental design
2. Test different random seeds to verify consistency
3. Consider alternative architectures for >30 step sequences