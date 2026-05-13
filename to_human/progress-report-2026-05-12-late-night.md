# Research Progress Report — May 12, 2026 (Late Night)

## Executive Summary

Extended attention validity to 185-195 steps, but discovered Unified+Attention combination fails on extreme complexity without autocorrelation.

## New Experiments Run (2 experiments, 50 total runs)

### ✅ H3.128: Attention Boundary at 185-195 Steps
- **Result**: +14.2% improvement, 6/6 wins
- **Status**: SUPPORTED
- **Key finding**: Attention extends to 185-195 step sequences with autocorrelation. Extends valid range beyond 180 steps.

### ❌ H1.228: Unified+Attention on Extreme Complex Tasks
- **Result**: -212.1% improvement, 0/3 wins
- **Status**: REFUTED
- **Key finding**: Unified+Attention combination performs WORSE than baseline on complex tasks without autocorrelation. The combination is fragile and requires specific conditions.

## Key Insights

### Attention Sequence Length Boundary
- **Previous limit**: 180 steps (H3.127)
- **New limit**: 195 steps (H3.128)
- **Fails at**: 200+ steps even with autocorrelation (H3.126)

### Unified+Attention Combination is Fragile
- **Works with autocorrelation** (H1.227): +13.4%
- **Fails without autocorrelation** (H1.228): -212.1%
- **Key insight**: The combination needs temporal structure to work

## Updated Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot data |
| H2 | ⚠️ INCONCLUSIVE | 1.7% difference |
| H3 | 🔄 MIXED | Works with autocorrelation |
| H3.128 | ✅ SUPPORTED | +14.2% at 185-195 steps |
| H1.228 | ❌ REFUTED | -212.1% on extreme complexity |
| H4 | 🔸 CLOSE | 25% optimal vs 28% hypothesis |

**Total: 20+ SUPPORTED, 1 INCONCLUSIVE, 12 REFUTED**

## Next Steps

1. Test attention at exactly 196-200 step boundary to find exact threshold
2. Investigate why Unified+Attention fails without autocorrelation
3. Explore hybrid approaches that combine strengths

## Files Updated
- `findings.md`: Added H3.128 and H1.228 results
- `research-state.yaml`: Updated with 2 new experiments (50 total runs)
- `experiments/H3.128-attention-boundary-185-195/`: New experiment
- `experiments/H1.228-unified-attention-extreme-complex/`: New experiment

---
*Generated: May 12, 2026 23:30 UTC*
*Total research time: 40+ hours*
*Next experiment: Test 196-200 step boundary*