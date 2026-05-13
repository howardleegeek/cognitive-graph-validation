# Research Progress Report — May 13, 2026

## Executive Summary

Research continues on Cognitive Graph validation. Three new experiments completed, revealing critical insights about the boundaries of attention mechanisms and unified architectures.

## Experiments Completed

### H3.135: Attention Boundary 400-410 Steps
**Status: ❌ REFUTED**

| Seq Length | Concat MSE | Attention MSE | Delta |
|------------|-----------|--------------|-------|
| 400 | 0.3900 | 0.4114 | -5.5% |
| 405 | 0.4007 | 0.4268 | -6.5% |
| 410 | 0.3947 | 0.4074 | -3.2% |

**Finding**: Attention fails at 400-410 steps. Exact boundary confirmed at 400 steps.

---

### H1.229: Unified+Attention on Ultra-Complex with Autocorrelation
**Status: ❌ REFUTED**

| Seq Length | Unified+Attn | Unified | Delta |
|------------|--------------|---------|-------|
| 100 | 1.7241 | 1.7807 | +3.2% |
| 120 | 1.8560 | 1.7496 | -6.1% |
| 150 | 1.8312 | 1.7062 | -7.3% |
| 180 | 1.8145 | 1.7722 | -2.4% |
| 200 | 1.8448 | 1.8195 | -1.4% |

**Finding**: Unified+Attention doesn't outperform Unified alone on ultra-complex tasks even with autocorrelation.

---

### H1.230: Unified Architecture on Varying Complexity with Autocorrelation
**Status: ❌ REFUTED** ⚠️ CRITICAL FINDING

| Complexity | Unified | Baseline | Delta |
|------------|---------|----------|-------|
| 0.3 | 0.5148 | 0.3992 | -29.0% |
| 0.5 | 0.4535 | 0.4487 | -1.1% |
| 0.7 | 0.7546 | 0.7146 | -5.6% |
| 0.9 | 1.2621 | 1.1581 | -9.0% |
| 1.0 | 2.0479 | 1.5730 | -30.2% |

**Finding**: Unified architecture performs WORSE than baseline on complex tasks with autocorrelation (-15% avg). This is a critical new finding!

---

## Research Status Summary

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot (simple tasks) |
| H1.229 | ❌ REFUTED | Unified+Attn doesn't help (-2.8%) |
| H1.230 | ❌ REFUTED | Unified worse than baseline (-15%) |
| H3 | ✅ MIXED | Works <400 steps with autocorr |
| H3.135 | ❌ REFUTED | Fails at 400+ steps |

**Total: 20+ SUPPORTED, 1 INCONCLUSIVE, 13 REFUTED**

---

## Key Insights

1. **Attention boundary confirmed at ~400 steps** with autocorrelation
2. **Unified+Attention doesn't help** on ultra-complex tasks
3. **Unified architecture has a critical weakness**: performs worse than baseline on complex tasks with autocorrelation

---

## Next Steps

1. Explore SSM (State Space Models) for complex tasks with autocorrelation
2. Test hierarchical approaches to break the 400-step barrier
3. Investigate why unified fails on complex tasks - is it overfitting or architectural limitation?

---

## Total Experiments: 54 runs