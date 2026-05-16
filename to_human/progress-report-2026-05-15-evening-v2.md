# Cognitive Graph Validation - Progress Report
## May 15, 2026 - Evening Cycle

### Executive Summary

Research continues to refine understanding of the cognitive graph architecture's performance characteristics. Two new experiments completed:

- **H1.366**: Transition zone (25-35 steps) - ✅ SUPPORTED (+25.0% CG, +29.2% CG+Attn)
- **H3.366**: Boundary zone (40-50 steps) - ⚠️ PARTIAL (concat still wins +18.2%)

### Key Findings This Cycle

#### 1. H1.366: Transition Zone (25-35 Steps) - SUPPORTED

| Steps | CG Improvement | CG+Attn Improvement |
|-------|-----------------|---------------------|
| 25 | +15.3% | +14.9% |
| 28 | +5.0% | +33.0% |
| 30 | +40.3% | +25.4% |
| 32 | +30.8% | +42.4% |
| 35 | +33.6% | +30.3% |

**Average: CG +25.0%, CG+Attn +29.2%**

The transition zone (25-35 steps) remains in the sweet spot! CG+Attention combination provides additional benefit (+29.2% vs +25.0%).

#### 2. H3.366: Boundary Zone (40-50 Steps) - PARTIAL

| Steps | Concat | Attention | Winner |
|-------|--------|-----------|--------|
| 40 | +26.5% | +8.2% | Concat |
| 42 | +13.1% | +21.5% | Attention |
| 45 | +25.4% | +20.0% | Concat |
| 48 | +8.4% | -4.7% | Concat |
| 50 | +17.4% | +32.0% | Attention |

**Average: Concat +18.2%, Attention +15.4%**

Concatenation still wins at 40-50 steps, but attention is competitive. This is the boundary zone where advantage starts to diminish.

### Updated Complexity Sweet Spot Map

| Complexity Range | CG Performance | Status |
|-----------------|---------------|--------|
| 10-15 steps | -23.6% | ❌ Too simple |
| 20-40 steps | +17.4% to +25.0% | ✅ Sweet spot |
| 25-35 steps | +25.0% | ✅ Transition zone |
| 40-50 steps | +18.2% (concat) | ⚠️ Boundary |
| 50-70 steps | -40.8% | ❌ Too complex |

### Research Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot |
| H1.365 | ✅ SUPPORTED | Sweet spot 20-40 steps (+17.4%) |
| H1.366 | ✅ SUPPORTED | Transition zone 25-35 steps (+25.0%) |
| H3.365 | ✅ SUPPORTED | Attention+Goal on 20-40 steps (+5.8%) |
| H3.366 | ⚠️ PARTIAL | Boundary zone 40-50 steps |

### Next Steps

1. **H1.367**: Test CG with attention on 25-35 steps (combined approach)
2. **H3.367**: Test attention on 45-55 steps (extended boundary)
3. **H1.368**: Test CG on 18-22 steps (lower sweet spot boundary)

### Running Experiments

Total experiments run: **135+**
- Supported: **20+**
- Refuted: **11+**
- Inconclusive/Partial: **5+**

---
*Generated: May 15, 2026 18:05 UTC*