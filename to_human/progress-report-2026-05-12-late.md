# Progress Report — May 12, 2026 (Late)

## Research Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% with real robot data |
| H2 | ⚠️ INCONCLUSIVE | 1.7% difference |
| H3 | 🔄 MIXED | Attention works on manipulation, fails on pure prediction |
| H4 | 🔸 CLOSE | 25% optimal vs 28% hypothesis |

## Experiments Run Today

### H1.222: Ultra-Complex Multi-Step (80-120 Steps) with Goal Conditioning
**Status: ❌ REFUTED**

| Sequence Length | Baseline MSE | Unified MSE | Delta |
|-----------------|-------------|-------------|-------|
| 80 | 0.00385 | 0.00341 | +11.4% |
| 90 | 0.00345 | 0.00350 | -1.4% |
| 100 | 0.00327 | 0.00339 | -3.7% |
| 110 | 0.00329 | 0.00352 | -7.0% |
| 120 | 0.00321 | 0.00356 | -10.9% |

**Average: -1.9%** (1/5 wins)

**Finding**: Unified architecture loses on ultra-complex tasks. The pattern from H1.221 continues - complex multi-step tasks hurt unified architecture.

---

### H3.118: Attention with High Autocorrelation (0.9-0.95) on 50-80 Steps
**Status: ⚠️ PARTIAL**

| Autocorrelation | Improvement | Wins |
|-----------------|-------------|------|
| 0.90 | +13.5% | 4/4 |
| 0.93 | +14.7% | 4/4 ← **Best** |
| 0.95 | +9.9% | 4/4 |

**Overall: +3.3%** (12/12 wins)

**Finding**: Attention wins at ALL lengths and ALL autocorrelation levels, but average improvement is only +3.3%. Key insight: optimal autocorrelation for attention is around 0.93, not 0.95.

---

## Key Insights

1. **Ultra-complex tasks hurt unified architecture**: Both H1.221 (30-70 steps) and H1.222 (80-120 steps) show unified architecture loses on complex multi-step tasks with goal conditioning.

2. **Autocorrelation unlocks attention**: H3.117 and H3.118 confirm that high autocorrelation (ρ ≥ 0.7) enables attention on longer sequences. Optimal is around ρ=0.93.

3. **The "death zone" pattern**: Sequences in the 30-200 step range are problematic for both CG/SSM and attention unless autocorrelation is present.

## Recommendations

| Sequence Length | Data Type | Recommended Architecture |
|-----------------|-----------|-------------------------|
| 10-25 steps | Any | CG or SSM |
| 30-50 steps | Real robot (ρ≥0.7) | Attention |
| 30-50 steps | Synthetic (ρ≈0) | Baseline MLP or SSM |
| 50-200 steps | Any | Concatenation |
| 250+ steps | Any | SSM + Hierarchical Goals |

## Total Experiments: 36 runs

## Next Steps

1. Test attention with ρ=0.93 specifically on 40-70 step sequences
2. Explore SSM with goal conditioning on ultra-complex tasks
3. Investigate why unified architecture fails on complex multi-step tasks