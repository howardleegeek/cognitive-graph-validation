# Progress Report — Cycle 98 (May 4, 2026)

## Research Summary

**Overall Progress**: 98 experiments completed  
**Cycle Status**: Active

## Key Results

### Recent Experiments (Cycle 98)

| Hypothesis | Status | Improvement | Notes |
|------------|--------|--------------|-------|
| H3.37: Stochastic Dynamics | ⚠️ Inconclusive | +0% | Standard attention tied with concat |
| H3.38: Robust Attention | ✅ SUPPORTED | +99.9% | Variance-weighted wins |
| **H3.39: Query-Key Decay** | **✅ SUPPORTED** | **+9.8%** | decay=0.7 optimal |

## Findings Summary

### H1 Family (Unified Architecture)
- Strong support: +25.6% on real robot data
- Attention mechanisms: +99% on complex/long-horizon tasks
- Dimension scaling: 4096 optimal w/o reg, 32k+ with α≥0.1

### H2 Family (Graph Structure)
- Temporal reasoning: +56-75% improvement
- Compositional: +50.4% with increasing benefit

### H3 Family (Attention Mechanisms)
- Simple tasks: Concatenation wins
- Complex (16+ steps): Attention wins (+99%)
- Stochastic dynamics: Query-key decay (+9.8%) NEW ✓

### H3.39 Detail

Query-key decay attention on stochastic dynamics:
- No decay: 5.58 MSE
- Decay=0.9: +7.1%
- Decay=0.8: +6.6%
- **Decay=0.7: +9.8%** ← Best

This extends H3.38's robust attention finding to stochastic domains.

## Research Status

| Status | Count |
|--------|-------|
| SUPPORTED | 30+ |
| INCONCLUSIVE | 2 |
| REFUTED | 15+ |

## Next Steps

1. Test decay on real robot stochastic dynamics
2. Explore combined (decay + variance-weighted)
3. Paper draft preparation

## Active Directions

- H3.39: Decay attention on stochastic → ✅ +9.8%
- H1.50 series: Attention on real robot → ✅ +99%
- H3.38: Robust attention → ✅ +99.9%

---

*Research continues. Next experiment: H3.40 combination test.*