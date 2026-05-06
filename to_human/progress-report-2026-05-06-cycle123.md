# Progress Report — May 6, 2026 (Cycle 123)

## Executive Summary

Research continues at high velocity. One new hypothesis tested today:

1. **H3.62**: Causal Attention for Continuous Control — **REFUTED (-45.0%)**

## Key Findings This Session

### H3.62: Causal Attention for Continuous Control
- **Result**: Causal (unidirectional) attention significantly hurts performance
- **Delta**: -45.0% average (Baseline 0.0154 vs Causal 0.0224)
- **Insight**: The unidirectional constraint is too restrictive for continuous control. Bidirectional attention is essential for robotic manipulation where past and future states inform current decisions.

## Research Status (May 6, 2026)

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 (Unified) | ✅ SUPPORTED | +25.6% real robot |
| H1.41-52 (Attention) | ✅ SUPPORTED | +99% universal |
| H1.51 (Task Types) | ✅ SUPPORTED | +99% across all tasks |
| H1.52 (Noise) | ✅ SUPPORTED | +98.5% robust |
| H2.x (Graph) | ✅ SUPPORTED | +56-75% temporal |
| H3 (Simple) | ❌ REFUTED | Concat wins |
| H3.34-57 (Long Seq) | ✅ SUPPORTED | Attention wins 20+ |
| H3.50 (SRH) | ✅ SUPPORTED | +45.5%, smaller better |
| H3.58 (Comb) | ✅ SUPPORTED | +17.2% +9.2% |
| H3.62 (Causal) | ❌ REFUTED | -45.0% |

**Total: 25+ SUPPORTED, 1 INCONCLUSIVE, 13 REFUTED**

## Architecture Recommendations

Based on all experiments:

1. **Use unified architecture** for same-dynamics tasks (+25.6%)
2. **Use bidirectional attention** for long-horizon (20+) tasks (+99%)
3. **Use graph structure** for temporal reasoning (+56-75%)
4. **Combine attention + invariant** for both temporal + transfer (+17%, +9%)
5. **Avoid causal attention** for continuous control (-45%)

## Next Steps for Paper

1. Write abstract and introduction
2. Prepare figures for key results
3. Draft methodology section
4. Complete experiments on edge cases

## Git Commit

- commit: H3.62 causal attention tested - REFUTED (-45.0%)

---
*Generated: May 6, 2026*