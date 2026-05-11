# Research Progress Report - May 10, 2026 (Evening)

## Summary

**H3.91: SUPPORTED (+86.6%)** — Attention on 20-40 timesteps WITH task structure dramatically outperforms concatenation!

## Key Findings

### H3.91: Attention on 20+ Timesteps WITH Task Structure

| Sequence Length | Concat MSE | Attention MSE | Delta |
|-----------------|-----------|--------------|-------|
| 20 | 0.000727 | 0.000141 | **+80.6%** |
| 25 | 0.000853 | 0.000104 | **+87.8%** |
| 30 | 0.000784 | 0.000135 | **+82.8%** |
| 35 | 0.001016 | 0.000096 | **+90.6%** |
| 40 | 0.001245 | 0.000112 | **+91.0%** |

**Average: +86.6%**
**Status: ✅ SUPPORTED**

## Critical Insight: Task Structure is the Key

This confirms the finding from H1.202:
- **Without task structure**: Concatenation wins (H3.89: -30.5%, H3.90: -20.5%)
- **WITH task structure**: Attention wins (H1.202: +89.7%, H3.91: +86.6%)

**Task structure** = goal states + action outcomes

## Research Status (May 10, 2026)

| # | Hypothesis | Status | Key Finding |
|---|------------|--------|-------------|
| H1 | Unified vs Baseline | ✅ +25.6% | Early fusion wins real robot |
| H1.201 | Complex multi-step (synthetic) | ❌ -7.2% | CG loses on synthetic complex |
| H1.202 | Task structure enables attention | ✅ +89.7% | Goal states + actions |
| H3.89 | Attention long (autocorrelation) | ❌ -30.5% | Concat wins, no task structure |
| H3.90 | SSM long (autocorrelation) | ❌ -20.5% | Concat wins, no task structure |
| H3.91 | Attention 20+ WITH task structure | ✅ +86.6% | Task structure enables attention! |

**Total: 40+ SUPPORTED, 2 INCONCLUSIVE, 25+ REFUTED**

## Architecture Recommendations

1. **For manipulation with goals**: Use Attention (+86-90%)
2. **For temporal reasoning**: Use SSM (+37%)
3. **For pure prediction**: Use Concatenation
4. **For combined systems**: Task-aware routing based on goal detection

## Next Steps

1. Test hybrid architecture with task-aware routing
2. Explore Graph Neural Networks for temporal reasoning
3. Continue validation on real robot datasets (LIBERO)