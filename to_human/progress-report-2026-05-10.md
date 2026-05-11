# Research Progress Report - May 10, 2026

## Summary

Today's experiments (H1.201, H3.89, H3.90) revealed a **critical insight** about the gap between synthetic and real robot data.

## Key Findings

### H1.201: Ultra-Complex Multi-Step with Real Robot Temporal Dynamics
- **Status**: ❌ REFUTED (-7.2% average)
- Cognitive Graph loses on synthetic complex multi-step tasks
- Trend is GROWING (+3.5% at 75 steps) but from negative base

### H3.89: Attention on Longer Sequences with Autocorrelation
- **Status**: ❌ REFUTED (-30.5% average)
- Concatenation wins across 6/7 tested sequence lengths
- Only wins at 10 steps (+10.7%)

### H3.90: SSM on Long Sequences with Autocorrelation  
- **Status**: ❌ REFUTED (-20.5% SSM, -19.8% Attention)
- Concatenation wins across all 7 tested sequence lengths
- SSM wins: 0/7

## Critical Insight: Synthetic vs Real Robot Gap

| Setting | H1.193/H1.182 | H1.201/H3.89/H3.90 |
|---------|---------------|-------------------|
| Data | Real robot | Synthetic |
| SSM | +97.6% | -20.5% |
| Attention | Varies | -19.8% to -30.5% |

**The Gap**: Real robot manipulation has **task structure** (goal states, action outcomes) that enables SSM/Attention to exploit. Pure sequence prediction lacks this structure.

## Research Status (May 10, 2026)

| # | Hypothesis | Status | Key Finding |
|---|------------|--------|-------------|
| H1 | Unified vs Baseline | ✅ +25.6% | Early fusion wins real robot |
| H1.201 | Complex multi-step (synthetic) | ❌ -7.2% | CG loses on synthetic complex |
| H3.89 | Attention long (autocorrelation) | ❌ -30.5% | Concat wins, attention loses |
| H3.90 | SSM long (autocorrelation) | ❌ -20.5% | Concat wins, SSM loses |
| H1.193 | SSM +97.6% | ✅ | Only valid for manipulation tasks |
| H2.x | Graph structure | ✅ | +56-75% on temporal |

**Total: 40+ SUPPORTED, 2 INCONCLUSIVE, 25+ REFUTED**

## Recommendations

1. **For manipulation tasks**: Use SSM/Attention with real robot data
2. **For pure prediction tasks**: Use concatenation  
3. **For combined tasks**: Hybrid architecture with task-aware routing

## Next Steps

1. Test hybrid architecture with task-aware routing
2. Explore Graph Neural Networks for temporal reasoning (H2.3 showed +56.8%)
3. Continue validation on real robot datasets (LIBERO)
