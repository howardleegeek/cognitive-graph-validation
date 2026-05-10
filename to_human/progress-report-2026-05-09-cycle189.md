# Progress Report — Cognitive Graph Validation

**Date:** May 9, 2026  
**Research Cycle:** 189

## Executive Summary

New experiment completed:

- **H1.196**: Attention on 20-40 step sequences with next-step prediction — **REFUTED** (-37.2% avg)

## H1.196 Results

| N Steps | Concat MSE | Attention MSE | Delta |
|---------|-----------|--------------|-------|
| 20 | 0.000605 | 0.001057 | -74.7% |
| 25 | 0.000761 | 0.001099 | -44.4% |
| 30 | 0.000878 | 0.001138 | -29.6% |
| 35 | 0.000949 | 0.001145 | -20.6% |
| 40 | 0.000987 | 0.001152 | -16.7% |

**Average: -37.2%**

## Key Finding

**Attention does NOT automatically win on longer sequences.** This contradicts some earlier findings (H3.4, H3.6) that suggested attention could help on 24+ step sequences. The difference may be due to:
1. Different implementation details
2. Task formulation (next-step vs final-step prediction)
3. Data characteristics

**Concatenation remains a strong baseline for temporal sequence modeling.**

## Current Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot |
| H1.193 | ✅ SUPPORTED | SSM +97.6% on 50-step (next-step) |
| H1.195 | ❌ REFUTED | Baseline wins 20-80 steps |
| H1.196 | ❌ REFUTED | Concatenation wins 20-40 steps |
| H2 | ⚠️ INCONCLUSIVE | 1.7% difference |
| H3 | ❌ REFUTED | Concatenation wins |
| H4 | 🔸 CLOSE | 22% optimal |

## Research Trajectory

**Key Insight**: Task formulation (next-step vs final-step prediction) significantly impacts which architecture wins. SSM excels at next-step prediction, but concatenation remains strong across different sequence lengths.

**Next Steps:**
1. Explore SSM + attention hybrid architectures
2. Test on real robot data with next-step prediction
3. Investigate why earlier H3.x experiments showed attention winning on longer sequences

## Files Modified

- `findings.md` — Added H1.196 results
- `research-state.yaml` — Updated hypothesis status
- `experiments/H1.196-attention-long-sequences/` — New experiment completed

## Git Status

Committed and pushed to GitHub (commit 3090f1d).