# Progress Report — Cognitive Graph Validation

**Date:** May 9, 2026  
**Research Cycle:** 190

## Executive Summary

New experiment completed:

- **H1.197**: SSM + Attention Hybrid on Complex Multi-Step Tasks — **REFUTED** (+0.0% avg)

## H1.197 Results

| N Steps | Concat MSE | SSM MSE | Attention MSE | Hybrid MSE | Best |
|---------|-----------|---------|---------------|------------|------|
| 30 | 0.005402 | 0.008213 | 0.006309 | 0.006397 | concat |
| 40 | 0.001329 | 0.005618 | 0.005542 | 0.005309 | concat |
| 50 | 0.003718 | 0.005072 | 0.005317 | 0.004983 | concat |
| 60 | 0.003306 | 0.004377 | 0.004532 | 0.004797 | concat |

**Average: +0.0%** — Concatenation wins on all complex multi-step tasks.

## Key Finding

**SSM+Attention hybrid does NOT transfer to complex multi-step tasks.** Even after H1.193 showed SSM achieving +97.6% on 50-step sequences, the hybrid architecture fails to outperform concatenation on 30-60 step complex tasks.

This confirms a consistent pattern:
- H1.196: Concatenation wins on 20-40 steps (-37.2%)
- H1.197: Concatenation wins on 30-60 steps (+0.0%)

**Concatenation remains the strongest baseline for temporal sequence modeling.**

## Current Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot |
| H1.193 | ✅ SUPPORTED | SSM +97.6% on 50-step (next-step) |
| H1.195 | ❌ REFUTED | Baseline wins 20-80 steps |
| H1.196 | ❌ REFUTED | Concatenation wins 20-40 steps |
| H1.197 | ❌ REFUTED | Concatenation wins 30-60 steps |
| H2 | ⚠️ INCONCLUSIVE | 1.7% difference |
| H3 | ❌ REFUTED | Concatenation wins |
| H4 | 🔸 CLOSE | 22% optimal |

## Research Trajectory

**Key Insight**: The success of SSM on H1.193 (next-step prediction with autocorrelation) does NOT generalize to complex multi-step tasks. Concatenation remains the strongest baseline across different sequence lengths and task complexities.

**Next Steps:**
1. Investigate why SSM works on next-step but not complex multi-step
2. Test SSM with different task formulations
3. Explore whether the issue is implementation or fundamental

## Files Modified

- `findings.md` — Added H1.197 results
- `research-state.yaml` — Updated hypothesis status
- `experiments/H1.197-ssm-attention-complex-multistep/` — New experiment completed

## Git Status

Changes ready to commit and push.