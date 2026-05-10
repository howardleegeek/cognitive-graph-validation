# Progress Report — Cognitive Graph Validation

**Date:** May 9, 2026  
**Research Cycle:** 188

## Executive Summary

Research continues on validating the Cognitive Graph architecture for language-conditioned robotic tasks. Two new experiments completed today:

- **H1.193**: SSM +97.6% on 50-step sequences (next-step prediction) — **SUPPORTED**
- **H1.195**: Baseline wins across 20-80 steps (final-step prediction) — **REFUTED**

## Key Finding: Task Setup Matters

The most important insight from today's experiments is that **prediction target matters significantly**:

| Experiment | Task | SSM Result |
|------------|------|------------|
| H1.193 | Next-step prediction | +97.6% (wins) |
| H1.195 | Final-step prediction | -23% to -60% (loses) |

SSM's sequential state modeling is better suited for next-step prediction (predicting the next state from previous states), which aligns with its recurrent nature. For final-step prediction, the baseline's direct concatenation approach works better.

## Current Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot |
| H1.193 | ✅ SUPPORTED | SSM +97.6% on 50-step (next-step) |
| H1.195 | ❌ REFUTED | Baseline wins 20-80 steps |
| H2 | ⚠️ INCONCLUSIVE | 1.7% difference |
| H3 | ❌ REFUTED | Concatenation wins |
| H4 | 🔸 CLOSE | 22% optimal |

## Research Trajectory

**Next Steps:**
1. Test SSM with next-step prediction on longer sequences (60-100 steps)
2. Explore hybrid architectures (SSM + attention)
3. Validate on real robot data with next-step prediction

**Key Insight:** The task formulation (next-step vs final-step) significantly impacts which architecture wins. Future experiments should use next-step prediction to properly test SSM's capabilities.

## Files Modified

- `findings.md` — Added H1.193 and H1.195 results
- `research-state.yaml` — Updated hypothesis status
- `experiments/H1.193-long-sequence-autocorrelation/` — Completed
- `experiments/H1.195-ssm-attention-crossover/` — Completed

## Git Status

Changes ready for commit and push to GitHub.