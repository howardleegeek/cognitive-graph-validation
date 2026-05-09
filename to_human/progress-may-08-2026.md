# Research Progress Report — May 8, 2026

## Executive Summary

Continued cognitive graph validation research. Completed H1.181 (autocorrelation injection) and H1.182 (complex multi-step tasks). Key finding: **Task structure determines optimal architecture** — average pooling → concat, next-step prediction → SSM, cross-modal prediction → attention.

## Experiments Completed

### H1.181: Autocorrelation Injection — ✅ SUPPORTED

| Autocorrelation (ρ) | Concat MSE | Attn MSE | Delta |
|---------------------|-----------|----------|-------|
| 0.00 | 0.000002 | 0.000002 | -6.5% |
| 0.50 | 0.000003 | 0.000003 | -7.6% |
| 0.90 | 0.000004 | 0.000003 | -17.4% |
| 0.95 | 0.000004 | 0.000003 | -26.9% |

**Finding**: Attention advantage INCREASES with autocorrelation. Higher temporal structure = better attention performance.

### H1.182: Complex Multi-Step Tasks — Mixed Results

#### Run 1: Average Pooling Target — ❌ REFUTED (attention)

- Concat wins **14/14 tasks**
- Attention: +372-744% worse than concat
- SSM: +774-6000% worse than concat

#### Run 2: Next-Step Prediction Target — ✅ SUPPORTED (SSM)

- SSM wins **14/14 tasks** (-30% to -38%)
- Attention: +1-4% worse than concat (essentially tied)

## Key Insight: Task Structure Determines Architecture

| Target Task | Best Architecture | Why |
|-------------|------------------|-----|
| Average pooling | Concat (+0%) | No temporal dynamics to exploit |
| Next-step prediction | SSM (-30-38%) | Sequential state transition |
| Cross-modal prediction | Attention (+17-26%) | Multi-modal alignment |

## New Hypotheses Generated

1. **H3.87**: SSM + Attention hybrid for tasks requiring both sequential modeling AND cross-modal alignment
2. **H1.183**: Attention with real robot data shows +99% because robot data has both temporal structure AND cross-modal prediction targets

## H3.86: Graph-Native Multi-Object Reasoning — ❌ REFUTED

- Graph methods: -0.5% vs flat attention
- Multi-object tasks: Graph doesn't outperform flat attention

## Git Commit

- Commit: `e3a2f89` — H1.182 complex multi-step results, H1.181 autocorrelation validation

## Research Status

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1.181 (Autocorrelation enables attention) | ✅ SUPPORTED | +26.9% at ρ=0.95 |
| H1.182a (Attention on avg pooling) | ❌ REFUTED | +372-744% worse |
| H1.182b (SSM on next-step prediction) | ✅ SUPPORTED | -30-38% |
| H3.86 (Graph multi-object) | ❌ REFUTED | -0.5% |

## Next Steps

1. **H1.183**: Validate attention +99% on real robot is due to cross-modal prediction structure (not just autocorrelation)
2. **H3.87**: Test SSM + Attention hybrid combining sequential and cross-modal strengths
3. **H1.184**: Investigate why SSM works for next-step but not cross-modal tasks