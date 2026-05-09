# Progress Report - Cycle 173 (May 8, 2026)

## Executive Summary

**2 new hypotheses tested. 1 REFUTED, 1 INCONCLUSIVE.**

Key finding: **Synthetic data doesn't transfer to real robot behavior.** H1.178 (+98.4% on synthetic) fails on synthetic real-robot simulation (-13.4%).

## Results

### H1.179: Adaptive Decay on Real Robot Data (Synthetic)

| Method | MSE | Improvement |
|--------|-----|-------------|
| Concatenation | 0.0418 | baseline |
| Fixed Decay 0.99 | 0.0473 | -13.4% |
| Adaptive Decay | 0.0473 | -13.4% |
| Multi-Scale | 0.0474 | -13.5% |

**Status: ❌ REFUTED** (-13.4%)

**Key insight**: H1.171 achieved +18.6% on actual real robot data. H1.179 achieves -13.4% on synthetic simulation. The gap between actual and simulated robot data is critical.

### H3.85: Multi-Object Attention Investigation

| Method | MSE | Improvement |
|--------|-----|-------------|
| Concatenation | 0.9996 | baseline |
| Multi-Scale (H3.83) | 1.3507 | -35.1% |
| Weighted Attn (H3.84) | 1.3738 | -37.4% |
| Object-Conditioned | 1.7598 | -76.1% |

**Status: ⚠️ INCONCLUSIVE** — No attention method works on multi-object tasks.

**Key insight**: H3.83 (-47.0%) and H3.84 (+25.2%) used different task configurations. In controlled comparison, both attention methods fail on multi-object tasks.

## Research Status

| Category | Count |
|----------|-------|
| SUPPORTED | 63 |
| INCONCLUSIVE | 4 |
| REFUTED | 24 |
| PENDING | 0 |

## Critical Gaps Identified

1. **Real vs Synthetic Gap**: Actual robot data (H1.171: +18.6%) vs synthetic simulation (H1.179: -13.4%)
2. **Multi-Object Attention**: All attention methods fail on multi-object tasks
3. **Task Configuration Differences**: H3.83 vs H3.84 discrepancy explained by task differences, not method differences

## Next Steps

1. **Obtain real robot data** — Critical for validation
2. **Focus on single-object tasks** — Multi-object attention unsolved
3. **Paper writing** — Compile 63+ supported hypotheses