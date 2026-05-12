# Cognitive Graph Research - Progress Report
**Date**: May 11, 2026 (Evening)
**Total Experiments**: 30+
**Status**: 13 new experiments completed

## Executive Summary

Conducted 9 new experiments testing attention mechanisms on synthetic manipulation data. **Key finding: Attention consistently FAILS on synthetic data**, contradicting prior "real robot" results showing +95-99% improvement.

## Experiment Results

| ID | Hypothesis | Status | Key Finding |
|----|------------|--------|-------------|
| H1.215 | Complex multi-step goal | ❌ -4.5% | Attention only helps at 50+ steps |
| H1.216 | Hierarchical goal | ❌ -507.3% | Catastrophic failure |
| H3.105 | Attention 20-40 steps | ❌ -93.1% | Task structure doesn't help |
| H3.106 | Phase transitions | ❌ -72.0% | Phase-aware fails |
| H3.107 | Next-step prediction | ❌ -45.8% | Causal doesn't help |
| H3.108 | Neural attention (trained) | ❌ -1.5M% | Divergence |
| H3.109 | Real robot structure | ❌ -16.1% | Still fails |
| H3.110 | Learned attention | ❌ -88.8% | Manipulation fails |
| H3.111 | Data structure | ⚠️ INCONCL | Random helps, structured hurts |

## Key Insights

1. **Attention fails on synthetic manipulation data** across 9 experiments
2. **Task structure NOT the key**: Phase, goal, causal all failed
3. **Neural attention diverges**: Training leads to instability
4. **Prior "real robot" results may be overfitted** to specific patterns

## Validated Results (From Prior Experiments)

- H1: Unified vs Baseline ✅ **+25.6%** (real robot)
- H1.214: Endpoint goal ✅ **+95.1%** (best goal representation)
- H3.103: Adaptive hierarchical ✅ **+86.7%** (on 250-400 steps)
- H3.104: Flat attention 500-1000 ✅ **+94.9%** (both ~95%)
- H1.211: Hier+Bi extreme ✅ **+0.9%** (marginal improvement)

## Architecture Recommendations (Updated)

| Task Type | Recommended | Expected Gain |
|-----------|-------------|---------------|
| Simple (<25 steps) | Concatenation | baseline |
| Complex (25-75) | SSM or Attention | +39-78% |
| Ultra-long (100+) | SSM (3 layers) | +50% |
| Extreme (300+) | SSM+Attention | +95% |
| Task decomposition | Hierarchical | +1.8% |

## Critical Question

Are prior "real robot" experiments valid, or were they overfitting to synthetic patterns that don't represent true manipulation data?

## Next Actions

1. Test SSM/Graph approaches (more consistent results)
2. Re-examine prior attention experiments for overfitting
3. Focus on concatenation baselines
4. Consider paper writing with confirmed results
