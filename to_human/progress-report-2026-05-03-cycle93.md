# Progress Report — May 3, 2026 (Cycle 93)

## Executive Summary

**Status**: Research continuing — H1.109: Complex compositional multi-step tasks **SUPPORTED** (+77.6%).

## Current Cycle: 93

### Latest Experiment

| Hypothesis | Result | Status |
|------------|--------|--------|
| H1.109 (Complex compositional multi-step) | +77.6% | ✅ SUPPORTED |

### H1.109 Results

- **Task**: 20-40 step tasks with 4 compositional subtasks
- **Best model**: Unified+SSM
- **Improvement**: +77.6% average over baseline

| Task Length | Baseline MSE | Unified+SSM MSE | Improvement |
|-------------|-------------|-----------------|-------------|
| 20-step | 0.0145 | 0.0029 | +80.3% |
| 30-step | 0.0126 | 0.0030 | +76.6% |
| 40-step | 0.0119 | 0.0029 | +76.0% |

## Research Status

- **Total SUPPORTED**: 27+
- **INCONCLUSIVE/MARGINAL**: 3
- **REFUTED**: 13

### Core Hypotheses

| Hypothesis | Status | Key Finding |
|------------|--------|-----------|
| H1: Unified vs Baseline | ✅ | +25.6% on real robot |
| H1.109: Complex compositional | ✅ | +77.6% on 20-40 step |
| H2: Graph structure | ✅ | +56-75% on temporal |
| H3: Attention | ❌ | Concat wins (simple), ⚠️ (complex) |
| H4: Dimension 22% | ✅ | 22-25% optimal |

## Key Findings This Cycle

1. **Unified+SSM excels on complex tasks**: +77.6% improvement on 20-40 step compositional tasks
2. **SSM adds +5.5%** over standard unified architecture
3. **Attention underperforms** on this task type (+37.0% vs +77.6% for SSM)
4. **Advantage decreases slightly with length**: 80% (20-step) → 76% (40-step)

## Next Steps

1. Test H3.34: Attention on longer sequences (20+ timesteps) with SSM-style gating
2. Validate H1.109 on real robot data
3. Paper consolidation for ICRA/RSS

## Files

- `findings.md`: Full research findings (2980+ lines)
- `research-state.yaml`: Hypothesis tracking (950+ lines)
- `experiments/H1.109-complex-compositional-multi-step/`: New experiment

## Git Commit

```
Cycle 93: H1.109 Complex compositional multi-step - SUPPORTED (+77.6%)
```

Pushed to: https://github.com/howardleegeek/cognitive-graph-validation