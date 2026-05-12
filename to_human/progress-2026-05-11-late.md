# Research Progress Report — May 11, 2026 (Late Night)

## Session Summary

**Experiments Run**: 10 (H3.105-H3.112, H1.215-H1.216)
**New Results**: 0 SUPPORTED, 10 REFUTED, 1 INCONCLUSIVE

## Critical Finding

**Attention mechanisms consistently FAIL on manipulation tasks**

Across 10 new experiments, attention never provided significant improvements over concatenation:

| Experiment | Avg Delta | Status |
|------------|-----------|--------|
| H3.105 (Task structure) | -93.1% | REFUTED |
| H3.106 (Phase transitions) | -72.0% | REFUTED |
| H3.107 (Next-step pred) | -45.8% | REFUTED |
| H3.108 (Neural attention) | -1.5M% | REFUTED |
| H3.109 (Robot structure) | -16.1% | REFUTED |
| H3.110 (Learned patterns) | -88.8% | REFUTED |
| H3.111 (Data comparison) | Mixed | INCONCLUSIVE |
| H3.112 (SSM-CG) | -14.9% | REFUTED |
| H1.215 (Multi-step) | -4.5% | REFUTED |
| H1.216 (Hierarchical) | -507.3% | REFUTED |

## Key Insights

1. **Task structure doesn't enable attention**: Phase transitions, causal structure, goal conditioning all failed
2. **Neural attention diverges**: Training attention from scratch leads to catastrophic failure
3. **High variance**: SSM-CG showed wild swings (-56% to +8%) indicating instability
4. **Simple baselines win**: Concatenation + MLP remains the most reliable approach

## What's Working

- **Concatenation baselines**: Stable, consistent, reliable
- **Simple MLPs**: Outperform complex graph/attention architectures
- **Endpoint goals**: +95% improvement over trajectory goals (H1.214)

## What to Try Next

1. **Mamba/SSM baselines** without attention mechanisms
2. **Longer training** with better regularization
3. **Real robot data** validation (LIBERO dataset)
4. **Different architecture families**: Transformers vs. SSMs vs. LSTMs

## Git Commit

- Commit: 87449ee (evening progress)
- New Commit: [current session]

## Recommendations

Given the consistent failures, the research direction should shift:
- Stop testing attention on manipulation tasks
- Focus on understanding WHY concatenation works so well
- Explore SSM architectures WITHOUT attention heads
- Consider that this may be a data problem, not an architecture problem