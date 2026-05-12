# Research Progress — May 11, 2026 (Night)

## Summary

1 new experiment completed (H3.103). Research continues with focus on ultra-long sequences.

## Key Findings

| Hypothesis | Status | Finding |
|------------|--------|---------|
| H3.104: Attention on 500-1000 steps | ❌ REFUTED | Flat wins 3/5, both +95% over concat |
| H3.103: Adaptive hierarchical on 250-400 steps | ✅ SUPPORTED | +86.7% avg, wins 3/4 lengths |
| H1.212: Hierarchical on 200-300 steps | ⚠️ PARTIAL | Wins 200-250, loses 275-300 |
| H3.102: SSM + Goal on short sequences | ❌ REFUTED | Attention dominates all lengths |

## Insights

1. **Hierarchical attention scales to 400+ steps**: Adaptive hierarchical with learned gating provides +86.7% improvement over concatenation on ultra-long sequences
2. **Sweet spot at 200-250 steps**: H1.212 shows hierarchical works best in this range
3. **Attention dominates short sequences**: H3.102 confirms attention is the right choice for ≤40 steps

## Research Trajectory

- Total experiments: 22 runs
- Supported: H1, H1.208, H1.211, H3.91-100, H3.103
- Refuted: H1.209, H1.210, H3.89-90, H3.102, H3.104
- Partial: H1.212, H3.101

## Next Experiments

| ID | Test | Priority |
|----|------|----------|
| H1.214 | Different goal representations | High |
| H3.104 | Attention on 500+ step sequences | Medium |

## Architecture Summary

**Best performing configurations:**
- Ultra-long (300-500): Combined endpoint + subgoals (+46.9%)
- Long (250-400): Adaptive hierarchical (+86.7%)
- Medium (100-200): Flat attention (+85%)
- Short (≤40): Standard attention (+99%)

**Key insight**: Different sequence lengths require different attention architectures. No single approach works across all scales.