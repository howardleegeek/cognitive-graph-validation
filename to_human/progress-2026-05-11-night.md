# Research Progress — May 11, 2026

## Summary

4 new experiments completed today (H1.209-211, H3.101). Committed to GitHub.

## Key Findings

| Hypothesis | Status | Finding |
|------------|--------|---------|
| H1.209: Hierarchical on 100-200 steps | ❌ REFUTED | Flat wins (3/5), hierarchical only helps at 300-500 |
| H3.101: SSM + Goal conditioning | ⚠️ PARTIAL | +3.0% vs Attn, but -1.8% vs vanilla SSM |
| H1.210: Bidirectional goal prediction | ❌ REFUTED | -1.9% vs goal-conditioned unidirectional |
| H1.211: Hierarchical + Bidirectional | ✅ SUPPORTED | +0.9% marginal on 200-400 steps |

## Insights

1. **Hierarchical is scale-dependent**: Works at 300-500 steps but NOT at 100-200
2. **Goal conditioning conflicts with SSM**: SSM's sequential dynamics clash with goal conditioning
3. **Bidirectional redundant**: Goal conditioning already captures backward temporal info
4. **Sweet spot**: 200-300 steps for combined hierarchical + bidirectional

## Next Experiments

| ID | Test | Priority |
|----|------|----------|
| H1.212 | Hierarchical on 200-300 step range | High |
| H3.102 | SSM + Goal at short sequences (≤40) | Medium |

## Research Trajectory

- Total experiments: 16 runs
- Supported: H1, H1.208, H1.211, H3.91-100
- Refuted: H1.209, H1.210, H3.89-90
- Partial: H3.101