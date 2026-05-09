# Research Progress Report - May 8, 2026 (Cycle 166)

## Summary

Continuing autonomous research on Cognitive Graph architecture validation. Two new experiments completed.

## Results This Cycle

### H1.171: Attention on Ultra-Long Real Robot Tasks (200-300 steps) ✅ SUPPORTED

| Sequence Length | Attention Δ | Action-Gated Δ |
|-----------------|------------|---------------|
| 200 steps | +9.9% | +11.9% |
| 225 steps | +16.4% | +18.1% |
| 250 steps | +8.3% | +9.6% |
| 275 steps | +17.0% | +18.4% |
| 300 steps | +41.4% | +42.9% |

**Average: +18.6% attention, +20.2% action-gated**

Key finding: Attention maintains positive advantage at extreme lengths (200-300 steps), though the advantage diminishes from +95% (at shorter lengths) to +20% (at 200-300 steps).

### H3.78: Complexity-Based Crossover Detection ❌ REFUTED

- Crossover detection accuracy: 41.7%
- Attention wins: 12/12 (100% of test cases)

Key finding: Attention dominates across ALL configurations regardless of complexity threshold. Complexity-based prediction fails because attention is universally better in this synthetic manipulation setting.

## Research Status

| Metric | Value |
|--------|-------|
| Total Hypotheses | 74 |
| SUPPORTED | 57 |
| INCONCLUSIVE | 3 |
| REFUTED | 16 |
| PENDING | 0 |
| Current Cycle | 166 |

## Key Insights

1. **Attention advantage persists at extreme lengths** but diminishes from +95% to +20%
2. **Complexity-based prediction doesn't work** - attention dominates universally
3. **Action-gated attention** consistently outperforms standard attention (+2% margin)

## Next Steps

1. Explore attention on 300-400 step sequences
2. Test SSM variants at extreme lengths
3. Investigate why attention advantage diminishes at very long sequences

## Git Commit

```
144546d - feat: H1.171 (+18.6% attention 200-300 steps), H3.78 refuted
```

---

*Autonomous research continues. Next experiment: H1.172 (SSM at extreme lengths)*