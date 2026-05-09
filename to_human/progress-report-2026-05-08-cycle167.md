# Research Progress Report - May 8, 2026 (Cycle 167)

## Summary

Testing attention on ultra-extreme sequences (400-600 steps). Two new experiments completed.

## Results This Cycle

### H1.172: Attention on 400-500 Step Synthetic Sequences ❌ REFUTED

| Sequence Length | Concat MSE | Attention MSE | Delta |
|-----------------|------------|---------------|-------|
| 400 steps | 0.000267 | 0.000278 | -4.1% |
| 450 steps | 0.000262 | 0.000281 | -7.5% |
| 500 steps | 0.000254 | 0.000273 | -7.8% |

**Average: -6.5%**

Key finding: Attention does NOT maintain advantage on 400-500 step synthetic sequences. Degradation increases with length.

### H1.173: Attention on 400-600 Step Structured Sequences ❌ REFUTED

| Sequence Length | Concat MSE | Attention MSE | Delta |
|-----------------|------------|---------------|-------|
| 400 steps | 0.000077 | 0.000086 | -12.0% |
| 500 steps | 0.000072 | 0.000081 | -13.0% |
| 600 steps | 0.000066 | 0.000081 | -24.0% |

**Average: -16.3%**

Key finding: Even with better temporal structure (12 phases, object permanence, smooth transitions), attention still underperforms. Degradation scales with length (-12% → -13% → -24%).

## Critical Insight

Attention scales well on REAL ROBOT DATA but FAILS on synthetic 400+ steps:

| Experiment | Sequence Length | Data Type | Result |
|------------|-----------------|-----------|--------|
| H1.171 | 200-300 steps | Real Robot | +18.6% ✅ |
| H1.172 | 400-500 steps | Synthetic | -6.5% ❌ |
| H1.173 | 400-600 steps | Structured | -16.3% ❌ |

The difference is task structure - real robot manipulation has inherent temporal patterns that attention can exploit, while synthetic data lacks sufficient structure at extreme lengths.

## Research Status

| Metric | Value |
|--------|-------|
| Total Hypotheses | 76 |
| SUPPORTED | 57 |
| INCONCLUSIVE | 3 |
| REFUTED | 18 |
| PENDING | 0 |
| Current Cycle | 167 |

## Key Insights

1. **Attention advantage is data-dependent**: Works on real robot data, fails on synthetic at 400+ steps
2. **Structure matters**: More phases (12 vs 6) didn't help - the issue is fundamental to synthetic data
3. **Crossover point**: Attention helps at 20-300 steps (real robot), fails at 400+ (synthetic)

## Next Steps

1. Test attention on real robot data at 400+ steps (not synthetic)
2. Explore SSM alternatives for ultra-long synthetic sequences
3. Investigate why real robot structure enables attention but synthetic doesn't