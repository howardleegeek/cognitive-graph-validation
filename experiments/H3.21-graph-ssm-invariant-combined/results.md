# H3.21 Results: Graph + SSM + Invariant Combined

## Summary

| Metric | Value |
|--------|-------|
| Temporal MSE | 2.0339 |
| Temporal Baseline | 1.3840 |
| Temporal Improvement | **-47.0%** |
| Transfer MSE | 1.1718 |
| Transfer Baseline | 1.3173 |
| Transfer Improvement | **+11.0%** |
| Combined Score | **-18.0%** |

## Analysis

The combined architecture shows **mixed results**:
- **Temporal**: Significantly worse than baseline (-47%)
- **Transfer**: Better than baseline (+11%)

This suggests that the SSM+Invariant combination doesn't work well for temporal reasoning in this synthetic setting. The model may be overfitting to the transfer task at the expense of temporal performance.

## Status

**REFUTED** for combined temporal+transfer performance. The architecture doesn't achieve both goals simultaneously in this synthetic setting.

## Comparison with Prior Results

| Experiment | Temporal | Transfer | Combined |
|------------|----------|----------|----------|
| H3.17 (Graph+SSM) | +25% | N/A | N/A |
| H3.14 (SSM+Invariant) | +7.3% | -2.3% | Partial |
| H1.8 (Invariant alone) | N/A | +5.4% | N/A |
| **H3.21 (Combined)** | **-47%** | **+11%** | **-18%** |

## Conclusion

The combined architecture doesn't solve both problems simultaneously. The temporal degradation suggests that the invariant learning component interferes with temporal reasoning in this simplified model.

**Recommendation**: Keep separate architectures for temporal (Graph+SSM) and transfer (Invariant) tasks rather than combining them.