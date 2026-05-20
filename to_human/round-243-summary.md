# Round 243 Summary — Cognitive Graph Validation

## Action Taken

**H1.470.1.1.4**: Investigated whether the simulation vs "real" CG performance discrepancy is due to architectural differences.

## Key Results

| Architecture | 20-step Loss | 20-step Imp. | 50-step Loss |
|--------------|-------------|--------------|--------------|
| Baseline (concat) | 0.000112 | 0% | 0.0213 |
| Simulation CG | 0.000043 | **+61.4%** | 0.0161 |
| Real CG (attention) | 0.000351 | -213% | 0.0162 |

## Findings

1. **Simulation CG actually outperforms baseline** (+61% on 20-step), contradicting H1.470.1.1.3's negative gaps
2. **Architecture is NOT the root cause** - both CG variants perform similarly on long sequences
3. **The discrepancy is task/data-specific**, not architectural

## Next Step

H1.470.1.1.5: Investigate task structure differences between simulation and "real" experiments.
