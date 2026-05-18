# Round 166 Summary

## Action Taken

**H1.395 - Protocol Standardization**: Resolved the discrepancy between H1.393 (which showed CG winning at medium complexity) and H1.394 (which showed CG losing everywhere) by running both experiment styles with identical seeds (42), data generation, and training parameters (20 epochs).

## Key Results

| Style | Correlation | Avg Improvement | CG Wins |
|-------|-------------|-----------------|---------|
| H1.393 (7 configs) | -0.621 | -3.2% | 1/7 |
| H1.394 (8 configs) | -0.506 | -5.0% | 1/8 |
| UNIFIED (10 configs) | -0.552 | -4.5% | 1/10 |

**Conclusion**: DISCREPANCY_RESOLVED. Both styles now show similar negative correlations, confirming that the original H1.393 positive correlation was a seed artifact. CG only shows slight advantage (+0.7%) at complexity=100, and underperforms baseline across all other complexity levels.

## Next Action

H1.396 - Architecture tuning: Investigate why CG underperforms baseline - adjust architecture parameters or try different task types.
