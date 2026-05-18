# Round 165 Summary: Quadratic Complexity Relationship Investigation

**Experiment**: H1.394 - Tested whether CG advantage follows an inverted-U (quadratic) relationship with task complexity.

**Method**: Ran 16 experiments across 8 complexity levels (50-600) with 2 seeds each. Fit both linear and quadratic models to CG advantage vs complexity.

**Key Results**:
- Quadratic model fits significantly better than linear (ΔAIC = -7.15, ΔR² = +0.23)
- Peak CG advantage at complexity ~215 (higher than predicted 150-170)
- **Concern**: CG underperformed baseline across ALL complexity levels (-21% to -38%)

**Conclusion**: PARTIALLY_SUPPORTED. The inverted-U pattern is confirmed by model comparison, but the peak location differs from H1.393 prediction. More concerning: H1.394 shows CG losing everywhere while H1.393 showed CG winning at medium complexity.

**Next Action**: H1.395 will standardize the experimental protocol between H1.393 and H1.394 to resolve this critical discrepancy. The difference may be due to training epochs (20 vs 50), data generation methods, or other protocol variations.

**Status**: One experiment completed, one sub-hypothesis tested with concrete numerical results.