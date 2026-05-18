# Round 170 Summary

## Action Taken

Ran **H1.401: Dimensionality Ratio Deep-Dive** — swept dim_ratio from 0.1 to 0.9 to test if this is the true moderator of CG advantage (following up on H1.400's outlier finding of 46.6% at dim_ratio=0.7).

## Key Finding: H1.400 CONTRADICTED

**Results**: CG loses to baseline across ALL 9 dim_ratio values tested:
- Best: dim_ratio=0.1 → -2.3% improvement
- Worst: dim_ratio=0.8 → -15.9% improvement
- Correlation: r = -0.501 (more physical dims = worse)

This directly contradicts H1.400's claim that "CG wins 100% of the time across ALL 96 configurations" with +14.2% average advantage.

## Interpretation

The CG architecture does NOT have a universal advantage. Its performance depends critically on:
1. **Data structure**: CG may only outperform when there's genuine cross-modal coupling
2. **Training dynamics**: The GNN + attention may need more epochs or different hyperparameters
3. **Task complexity**: Simple linear tasks favor the simpler baseline

H1.400's findings are now in question — the discrepancy needs investigation.

## Next Action

H1.402: Investigate the H1.400 vs H1.401 discrepancy — either replicate H1.400's data generation to verify those claims, or test with longer training (CG may need more epochs to show advantage).
